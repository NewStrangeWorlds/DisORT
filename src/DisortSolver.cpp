#include "DisortSolver.hpp"
#include "BandSolver.hpp"
#include "Planck.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace disortpp {

// Forward declaration of file-static Chapman function helper (defined later)
static double chapmanImpl(int lc, double tau_frac,
              const std::vector<double>& level_altitudes,
              const std::vector<double>& dtau_lyr,
              int num_layers, double r, double direct_beam_mu);

// ============================================================================
// Main Solve Method
// ============================================================================

DisortResult DisortSolver::solve(DisortConfig& config) 
{
  // Store configuration reference
  config_ = &config;

  // Auto-allocate arrays if the caller skipped allocate()
  if (config.delta_tau.empty()) {
    config.allocate();
  }

  // Validate configuration
  config.validate();

  // Extract dimensions
  nlyr_ = config.num_layers;
  nstr_ = config.num_streams;
  nn_ = nstr_ / 2;

  // Set num_user_tau if not specified by user (output at layer boundaries)
  if (!config.flags.use_user_tau) {
    config.num_user_tau = nlyr_ + 1;
    // Reallocate tau_user with correct size
    config.tau_user.resize(nlyr_ + 1);
  }
  ntau_ = config.num_user_tau;

  // SPECIAL_BC: double num_user_mu so both positive and negative user angles are included
  // (mirrors C code lines 437-439: "if (use_user_mu && SPECIAL_BC) num_user_mu *= 2")
  const bool is_special_bc = (config.flags.ibcnd == BoundaryConditionType::Special);
  if (is_special_bc && config.flags.use_user_mu) {
    config.num_user_mu *= 2;
    config.mu_user.resize(config.num_user_mu + 1);
  }

  // Set num_user_mu if not specified by user (use computational angles)
  if (!config.flags.use_user_mu || config.flags.comp_only_fluxes) {
    config.num_user_mu = nstr_;
    // Reallocate mu_user with correct size
    config.mu_user.resize(nstr_);
  }
  numu_ = config.num_user_mu;

  nphi_ = config.num_phi;

  // Determine if scattering is present
  scat_yes_ = false;
  for (int lc = 0; lc < nlyr_; ++lc) {
    if (config.single_scat_albedo[lc] > 0.0) {
      scat_yes_ = true;
      break;
    }
  }

  // Determine if delta-M transformation is needed
  deltam_ = true;
  bool needdeltam = false;
  if (deltam_) {
    for (int lc = 0; lc < nlyr_; ++lc) {
      if (config.phaseFunctionMoments(nstr_, lc) != 0.0) {
        needdeltam = true;
        break;
      }
    }
    if (!needdeltam) {
      deltam_ = false;
    }
  }

  // Determine if Delta-M+ should be used (Lin et al. 2018)
  deltam_plus_ = config.flags.use_delta_m_plus;
  if (deltam_plus_) {
    // Global fallback: check ALL layers (Fortran DISORT lines 482-498)
    for (int lc = 0; lc < nlyr_; ++lc) {
      double pm  = config.phaseFunctionMoments(nstr_, lc);
      double pm1 = config.phaseFunctionMoments(nstr_ + 1, lc);
      if (pm < 1e-4 || pm1 < 0.7 * pm) {
        deltam_plus_ = false;
        break;
      }
    }
    if (deltam_plus_) {
      deltam_ = false;  // Delta-M+ replaces standard Delta-M
    } else {
      deltam_ = needdeltam;  // Fall back to standard Delta-M
    }
  }

  // Allocate working arrays — skip if dimensions are identical to last call
  const bool compute_intensity = !config.flags.comp_only_fluxes;
  const bool dims_changed = (nlyr_  != alloc_nlyr_  ||
                 nstr_  != alloc_nstr_  ||
                 numu_  != alloc_numu_  ||
                 ntau_  != alloc_ntau_  ||
                 is_special_bc        != alloc_special_bc_        ||
                 compute_intensity    != alloc_compute_intensity_);
  if (dims_changed) {
    allocateWorkingArrays();
    result_.allocate(config.num_user_tau, config.num_user_mu, config.num_phi, config.num_streams,
            is_special_bc, compute_intensity, false /* output_fourier_expansion */);
    alloc_nlyr_              = nlyr_;
    alloc_nstr_              = nstr_;
    alloc_numu_              = numu_;
    alloc_ntau_              = ntau_;
    alloc_special_bc_        = is_special_bc;
    alloc_compute_intensity_ = compute_intensity;
  } else {
    // Dimensions unchanged: zero result in-place to avoid stale data
    std::fill(result_.flux_direct_beam.begin(), result_.flux_direct_beam.end(), 0.0);
    std::fill(result_.flux_down.begin(), result_.flux_down.end(), 0.0);
    std::fill(result_.flux_up.begin(), result_.flux_up.end(), 0.0);
    std::fill(result_.flux_tau_divergence.begin(), result_.flux_tau_divergence.end(), 0.0);
    std::fill(result_.mean_intensity.begin(), result_.mean_intensity.end(), 0.0);
    std::fill(result_.mean_intensity_down.begin(), result_.mean_intensity_down.end(), 0.0);
    std::fill(result_.mean_intensity_up.begin(), result_.mean_intensity_up.end(), 0.0);
    std::fill(result_.mean_intensity_direct_beam.begin(), result_.mean_intensity_direct_beam.end(), 0.0);
    std::fill(result_.mu_angles.begin(), result_.mu_angles.end(), 0.0);
    for (auto& level : result_.intensity)
      for (auto& angles : level)
        std::fill(angles.begin(), angles.end(), 0.0);
    for (auto& level : result_.intensity_azimuth_avg)
      std::fill(level.begin(), level.end(), 0.0);
    if (!result_.albedo_medium.empty())
      std::fill(result_.albedo_medium.begin(), result_.albedo_medium.end(), 0.0);
    if (!result_.transmissivity_medium.empty())
      std::fill(result_.transmissivity_medium.begin(), result_.transmissivity_medium.end(), 0.0);
  }

  // Use the pre-allocated result_ throughout
  DisortResult& result = result_;

  // Setup computational mesh and apply delta-M transformation
  disortSet(config, deltam_);

  // Store the polar angle cosines that intensities are returned at.
  // After disortSet(), config.mu_user contains the final angles
  // (user-specified or computational quadrature angles).
  for (int iu = 0; iu < numu_; ++iu) {
    result.mu_angles[iu] = config.mu_user[iu];
  }

  // ========================================================================
  // SPECIAL_BC: Albedo and Transmissivity of Entire Medium
  // ========================================================================
  if (is_special_bc) {
    computeAlbTrans(config, result);
    return result;
  }

  // Precompute Planck functions for thermal emission boundaries (mirrors C lines 692-702)
  planck_bottom_ = 0.0;
  planck_top_ = 0.0;
  if (config.flags.use_thermal_emission) {
    planck_top_ = planckFunction2(config.wavenumber_low, config.wavenumber_high, config.bc.temperature_top) * config.bc.emissivity_top;
    planck_bottom_ = planckFunction2(config.wavenumber_low, config.wavenumber_high, config.bc.temperature_bottom);
  }

  const bool do_user_intensities = !config.flags.comp_only_fluxes && config.flags.use_user_mu;

  // ========================================================================
  // Main Solution Loop: Azimuthal Fourier Decomposition
  // ========================================================================
  //
  // The radiative field is expanded in azimuthal Fourier modes:
  //   I(τ, μ, φ) = Σ (2-δ₀ₘ) Iₘ(τ, μ) cos[m(φ - φ₀)]
  //
  // For each mode m, we solve the 1D radiative transfer problem
  // and accumulate contributions to get the full 3D solution.
  //
  // Determine maximum azimuthal mode needed (naz).
  // Mirrors C code lines 708-721 in cdisort.c.
  //
  // Azimuth-independent case: only m=0 is needed when:
  //  - No incident beam (direct_beam_flux == 0)
  //  - Beam is near-vertical (|1 - direct_beam_mu| < 1e-5)
  //  - Only fluxes requested (comp_only_fluxes=true)
  //  - Single user angle mu=1 or mu=-1 (degenerate)
  //  - Two user angles mu=-1, mu=+1 (degenerate)
  int naz = nstr_ - 1;
  {
    const double direct_beam_flux = config.bc.direct_beam_flux;
    const double direct_beam_mu  = config.bc.direct_beam_mu;
    const bool   comp_only_fluxes = config.flags.comp_only_fluxes;
    bool azimuth_independent =
      (direct_beam_flux == 0.0)                                               ||
      (std::abs(1.0 - direct_beam_mu) < 1.e-5)                              ||
      comp_only_fluxes                                                        ||
      (numu_ == 1 && std::abs(1.0 - config.mu_user[1]) < 1.e-5)       ||
      (numu_ == 1 && std::abs(1.0 + config.mu_user[1]) < 1.e-5)       ||
      (numu_ == 2 && std::abs(1.0 + config.mu_user[1]) < 1.e-5 &&
               std::abs(1.0 - config.mu_user[2]) < 1.e-5);
    if (azimuth_independent) {
      naz = 0;
    }
  }

  // Loop over azimuthal modes (m = 0 to naz)
  int kconv = 0;  // Azimuthal convergence counter
  for (int azimuth_mode = 0; azimuth_mode <= naz; ++azimuth_mode) {
    double kronecker_mode_0 = (azimuth_mode == 0) ? 1.0 : 0.0;

    // Compute Legendre polynomials at computational angles for this azimuthal mode.
    // legendre_quad_(l, jq) = P_l^m(quad_angle_[jq]) — only depends on azimuth_mode, not on layer.
    // Computing once here avoids num_layers redundant calls inside solveEigen.
    legendrePolynomialsFlat(nstr_, azimuth_mode, nstr_ - 1, nstr_ - 1, cmu_vec_.data(), legendre_quad_.data());

    // Pre-compute ylmc_cwt(l, jq) = legendre_quad_(l, jq) * quad_weight_(jq) for interpEigenvec.
    // This avoids recomputing the product num_layers*num_streams times inside the hot loop.
    if (do_user_intensities) {
      for (int l = azimuth_mode; l < nstr_; ++l)
        for (int jq = 0; jq < nstr_; ++jq)
          interp_ylmc_cwt_(l, jq) = legendre_quad_(l, jq) * quad_weight_(jq);
    }

    // Compute Legendre polynomials at user angles for this azimuthal mode.
    // Mirrors C code line 743: c_legendre_poly(num_user_mu, azimuth_mode, num_streams, num_streams-1, mu_user, ylmu)
    // Must be done before interpEigenvec / interpSource which use legendre_user_.
    if (do_user_intensities) {
      for (int iu = 0; iu < numu_; ++iu) umu_vec_[iu] = config.mu_user[iu];
      legendrePolynomialsFlat(numu_, azimuth_mode, numu_ - 1, nstr_ - 1, umu_vec_.data(), legendre_user_.data());
    }

    // --- Eigenvalue Problem for Each Layer ---
    // Computes eigenvalues (kk) and eigenvectors (gc) that describe
    // the homogeneous solution (scattering without sources)
    for (int lc = 0; lc < nlyr_; ++lc) {
      solveEigen(lc, azimuth_mode);

      // Interpolate eigenvectors to user angles (only needs eigenvalues_/eigenvectors_/reduced_eigenvectors_)
      if (do_user_intensities) {
        interpEigenvec(lc, azimuth_mode);
      }
    }

    // --- Particular Solutions: Source Terms ---

    // Beam source (solar/stellar incident radiation)
    // Must run BEFORE interpSource since it fills beam_source_raw_[lc]
    if (config.bc.direct_beam_flux > 0.0) {
      if (config.flags.use_spherical_beam) {
        computeBeamSourceSpherical(azimuth_mode);
      } else {
        computeBeamSource(azimuth_mode);
      }
    }

    // Thermal source (Planck emission from internal heating)
    // Must run BEFORE interpSource since it fills thermal_particular_z0_/thermal_particular_z1_/planck_intercept_/planck_slope_ per layer
    // Only for azimuth mode 0 (thermal emission is isotropic)
    if (config.flags.use_thermal_emission && azimuth_mode == 0) {
      computeIsotropicSource(azimuth_mode);
    }

    // Interpolate sources to user angles (needs beam_source_raw_[lc] from computeBeamSource
    // and thermal_particular_z0_/thermal_particular_z1_/planck_intercept_/planck_slope_ from computeIsotropicSource)
    if (do_user_intensities) {
      for (int lc = 0; lc < nlyr_; ++lc) {
        interpSource(lc, azimuth_mode, kronecker_mode_0);
      }
    }

    // --- Surface Bidirectional Reflectivity ---
    // Must be computed BEFORE setMatrix and solve0 since they use surface_refl_quad_/surface_emis_quad_.
    // Also fills surface_refl_user_/surface_emis_user_ for user-angle intensity computation.
    // Mirrors C code order: c_surface_bidir() called before c_set_matrix().
    computeSurfaceBidir(kronecker_mode_0, azimuth_mode);

    // Store beam BRDF Fourier coefficient at user angles for BRDF intensity correction
    // (Fortran V3: accumulate RHOU(IU, 0, MAZIM) for later INTCOR_BEAM_REFLEC)
    if (!config.flags.use_lambertian_surface && do_user_intensities
        && config.bc.direct_beam_flux > 0.0) {
      for (int iu = 0; iu < numu_; ++iu) {
        brdf_fourier_user_beam_(iu, azimuth_mode) = surface_refl_user_(iu, 0);
      }
    }

    // --- Boundary Value Problem ---
    // Construct coefficient matrix encoding boundary conditions
    setMatrix(kronecker_mode_0, lyrcut_);

    // Solve for integration constants that satisfy all boundary conditions
    solve0(azimuth_mode, kronecker_mode_0);

    // --- Compute Fluxes and Intensities ---

    // Fluxes are only computed for azimuth mode 0 (azimuthally averaged)
    if (azimuth_mode == 0) {
      computeFluxes(result);
    }

    // Intensities at user angles
    if (do_user_intensities) {
      // Compute intensities and accumulate into result.
      // Returns max_azimuth_contrib = max relative azimuthal contribution (for azimuth_mode > 0).
      double max_azimuth_contrib = computeUserIntensities(azimuth_mode, kronecker_mode_0, result, planck_bottom_, planck_top_);

      // Azimuthal convergence check (mirrors C code lines 936-941).
      // Break early once two consecutive modes contribute less than accuracy_fourier_series
      // to the accumulated intensity field.
      if (azimuth_mode > 0 && config.accuracy_fourier_series > 0.0) {
        if (max_azimuth_contrib <= config.accuracy_fourier_series) {
          ++kconv;
        }
        if (kconv >= 2) {
          break;
        }
      }
    }
  }

  // ========================================================================
  // Post-Processing: Nakajima-Tanaka Intensity Correction
  // ========================================================================
  //
  // Apply TMS/IMS intensity correction when:
  //   - intensity_corr_buras flag is set
  //   - delta-M was applied
  //   - there is a beam source
  //   - scattering is present
  //   - user intensities were computed (not comp_only_fluxes)
  //
  // Mirrors C code lines 952-966 in cdisort.c.
  const bool corint = config.flags.intensity_corr_buras &&
            deltam_                            &&
            config.bc.direct_beam_flux > 0.0             &&
            scat_yes_                         &&
            !config.flags.comp_only_fluxes;
  if (corint) {
    if (config.flags.intensity_corr_nakajima || config.num_phase_func_angles <= 0) {
      // Nakajima-Tanaka algorithm (default, uses Legendre moments)
      applyIntensityCorrection(result);
    } else {
      // Buras-Emde algorithm (requires tabulated phase function in
      // config.mu_phase_function[] and config.phase_function[])
      applyNewIntensityCorrection(result);
    }
  }

  // ========================================================================
  // Post-Processing: Analytic BRDF Intensity Correction (Fortran V3)
  // ========================================================================
  //
  // Correct reflected beam intensity by comparing exact BRDF at user angles
  // with the Fourier-reconstructed approximation. In Fortran V3 this is
  // part of INTCOR_BEAM_REFLEC, gated on the same corint condition.
  if (config.flags.intensity_corr_buras && !deltam_plus_
      && !config.flags.use_lambertian_surface && !config.flags.comp_only_fluxes
      && config.flags.use_user_mu && config.bc.direct_beam_flux > 0.0) {
    applyBrdfIntensityCorrection(result);
  }

  return result;
}

// ============================================================================
// Memory Allocation
// ============================================================================

void DisortSolver::allocateWorkingArrays() 
{
  // Delta-M scaled arrays
  scaled_dtau_.resize(nlyr_ + 1);
  scaled_tau_cumulative_.resize(nlyr_ + 1);
  scaled_ssa_.resize(nlyr_ + 1);
  forward_scatter_frac_.resize(nlyr_ + 1);

  scaled_phase_coeffs_.resize(nstr_, nlyr_ + 1);

  // Beam attenuation
  chapman_.resize(nlyr_ + 1);
  chapman_tau_.resize(2 * nlyr_ + 2);
  beam_transmission_.resize(nlyr_ + 1);

  // User output levels
  scaled_tau_user_.resize(ntau_ + 1);
  layer_of_user_level_.resize(ntau_ + 1);

  // Quadrature angles
  quad_angle_.resize(nstr_);
  quad_weight_.resize(nstr_);

  // Legendre polynomials (RowMajor: ylm(l, angle_idx))
  legendre_quad_.resize(nstr_ + 1, nstr_);
  legendre_user_.resize(nstr_ + 1, numu_);
  legendre_beam_.resize(nstr_ + 1);

  // Eigenvalues and eigenvectors (contiguous per-layer storage)
  eigenvalues_.resize(nstr_, nlyr_ + 1);
  eigenvectors_.resize(nstr_, nstr_, nlyr_ + 1);

  // Coefficient matrices
  phase_matrix_.resize(nstr_, nstr_, nlyr_);
  alpha_minus_beta_.resize(nn_, nn_);
  alpha_plus_beta_.resize(nn_, nn_);
  eigen_product_.resize(nn_, nn_);
  reduced_eigenvectors_.resize(nstr_, nstr_);  // Full eigenvector matrix (num_streams x num_streams)
  reduced_eigenvalues_.resize(nn_);

  // Particular solution arrays (contiguous per-layer storage)
  beam_particular_.resize(nstr_, nlyr_ + 1);
  beam_particular_slope_.resize(nstr_, nlyr_ + 1);
  beam_particular_atten_.assign(nlyr_ + 1, 0.0);
  thermal_particular_z0_.resize(nstr_, nlyr_ + 1);
  thermal_particular_z1_.resize(nstr_, nlyr_ + 1);

  // Thermal expansion coefficients (per layer, only used when use_thermal_emission=true)
  planck_intercept_.assign(nlyr_, 0.0);
  planck_slope_.assign(nlyr_, 0.0);

  // Solution vector
  integration_constants_.resize(nstr_ * nlyr_);

  // Output working arrays
  intensity_quad_.resize(nstr_, ntau_);
  intensity_quad_.setZero();

  // Work arrays
  work_vec_.resize(nstr_);

  // BRDF arrays (always needed for setMatrix and solve0 boundary conditions)
  surface_refl_quad_.resize(nn_, nn_ + 1);
  surface_refl_quad_.setZero();
  surface_emis_quad_.resize(nn_);
  surface_emis_quad_.setZero();

  // BRDF azimuthal quadrature: NMUG=200 Gauss-Legendre points on [-1,1]
  // Mirrors C code: c_gaussian_quadrature(NMUG/2, gmu, gwt) then reflect
  gaussianQuadrature(NMUG / 2, gmu_brdf_, gwt_brdf_);
  // gaussianQuadrature fills 2*(NMUG/2) = NMUG elements: [0..NMUG/2-1] positive, [NMUG/2..NMUG-1] negative

  // Precomputed cosine series for BRDF azimuthal quadrature (Fortran V3 optimization).
  // cosmp_brdf_(m, k) = cos(m * PI * gmu_brdf_(k)) for m=0..nstr_-1, k=0..NMUG/2-1
  cosmp_brdf_.resize(nstr_, NMUG / 2);
  for (int k = 0; k < NMUG / 2; ++k) {
    cosmp_brdf_(0, k) = 1.0;
    for (int m = 1; m < nstr_; ++m) {
      cosmp_brdf_(m, k) = std::cos(static_cast<double>(m) * M_PI * gmu_brdf_(k));
    }
  }

  // User-angle intensity arrays (contiguous storage, allocated when !comp_only_fluxes && use_user_mu)
  if (!config_->flags.comp_only_fluxes && config_->flags.use_user_mu) {
    eigenvec_user_.resize(numu_, nstr_, nlyr_);
    beam_source_user_.resize(numu_, nlyr_);
    thermal_source_z0_user_.resize(numu_, nlyr_);
    thermal_source_z1_user_.resize(numu_, nlyr_);
    beam_source_raw_.resize(nstr_, nlyr_);
    intensity_azim_mode_.resize(numu_, ntau_);
    surface_refl_user_.resize(numu_, nn_ + 1);
    surface_emis_user_.resize(numu_);
    surface_refl_user_.setZero();
    surface_emis_user_.setZero();

    // BRDF intensity correction: store beam Fourier coefficients across azimuthal modes
    if (!config_->flags.use_lambertian_surface && config_->bc.direct_beam_flux > 0.0) {
      brdf_fourier_user_beam_.resize(numu_, nstr_);
      brdf_fourier_user_beam_.setZero();
    }
  }

  // Pre-allocate all scratch buffers to eliminate per-call heap allocation in hot path

  // solveEigen() (called num_layers*num_streams times per solve)
  cmu_vec_.resize(nstr_);
  eigenvalues_tmp_.resize(nn_);
  eigenvectors_tmp_.resize(nn_, nn_);
  sort_indices_tmp_.resize(nn_);
  similarity_d_.resize(nn_);
  similarity_d_inv_.resize(nn_);
  sym_A_s_.resize(nn_, nn_);
  sym_neg_B_s_.resize(nn_, nn_);
  v_tmp_.resize(nn_);

  // Main azimuthal loop + computeBeamSource() + computeBeamSourceSpherical() + solve0()
  umu_vec_.resize(numu_);
  umu0_vec_.resize(1);
  beam_A_.resize(nstr_, nstr_);
  beam_b_.resize(nstr_);
  beam_x_.resize(nstr_);
  isot_A_.resize(nstr_, nstr_);
  isot_b_.resize(nstr_);
  isot_z1_.resize(nstr_);
  isot_z0_.resize(nstr_);
  spher_A_.resize(nstr_, nstr_);
  spher_zj_.resize(nstr_);
  spher_wk_.resize(nstr_);
  solve0_b_.resize(nstr_ * nlyr_);   // conservative upper bound; ncut <= num_layers
  band_pivot_.resize(nstr_ * nlyr_);

  // solveEigen() phase_matrix_ DGEMM scratch (called num_layers*num_streams times per solve)
  cc_gl_ylmc_.resize(nstr_, nn_);  // Only need first nn_ cols as DGEMM left operand
  cc_top_half_.resize(nn_, nstr_);

  // interpEigenvec() + interpSource() (called num_layers*num_streams times per solve)
  interp_wk_l_.resize(nstr_);
  interp_psi0_.resize(nstr_);
  interp_W_.resize(nstr_ + 1, nstr_);
  interp_GU_tmp_.resize(numu_, nstr_);
  interp_ylmc_cwt_.resize(nstr_ + 1, nstr_);

  // computeUserIntensities() (called ~num_layers*num_user_tau*num_user_mu times per solve)
  ci_wk_neg_.resize(nn_);
  ci_wk_curr_neg_.resize(nn_);
  ci_wk_bnd_.resize(nn_);

  // applyIntensityCorrection() (called once per solve when corint=true)
  ic_phasa_.resize(nlyr_ + 1);
  ic_phasm_.resize(nlyr_ + 1);
  ic_phast_.resize(nlyr_ + 1);
  ic_ssalb_.resize(nlyr_ + 1);
  ic_oprim_.resize(nlyr_ + 1);
  ic_phirad_.resize(nphi_ + 1);
}


// ============================================================================
// Gaussian Quadrature
// ============================================================================

void DisortSolver::gaussianQuadrature(int n, Eigen::VectorXd& cmu, Eigen::VectorXd& cwt) 
{
  // Compute Gauss-Legendre quadrature points and weights on [0, 1]
  // Using iterative algorithm from Numerical Recipes

  const double eps = 3.0e-14;  // Convergence criterion
  const int max_iter = 10;

  cmu.resize(2 * n);
  cwt.resize(2 * n);

  // Compute for positive angles (upward)
  for (int i = 1; i <= n; ++i) {
    // Initial guess for ith root
    double z = std::cos(M_PI * (i - 0.25) / (n + 0.5));

    double z1, pp;
    int iter = 0;
    do {
      // Compute Legendre polynomial and derivative at z
      double p1 = 1.0;
      double p2 = 0.0;

      for (int j = 1; j <= n; ++j) {
        double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
      }

      // Derivative
      pp = n * (z * p1 - p2) / (z * z - 1.0);

      // Newton's method
      z1 = z;
      z = z1 - p1 / pp;

      ++iter;
      if (iter > max_iter) {
        throw std::runtime_error(
          "gaussianQuadrature: Failed to converge for root " + std::to_string(i));
      }
    } while (std::abs(z - z1) > eps);

    // Scale to [0, 1] and store (upward angles are positive)
    cmu(i - 1) = 0.5 * (1.0 + z);  // 0-based indexing

    // Weight
    cwt(i - 1) = 1.0 / ((1.0 - z * z) * pp * pp);
  }

  // Downward (negative) angles and weights
  // C++ convention: cmu[n+i] = -cmu[i] (direct pairing)
  // The self-consistent C++ stream ordering uses direct pairing throughout.
  for (int i = 0; i < n; ++i) {
    cmu(n + i) = -cmu(i);
    cwt(n + i) = cwt(i);
  }
}

// ============================================================================
// Setup and Delta-M Transformation
// ============================================================================

void DisortSolver::disortSet(DisortConfig& config, bool deltam) 
{
  const double abscut = 10.0;  // Absorption optical depth cutoff

  // Compute cumulative optical depths (tauc) from delta_tau.
  // Store as member tau_cumulative_ (needed later by intensity correction).
  // Using 0-based index: tau_cumulative_[0]=0, tau_cumulative_[lc] = sum(delta_tau[1..lc])
  tau_cumulative_.resize(nlyr_ + 1);
  tau_cumulative_[0] = 0.0;
  for (int lc = 0; lc < nlyr_; ++lc) {
    tau_cumulative_[lc + 1] = tau_cumulative_[lc] + config.delta_tau[lc];
  }
  const std::vector<double>& tauc = tau_cumulative_;  // local alias for readability

  // Set output levels at computational layer boundaries if not user-specified
  if (!config.flags.use_user_tau) {
    for (int lc = 0; lc < ntau_; ++lc) {
      config_->tau_user[lc] = tau_cumulative_[lc];
    }
  }

  // Apply delta-M scaling
  scaled_tau_cumulative_[0] = 0.0;
  double abstau = 0.0;
  ncut_ = nlyr_;

  for (int lc = 0; lc < nlyr_; ++lc) {
    // P_0 = 1 (normalization)
    const_cast<DisortConfig*>(config_)->phaseFunctionMoments(0, lc) = 1.0;

    // Track absorption optical depth for layer cutoff
    // ncut_ = number of layers included (1-based count = last included layer + 1).
    // C uses 1-based: ncut = lc (1..num_layers). C++ uses 0-based: ncut_ = lc + 1.
    if (abstau < abscut) {
      ncut_ = lc + 1;
    }
    abstau += (1.0 - config.single_scat_albedo[lc]) * config.delta_tau[lc];

    double f;  // Forward-scatter fraction

    if (!deltam && !deltam_plus_) {
      // No delta-M transformation
      scaled_ssa_[lc] = config.single_scat_albedo[lc];
      scaled_dtau_[lc] = config.delta_tau[lc];
      scaled_tau_cumulative_[lc + 1] = tauc[lc + 1];

      for (int k = 0; k < nstr_; ++k) {
        scaled_phase_coeffs_(k, lc) = (2.0 * k + 1.0) * scaled_ssa_[lc] * config.phaseFunctionMoments(k, lc);
      }
      f = 0.0;
    } else if (!deltam_plus_ ||
               config.phaseFunctionMoments(nstr_ - 1, lc) == config.phaseFunctionMoments(nstr_, lc)) {
      // Standard Delta-M transformation
      // Also used as per-layer fallback when consecutive moments M-1 and M are equal
      f = config.phaseFunctionMoments(nstr_, lc);
      scaled_ssa_[lc] = config.single_scat_albedo[lc] * (1.0 - f) / (1.0 - f * config.single_scat_albedo[lc]);
      scaled_dtau_[lc] = (1.0 - f * config.single_scat_albedo[lc]) * config.delta_tau[lc];
      scaled_tau_cumulative_[lc + 1] = scaled_tau_cumulative_[lc] + scaled_dtau_[lc];

      for (int k = 0; k < nstr_; ++k) {
        scaled_phase_coeffs_(k, lc) = (2.0 * k + 1.0) * scaled_ssa_[lc] *
               (config.phaseFunctionMoments(k, lc) - f) / (1.0 - f);
      }
    } else {
      // Delta-M+ transformation (Lin et al. 2018)
      double pm  = config.phaseFunctionMoments(nstr_, lc);
      double pm1 = config.phaseFunctionMoments(nstr_ + 1, lc);

      if (pm == pm1) {
        // Degenerate case: fall back to standard Delta-M for this layer
        f = pm;
        scaled_ssa_[lc] = config.single_scat_albedo[lc] * (1.0 - f) / (1.0 - f * config.single_scat_albedo[lc]);
        scaled_dtau_[lc] = (1.0 - f * config.single_scat_albedo[lc]) * config.delta_tau[lc];
        scaled_tau_cumulative_[lc + 1] = scaled_tau_cumulative_[lc] + scaled_dtau_[lc];
        for (int k = 0; k < nstr_; ++k) {
          scaled_phase_coeffs_(k, lc) = (2.0 * k + 1.0) * scaled_ssa_[lc] *
                 (config.phaseFunctionMoments(k, lc) - f) / (1.0 - f);
        }
      } else {
        double M = static_cast<double>(nstr_);
        double sigma_sq = ((M + 1.0) * (M + 1.0) - M * M) /
                          (std::log(pm * pm) - std::log(pm1 * pm1));
        double c = std::exp(M * M / (2.0 * sigma_sq));
        f = c * pm;

        scaled_ssa_[lc] = config.single_scat_albedo[lc] * (1.0 - f) / (1.0 - f * config.single_scat_albedo[lc]);
        scaled_dtau_[lc] = (1.0 - f * config.single_scat_albedo[lc]) * config.delta_tau[lc];
        scaled_tau_cumulative_[lc + 1] = scaled_tau_cumulative_[lc] + scaled_dtau_[lc];

        for (int k = 0; k < nstr_; ++k) {
          double kd = static_cast<double>(k);
          scaled_phase_coeffs_(k, lc) = (2.0 * k + 1.0) * scaled_ssa_[lc] *
                 (config.phaseFunctionMoments(k, lc) - f * std::exp(-kd * kd / (2.0 * sigma_sq))) / (1.0 - f);
        }
      }
    }

    forward_scatter_frac_[lc] = f;
  }

  // Calculate beam attenuation
  if (config.bc.direct_beam_flux > 0.0) {
    chapman_tau_[0] = 0.0;
    beam_transmission_[0] = 1.0;

    if (config.flags.use_spherical_beam) {
      // Pseudo-spherical geometry: use Chapman function for curved-atmosphere
      // beam path (port of c_disort_set() lines 3592-3617 in cdisort.c).

      // Special case: sun below horizon (direct_beam_mu < 0).
      // Adjust beam_transmission_[0] using unscaled optical depths (as in C reference).
      if (config.bc.direct_beam_mu < 0.0) {
        beam_transmission_[0] = std::exp(-chapmanImpl(1, 0.0,
                           config.level_altitudes, config.delta_tau,
                           nlyr_, config.bottom_radius,
                           config.bc.direct_beam_mu));
      }

      for (int lc = 0; lc < ncut_; ++lc) {
        // Optical depth at midpoint of layer lc (delta-M scaled)
        const double taup_mid = scaled_tau_cumulative_[lc] + scaled_dtau_[lc] / 2.0;
        // Chapman factor at midpoint → effective cosine CH(lc)
        const double chtau_mid = chapmanImpl(lc + 1, 0.5,
                           config.level_altitudes, scaled_dtau_,
                           nlyr_, config.bottom_radius,
                           config.bc.direct_beam_mu);
        chapman_[lc]     = taup_mid / chtau_mid;
        // Chapman optical depth at the bottom of 0-based layer lc.
        // Stored at level index lc+1 so that:
        //   chapman_tau_[0]    = 0 (TOA, set before loop)
        //   chapman_tau_[lc+1] = Chapman depth at bottom of layer lc
        // This matches the convention expected by computeBeamSource:
        //   q0a = exp(-chapman_tau_[lc])   = attenuation at TOP of layer lc
        //   q2a = exp(-chapman_tau_[lc+1]) = attenuation at BOTTOM of layer lc
        chapman_tau_[lc + 1]  = chapmanImpl(lc + 1, 0.0,
                        config.level_altitudes, scaled_dtau_,
                        nlyr_, config.bottom_radius,
                        config.bc.direct_beam_mu);
        beam_transmission_[lc + 1] = std::exp(-chapman_tau_[lc + 1]);
      }
    } else {
      // Plane-parallel geometry
      for (int lc = 0; lc < ncut_; ++lc) {
        chapman_[lc]     = config.bc.direct_beam_mu;
        beam_transmission_[lc + 1] = std::exp(-scaled_tau_cumulative_[lc + 1] / config.bc.direct_beam_mu);
      }
    }
  } else {
    for (int lc = 0; lc < ncut_; ++lc) {
      beam_transmission_[lc + 1] = 0.0;
    }
  }

  // Layer cutoff for absorption (if no thermal emission)
  lyrcut_ = false;
  if (abstau >= abscut && !config.flags.use_thermal_emission && nlyr_ > 1) {
    lyrcut_ = true;
  }
  if (!lyrcut_) {
    ncut_ = nlyr_;
  }

  // Map user output levels to computational mesh
  for (int lu = 0; lu < ntau_; ++lu) {
    int lc;
    for (lc = 0; lc < nlyr_ - 1; ++lc) {
      if (config.tau_user[lu] >= tauc[lc] &&
        config.tau_user[lu] <= tauc[lc + 1]) {
        break;
      }
    }

    scaled_tau_user_[lu] = config.tau_user[lu];
    if (deltam || deltam_plus_) {
      scaled_tau_user_[lu] = scaled_tau_cumulative_[lc] +
             (1.0 - config.single_scat_albedo[lc] * forward_scatter_frac_[lc]) *
             (config.tau_user[lu] - tauc[lc]);
    }
    layer_of_user_level_[lu] = lc;
  }

  // Compute Gaussian quadrature angles and weights
  gaussianQuadrature(nn_, quad_angle_, quad_weight_);

  // Populate cmu_vec_ (std::vector copy of quad_angle_) for legendrePolynomials().
  // Done once here so solveEigen() can reuse it without per-call allocation.
  for (int i = 0; i < nstr_; ++i) cmu_vec_[i] = quad_angle_(i);

  // Precompute similarity transform vectors for eigenvalue symmetrization:
  // D = diag(sqrt(cwt_i * mu_i)), D^{-1} = diag(1 / sqrt(cwt_i * mu_i))
  for (int iq = 0; iq < nn_; ++iq) {
    similarity_d_(iq) = std::sqrt(quad_weight_(iq) * quad_angle_(iq));
    similarity_d_inv_(iq) = 1.0 / similarity_d_(iq);
  }

  // Note: beam angle matching a quadrature angle is handled by per-layer
  // eigenvalue-based dithering in computeBeamSource() (Fortran DISORT V3 approach).

  // legendre_beam_ is sized here; values are computed per azimuthal mode in computeBeamSource()
  // (Associated Legendre recurrence requires sequential computation over azimuth_mode)

  // Set user output angles if not specified
  if (!config.flags.use_user_mu || config.flags.comp_only_fluxes) {
    for (int iu = 0; iu < nn_; ++iu) {
      const_cast<DisortConfig*>(config_)->mu_user[iu] = -quad_angle_(nn_ - 1 - iu);
    }
    for (int iu = nn_; iu < nstr_; ++iu) {
      const_cast<DisortConfig*>(config_)->mu_user[iu] = quad_angle_(iu - nn_);
    }
  }

  // SPECIAL_BC: rearrange UMU array to [-UMU(num_user_mu/2),...,-UMU(1), UMU(1),...,UMU(num_user_mu/2)]
  // (mirrors c_disort_set() lines 3691-3700)
  if (config.flags.use_user_mu && config.flags.ibcnd == BoundaryConditionType::Special) {
    const int numu_half = numu_ / 2;  // original user num_user_mu (numu_ was doubled in solve())
    auto& mu_user = const_cast<DisortConfig*>(config_)->mu_user;
    // Copy positive values to upper half
    for (int iu = 0; iu < numu_half; ++iu) {
      mu_user[iu + numu_half] = mu_user[iu];
    }
    // Put negatives (reversed) in lower half
    for (int iu = 0; iu < numu_half; ++iu) {
      mu_user[iu] = -mu_user[2 * numu_half - 1 - iu];  // -UMU(numu_half+1-iu)
    }
  }
}

// ============================================================================
// Eigenvalue Problem
// ============================================================================

void DisortSolver::solveEigen(int lc, int azimuth_mode) 
{
  // legendre_quad_ is pre-computed once per azimuth_mode in the main loop (solve()) — no recompute needed here.

  // Calculate coefficient matrix CC for this layer (eqs. SS(5-6), STWL(8b,15,23f))
  // Stored per layer (phase_matrix_[lc]) so each layer's matrix is available for beam/thermal sources.
  // Uses DGEMM:  cc_top(iq, jq) = 0.5 * cwt(jq) * sum_l { gl(l,lc) * ylmc(l,iq) * ylmc(l,jq) }
  const int nact = nstr_ - azimuth_mode;

  // Step 1: scale each row l of legendre_quad_[:, :nn_] block by scaled_phase_coeffs_(l+azimuth_mode, lc).
  // Only the first nn_ columns of legendre_quad_ are needed as DGEMM left operand.
  cc_gl_ylmc_.topRows(nact).noalias() =
    (legendre_quad_.block(azimuth_mode, 0, nact, nn_).array().colwise()
     * scaled_phase_coeffs_.col(lc).segment(azimuth_mode, nact).array()).matrix();

  // Step 2: DGEMM — cc_top_half_(iq, jq) = sum_l { cc_gl_ylmc_(l,iq) * legendre_quad_(l+azimuth_mode,jq) }
  cc_top_half_.noalias() = cc_gl_ylmc_.topRows(nact).transpose()
                * legendre_quad_.block(azimuth_mode, 0, nact, nstr_);
  cc_top_half_ *= 0.5;
  cc_top_half_.array().rowwise() *= quad_weight_.transpose().array();  // column-scale by quad_weight_

  // Step 3: fill phase_matrix_[lc] — top half explicit, bottom half by symmetry.
  phase_matrix_[lc].topRows(nn_)              = cc_top_half_;
  phase_matrix_[lc].bottomLeftCorner(nn_, nn_) = cc_top_half_.rightCols(nn_);
  phase_matrix_[lc].bottomRightCorner(nn_, nn_) = cc_top_half_.leftCols(nn_);

  // Steps 4–7: eigenvalue problem — dispatch to fixed-size or dynamic implementation
  switch (nn_) {
    case 2:  solveEigenCore<2>(lc);  return;
    case 4:  solveEigenCore<4>(lc);  return;
    case 6:  solveEigenCore<6>(lc);  return;
    case 8:  solveEigenCore<8>(lc);  return;
    case 10:  solveEigenCore<10>(lc);  return;
    case 12:  solveEigenCore<12>(lc);  return;
    case 14:  solveEigenCore<14>(lc);  return;
    case 16: solveEigenCore<16>(lc); return;
    default: solveEigenCoreDynamic(lc); return;
  }
}


// ============================================================================
// solveEigenCoreDynamic — dynamic-size fallback for unusual NN values
// ============================================================================

void DisortSolver::solveEigenCoreDynamic(int lc)
{
  // Step 4: build alpha_minus_beta_ and alpha_plus_beta_ for the reduced eigenvalue problem.
  for (int iq = 0; iq < nn_; ++iq) {
    const double inv_cmu = 1.0 / quad_angle_(iq);
    alpha_minus_beta_.row(iq).noalias() = inv_cmu * (cc_top_half_.row(iq).head(nn_)
                       - cc_top_half_.row(iq).tail(nn_));
    alpha_plus_beta_.row(iq).noalias() = inv_cmu * (cc_top_half_.row(iq).head(nn_)
                       + cc_top_half_.row(iq).tail(nn_));
    alpha_minus_beta_(iq, iq) -= inv_cmu;
    alpha_plus_beta_(iq, iq) -= inv_cmu;
  }

  // Symmetrize via similarity transform D = diag(sqrt(cwt*mu))
  for (int i = 0; i < nn_; ++i) {
    for (int j = 0; j < nn_; ++j) {
      sym_A_s_(i, j) = similarity_d_(i) * alpha_plus_beta_(i, j) * similarity_d_inv_(j);
      sym_neg_B_s_(i, j) = -similarity_d_(i) * alpha_minus_beta_(i, j) * similarity_d_inv_(j);
    }
  }

  bool use_spd_path = (scaled_ssa_[lc] < 1.0 - 1e-12);
  if (use_spd_path) {
    cholesky_.compute(sym_neg_B_s_);
    use_spd_path = (cholesky_.info() == Eigen::Success);
  }
  if (use_spd_path) {
    const auto& L = cholesky_.matrixL();
    eigen_product_.noalias() = -sym_A_s_ * L;
    sym_neg_B_s_.noalias() = L.transpose() * eigen_product_;

    sym_eig_solver_.compute(sym_neg_B_s_);
    eigenvalues_tmp_ = sym_eig_solver_.eigenvalues();

    const auto& U = cholesky_.matrixU();
    for (int i = 0; i < nn_; ++i) {
      v_tmp_.noalias() = U.solve(sym_eig_solver_.eigenvectors().col(i));
      for (int j = 0; j < nn_; ++j) {
        eigenvectors_tmp_(j, i) = similarity_d_inv_(j) * v_tmp_(j);
      }
    }
  } else {
    eigen_product_.noalias() = alpha_plus_beta_ * alpha_minus_beta_;
    eig_solver_.compute(eigen_product_, true);

    const auto& cevals = eig_solver_.eigenvalues();
    const auto& cevecs = eig_solver_.eigenvectors();

    for (int i = 0; i < nn_; ++i) sort_indices_tmp_[i] = i;
    std::sort(sort_indices_tmp_.begin(), sort_indices_tmp_.end(),
          [&cevals](int a, int b) {
            return std::abs(cevals(a)) < std::abs(cevals(b));
          });

    for (int i = 0; i < nn_; ++i) {
      eigenvalues_tmp_(i) = cevals(sort_indices_tmp_[i]).real();
      for (int j = 0; j < nn_; ++j) {
        eigenvectors_tmp_(j, i) = cevecs(j, sort_indices_tmp_[i]).real();
      }
    }
  }

  // Extract eigenvalues: k = sqrt(|λ|)
  for (int iq = 0; iq < nn_; ++iq) {
    reduced_eigenvalues_(iq) = std::sqrt(std::abs(eigenvalues_tmp_(iq)));
    eigenvalues_[lc](iq + nn_) = reduced_eigenvalues_(iq);
    eigenvalues_[lc](nn_ - 1 - iq) = -reduced_eigenvalues_(iq);
  }

  // Compute eigenvectors (G+) + (G-) via alpha_minus_beta * eigenvectors_tmp / k
  for (int jq = 0; jq < nn_; ++jq) {
    for (int iq = 0; iq < nn_; ++iq) {
      double sum = 0.0;
      for (int kq = 0; kq < nn_; ++kq) {
        sum += alpha_minus_beta_(iq, kq) * eigenvectors_tmp_(kq, jq);
      }
      alpha_plus_beta_(iq, jq) = sum / reduced_eigenvalues_(jq);
    }
  }

  // Recover G+, G- and store in reduced_eigenvectors_ and eigenvectors_[lc]
  for (int jq = 0; jq < nn_; ++jq) {
    for (int iq = 0; iq < nn_; ++iq) {
      double gpplgm = alpha_plus_beta_(iq, jq);
      double gpmigm = eigenvectors_tmp_(iq, jq);

      reduced_eigenvectors_(iq, jq) = 0.5 * (gpplgm + gpmigm);
      reduced_eigenvectors_(iq + nn_, jq) = 0.5 * (gpplgm - gpmigm);

      gpplgm *= -1.0;
      reduced_eigenvectors_(iq, jq + nn_) = 0.5 * (gpplgm + gpmigm);
      reduced_eigenvectors_(iq + nn_, jq + nn_) = 0.5 * (gpplgm - gpmigm);

      eigenvectors_[lc](nn_ + iq, nn_ + jq) = reduced_eigenvectors_(iq, jq);
      eigenvectors_[lc](nn_ - iq - 1, nn_ + jq) = reduced_eigenvectors_(iq + nn_, jq);
      eigenvectors_[lc](nn_ + iq, nn_ - jq - 1) = reduced_eigenvectors_(iq, jq + nn_);
      eigenvectors_[lc](nn_ - iq - 1, nn_ - jq - 1) = reduced_eigenvectors_(iq + nn_, jq + nn_);
    }
  }
}

// ============================================================================
// solveEigenCore<NN> — fixed-size eigenvalue computation for common NN values
// ============================================================================

template<int NN>
void DisortSolver::solveEigenCore(int lc)
{
  using HalfMat = Eigen::Matrix<double, NN, NN>;
  using HalfVec = Eigen::Matrix<double, NN, 1>;

  // Step 4: build alpha_minus_beta and alpha_plus_beta from cc_top_half_
  HalfMat amb, apb;
  for (int iq = 0; iq < NN; ++iq) {
    const double inv_cmu = 1.0 / quad_angle_(iq);
    for (int jq = 0; jq < NN; ++jq) {
      amb(iq, jq) = inv_cmu * (cc_top_half_(iq, jq) - cc_top_half_(iq, jq + NN));
      apb(iq, jq) = inv_cmu * (cc_top_half_(iq, jq) + cc_top_half_(iq, jq + NN));
    }
    amb(iq, iq) -= inv_cmu;
    apb(iq, iq) -= inv_cmu;
  }

  // Symmetrize via similarity transform D = diag(sqrt(cwt*mu))
  HalfMat A_s, neg_B_s;
  for (int i = 0; i < NN; ++i) {
    for (int j = 0; j < NN; ++j) {
      A_s(i, j) = similarity_d_(i) * apb(i, j) * similarity_d_inv_(j);
      neg_B_s(i, j) = -similarity_d_(i) * amb(i, j) * similarity_d_inv_(j);
    }
  }

  // Step 5: Eigenvalue solve
  HalfVec evals;
  HalfMat evecs;

  bool use_spd_path = (scaled_ssa_[lc] < 1.0 - 1e-12);
  if (use_spd_path) {
    Eigen::LLT<HalfMat> chol(neg_B_s);
    use_spd_path = (chol.info() == Eigen::Success);
    if (use_spd_path) {
      const auto& L = chol.matrixL();
      HalfMat product;
      product.noalias() = -A_s * L;
      HalfMat S;
      S.noalias() = L.transpose() * product;

      Eigen::SelfAdjointEigenSolver<HalfMat> eig(S);
      evals = eig.eigenvalues();

      // Recover eigenvectors: u = D^{-1} * L^{-T} * w
      const auto& U = chol.matrixU();
      for (int i = 0; i < NN; ++i) {
        HalfVec v = U.solve(eig.eigenvectors().col(i));
        for (int j = 0; j < NN; ++j) {
          evecs(j, i) = similarity_d_inv_(j) * v(j);
        }
      }
    }
  }
  if (!use_spd_path) {
    // Fallback for conservative scattering (omega=1) or degenerate cases
    HalfMat product;
    product.noalias() = apb * amb;
    Eigen::EigenSolver<HalfMat> eig(product, true);

    const auto& cevals = eig.eigenvalues();
    const auto& cevecs = eig.eigenvectors();

    std::array<int, NN> idx;
    for (int i = 0; i < NN; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
          [&cevals](int a, int b) {
            return std::abs(cevals(a)) < std::abs(cevals(b));
          });

    for (int i = 0; i < NN; ++i) {
      evals(i) = cevals(idx[i]).real();
      for (int j = 0; j < NN; ++j) {
        evecs(j, i) = cevecs(j, idx[i]).real();
      }
    }
  }

  // Step 6: Extract eigenvalues: k = sqrt(|λ|)
  HalfVec reduced_evals;
  for (int iq = 0; iq < NN; ++iq) {
    reduced_evals(iq) = std::sqrt(std::abs(evals(iq)));
    eigenvalues_[lc](iq + NN) = reduced_evals(iq);
    eigenvalues_[lc](NN - 1 - iq) = -reduced_evals(iq);
  }

  // Step 7: Compute eigenvectors (G+) + (G-) via amb * evecs / k
  HalfMat gpplgm;
  for (int jq = 0; jq < NN; ++jq) {
    for (int iq = 0; iq < NN; ++iq) {
      double sum = 0.0;
      for (int kq = 0; kq < NN; ++kq) {
        sum += amb(iq, kq) * evecs(kq, jq);
      }
      gpplgm(iq, jq) = sum / reduced_evals(jq);
    }
  }

  // Recover G+, G- and store in reduced_eigenvectors_ and eigenvectors_[lc]
  for (int jq = 0; jq < NN; ++jq) {
    for (int iq = 0; iq < NN; ++iq) {
      double gp = gpplgm(iq, jq);
      double gm = evecs(iq, jq);

      reduced_eigenvectors_(iq, jq) = 0.5 * (gp + gm);
      reduced_eigenvectors_(iq + NN, jq) = 0.5 * (gp - gm);

      gp = -gp;
      reduced_eigenvectors_(iq, jq + NN) = 0.5 * (gp + gm);
      reduced_eigenvectors_(iq + NN, jq + NN) = 0.5 * (gp - gm);

      eigenvectors_[lc](NN + iq, NN + jq) = reduced_eigenvectors_(iq, jq);
      eigenvectors_[lc](NN - iq - 1, NN + jq) = reduced_eigenvectors_(iq + NN, jq);
      eigenvectors_[lc](NN + iq, NN - jq - 1) = reduced_eigenvectors_(iq, jq + NN);
      eigenvectors_[lc](NN - iq - 1, NN - jq - 1) = reduced_eigenvectors_(iq + NN, jq + NN);
    }
  }
}

// Explicit instantiations for common NN values
template void DisortSolver::solveEigenCore<2>(int lc);
template void DisortSolver::solveEigenCore<4>(int lc);
template void DisortSolver::solveEigenCore<6>(int lc);
template void DisortSolver::solveEigenCore<8>(int lc);
template void DisortSolver::solveEigenCore<10>(int lc);
template void DisortSolver::solveEigenCore<12>(int lc);
template void DisortSolver::solveEigenCore<14>(int lc);
template void DisortSolver::solveEigenCore<16>(int lc);


void DisortSolver::setMatrix(double kronecker_mode_0, bool lyrcut)
{
  /*
   * Build the banded coefficient matrix for boundary value problem.
   * Based on c_set_matrix() from cdisort.c:3776-3912
   *
   * Matrix structure: (9*nn-2) × (num_streams*num_layers) band matrix
   * - Encodes continuity at layer interfaces (STWJ eq. 17, 20)
   * - Top boundary condition: no downward diffuse radiation (STWJ 20a)
   * - Bottom boundary condition: surface reflection (STWJ 20c)
   *
   * Uses LINPACK banded storage format with scaling transformation (STWJ 22)
   */

  const int nn = nn_;
  const int mi9m2 = 9 * nn - 2;      // Band matrix row storage
  const int nnlyri = nstr_ * nlyr_;  // Total unknowns
  const int ncd = 3 * nn - 1;        // Number of diagonals above/below main
  const int lda = 3 * ncd + 1;       // Leading dimension for band storage
  const int nshift = lda - 2 * nstr_ + 1;  // Index shift for banded storage

  // Allocate and zero out the banded matrix
  band_matrix_.resize(mi9m2, nnlyri);
  band_matrix_.setZero();

  int ncol = 0;  // Column counter (0-based in C++)

  // ========================================================================
  // Layer Continuity: Use STWJ(17) to form coefficient matrix in STWJ(20)
  // Employ scaling transformation STWJ(22)
  // ========================================================================
  for (int lc = 0; lc < ncut_; ++lc) {
    // Compute exponential scaling factors for numerical stability
    for (int iq = 0; iq < nn; ++iq) {
      work_vec_(iq) = exp(eigenvalues_[lc](iq) * scaled_dtau_[lc]);  // scaled_dtau_ is 1-based
    }

    int jcol = 0;

    // Downward streams (iq = 0 to nn-1, corresponding to C's iq = 1 to nn)
    for (int iq = 0; iq < nn; ++iq) {
      int irow = nshift - jcol - 1;  // Convert to 0-based
      for (int jq = 0; jq < nstr_; ++jq) {
        // Check bounds before access
        if (irow + nstr_ >= 0 && irow + nstr_ < band_matrix_.rows() && ncol < band_matrix_.cols()) {
          // Continuity at layer top
          band_matrix_(irow + nstr_, ncol) = eigenvectors_[lc](jq, iq);
        }
        if (irow >= 0 && irow < band_matrix_.rows() && ncol < band_matrix_.cols()) {
          // Continuity at layer bottom (scaled)
          band_matrix_(irow, ncol) = -eigenvectors_[lc](jq, iq) * work_vec_(iq);
        }
        irow++;
      }
      ncol++;
      jcol++;
    }

    // Upward streams (iq = nn to num_streams-1, corresponding to C's iq = nn+1 to num_streams)
    for (int iq = nn; iq < nstr_; ++iq) {
      int irow = nshift - jcol - 1;  // Convert to 0-based
      for (int jq = 0; jq < nstr_; ++jq) {
        // Continuity at layer top (scaled with reversed index)
        band_matrix_(irow + nstr_, ncol) = eigenvectors_[lc](jq, iq) * work_vec_(nstr_ - iq - 1);
        // Continuity at layer bottom
        band_matrix_(irow, ncol) = -eigenvectors_[lc](jq, iq);
        irow++;
      }
      ncol++;
      jcol++;
    }
  }

  // ========================================================================
  // Top Boundary Condition: STWJ(20a) - No downward diffuse radiation at top
  // ========================================================================
  int jcol = 0;

  // Downward streams at top
  for (int iq = 0; iq < nn; ++iq) {
    double expa = exp(eigenvalues_[0](iq) * scaled_tau_cumulative_[1]);  // scaled_tau_cumulative_[0]=0, scaled_tau_cumulative_[1]=bottom of layer 1
    int irow = nshift - jcol + nn - 1;  // Convert to 0-based
    for (int jq = nn - 1; jq >= 0; --jq) {  // Reverse order
      band_matrix_(irow, jcol) = eigenvectors_[0](jq, iq) * expa;
      irow++;
    }
    jcol++;
  }

  // Upward streams at top
  for (int iq = nn; iq < nstr_; ++iq) {
    int irow = nshift - jcol + nn - 1;  // Convert to 0-based
    for (int jq = nn - 1; jq >= 0; --jq) {  // Reverse order
      band_matrix_(irow, jcol) = eigenvectors_[0](jq, iq);
      irow++;
    }
    jcol++;
  }

  // ========================================================================
  // Bottom Boundary Condition: STWJ(20c) - Surface reflection
  // ========================================================================
  // Note: In C (1-based), nncol = *ncol - num_streams, then ++nncol accesses (*ncol - num_streams + 1) to *ncol
  // In C++ (0-based), we need nncol to access (ncol - num_streams) to (ncol - 1)
  // So we start at (ncol - num_streams - 1), then ++nncol gives us the correct range
  int nncol = ncol - nstr_ - 1;  // Start column for bottom BC (adjusted for 0-based)
  jcol = 0;

  const int ncut = ncut_;  // Last computational layer (0-based)

  // Downward streams at bottom
  for (int iq = 0; iq < nn; ++iq) {
    nncol++;
    int irow = nshift - jcol + nstr_ - 1;  // Convert to 0-based
    // BDR reflection coupling: BDR(jq-nn, k) depends on the output stream jq.
    // For Lambertian: surface_refl_quad_(jq-nn_, k+1)=surface_albedo (constant), sum could be precomputed.
    // For BRDF: must compute per-jq. We always compute per-jq for generality.
    // C code: STWJ(20c), lines 3900-3904.
    for (int jq = nn; jq < nstr_; ++jq) {
      if (lyrcut || (config_->flags.use_lambertian_surface && kronecker_mode_0 == 0.0)) {
        // Lambertian surface + azimuth_mode>0, or truncated layer: no coupling
        band_matrix_(irow, nncol) = eigenvectors_[ncut - 1](jq, iq);
      } else {
        // Surface reflection coupling: C code BDR(jq-nn, k) → surface_refl_quad_(jq-nn_, k+1)
        double refl_sum = 0.0;
        for (int k = 0; k < nn_; ++k) {
          refl_sum += quad_weight_(k) * quad_angle_(k) * surface_refl_quad_(jq - nn_, k + 1)
                * eigenvectors_[ncut - 1](nn_ - 1 - k, iq);
        }
        band_matrix_(irow, nncol) = eigenvectors_[ncut - 1](jq, iq) - (1.0 + kronecker_mode_0) * refl_sum;
      }
      irow++;
    }
    jcol++;
  }

  // Upward streams at bottom
  for (int iq = nn; iq < nstr_; ++iq) {
    nncol++;
    int irow = nshift - jcol + nstr_ - 1;  // Convert to 0-based
    double expa = work_vec_(nstr_ - iq - 1);  // Use stored exponential from continuity loop
    for (int jq = nn; jq < nstr_; ++jq) {
      if (lyrcut || (config_->flags.use_lambertian_surface && kronecker_mode_0 == 0.0)) {
        // Lambertian + azimuth_mode>0, or truncated layer
        band_matrix_(irow, nncol) = eigenvectors_[ncut - 1](jq, iq) * expa;
      } else {
        // Surface reflection coupling using surface_refl_quad_
        double refl_sum = 0.0;
        for (int k = 0; k < nn_; ++k) {
          refl_sum += quad_weight_(k) * quad_angle_(k) * surface_refl_quad_(jq - nn_, k + 1)
                * eigenvectors_[ncut - 1](nn_ - 1 - k, iq);
        }
        band_matrix_(irow, nncol) = (eigenvectors_[ncut - 1](jq, iq) - (1.0 + kronecker_mode_0) * refl_sum) * expa;
      }
      irow++;
    }
    jcol++;
  }

  // Store the number of columns used (for solve0)
  // Note: In C++ we use 0-based, but the algorithm still needs to know the count
  // This will be used when we implement solve0
}

void DisortSolver::computeBeamSource(int azimuth_mode) 
{
  /*
   * Compute beam source particular solution for all layers.
   * Based on c_upbeam() from cdisort.c:5187-5241
   *
   * Solves: [I + μ/μ₀·δᵢⱼ - C] Z = RHS
   * where RHS involves phase function expansion
   *
   * This is called once per azimuthal mode for all layers.
   */

  const double kronecker_mode_0 = (azimuth_mode == 0) ? 1.0 : 0.0;
  const double direct_beam_mu = config_->bc.direct_beam_mu;
  const double direct_beam_flux = config_->bc.direct_beam_flux;

  // Compute associated Legendre polynomials of beam angle for this azimuthal mode.
  // Must be called in sequential order (azimuth_mode = 0, 1, 2, ...) because the m > 0
  // recurrence uses legendre_beam_[m-1] set by the previous azimuth_mode call.
  // CRITICAL: Use -direct_beam_mu, not +direct_beam_mu! The beam travels DOWNWARD, so its direction
  // cosine is -direct_beam_mu (negative). C code uses: angcos = -ds->bc.direct_beam_mu.
  // For Rayleigh (even-k only): P_k(-x)=P_k(x), so sign doesn't matter.
  // For HG/Haze (all k): P_k(-x)=(-1)^k*P_k(x), sign matters for odd k.
  // Mirrors C: c_legendre_poly(1, azimuth_mode, num_streams, num_streams-1, &angcos, ylm0) where angcos=-direct_beam_mu
  umu0_vec_[0] = -direct_beam_mu;
  legendrePolynomialsFlat(1, azimuth_mode, 0, nstr_ - 1, umu0_vec_.data(), legendre_beam_.data());

  // Dither threshold for beam angle singularity detection (Fortran DISORT V3 approach).
  // When |1 - μ₀² × κᵢ²| < threshold for any eigenvalue κᵢ, dither μ₀ by 0.1%.
  constexpr double dither_threshold = 1000.0 * std::numeric_limits<double>::epsilon();

  // Loop over all computational layers
  for (int lc = 0; lc < nlyr_; ++lc) {
    // Check if beam angle creates near-singularity with any eigenvalue (V3 fix)
    double beam_mu = direct_beam_mu;
    const double mu0sq = direct_beam_mu * direct_beam_mu;
    for (int iq = 0; iq < nn_; ++iq) {
      double kk = eigenvalues_[lc](iq + nn_);  // positive eigenvalue
      if (std::abs(1.0 - mu0sq * kk * kk) < dither_threshold) {
        beam_mu = 0.999 * direct_beam_mu;
        break;
      }
    }

    // Build coefficient matrix: A = I + diag(μ/μ₀) - C  (vectorized Eigen ops)
    beam_A_.noalias() = -phase_matrix_[lc];
    beam_A_.diagonal().array() += 1.0 + quad_angle_.array() / beam_mu;

    // Build RHS: (2-kronecker_mode_0) * direct_beam_flux * sum_k[GL(k,lc)*YLMC(k,iq)*YLM0(k)] / (4π)
    for (int iq = 0; iq < nstr_; ++iq) {
      double sum = 0.0;
      for (int k = azimuth_mode; k < nstr_; ++k) {
        sum += scaled_phase_coeffs_(k, lc) * legendre_quad_(k, iq) * legendre_beam_(k);
      }
      beam_b_(iq) = (2.0 - kronecker_mode_0) * direct_beam_flux * sum / (4.0 * M_PI);
    }

    // Solve: reuse member LU to avoid per-layer heap allocation (warm path: ~96×/solve)
    beam_lu_.compute(beam_A_);
    beam_x_.noalias() = beam_lu_.solve(beam_b_);

    // Save raw solution (ZJ in C code) for use by interpSource
    // This must be done BEFORE the reordering below
    if (!beam_source_raw_.empty()) {
      beam_source_raw_[lc] = beam_x_;
    }

    // Store solution in beam_particular_ with angular reordering for symmetry
    // C code: ZZ(nn+iq, lc) = ZJ(iq) for iq=1..nn (downward)
    //         ZZ(nn-iq+1, lc) = ZJ(iq+nn) for iq=1..nn (upward)
    // C++: Convert to 0-based indexing
    for (int iq = 0; iq < nn_; ++iq) {
      beam_particular_[lc](nn_ + iq)     = beam_x_(iq);         // Downward streams
      beam_particular_[lc](nn_ - iq - 1) = beam_x_(iq + nn_);   // Upward streams (reversed)
    }
  }
}

void DisortSolver::computeBeamSourceSpherical(int azimuth_mode) 
{
  /*
   * Compute pseudo-spherical beam source particular solution.
   * Port of c_set_coefficients_beam_source() + c_upbeam_pseudo_spherical()
   * from cdisort.c:5026-5167 and 5265-5400.
   *
   * For each layer lc, computes ZBEAM0(iq,lc) and ZBEAM1(iq,lc) such that:
   *   beam_source(iq, tau) = exp(-ZBEAMA(lc) * tau) * (ZBEAM0(iq,lc) + ZBEAM1(iq,lc)*tau)
   *
   * Stored in beam_particular_[lc-1] (=ZBEAM0), beam_particular_slope_[lc-1] (=ZBEAM1), beam_particular_atten_[lc] (=XBA).
   * (beam_particular_ indices are 0-based; lc here is 1-based layer counter)
   */

  const double kronecker_mode_0  = (azimuth_mode == 0) ? 1.0 : 0.0;
  const double direct_beam_flux  = config_->bc.direct_beam_flux;
  const double big    = std::sqrt(std::numeric_limits<double>::max()) / 1.0e10;

  // Compute Legendre polynomials at beam angle (same as plane-parallel)
  umu0_vec_[0] = -config_->bc.direct_beam_mu;
  legendrePolynomialsFlat(1, azimuth_mode, 0, nstr_ - 1, umu0_vec_.data(), legendre_beam_.data());

  Eigen::MatrixXd& A  = spher_A_;
  Eigen::VectorXd& zj = spher_zj_;

  for (int lc = 0; lc < nlyr_; ++lc) {
    // ========================================================
    // c_set_coefficients_beam_source() equivalent
    // ========================================================

    // ZJ(iq) = (2-kronecker_mode_0) * direct_beam_flux * sum(GL*YLMC*YLM0) / (4*pi)
    for (int iq = 0; iq < nstr_; ++iq) {
      double sum = 0.0;
      for (int k = azimuth_mode; k < nstr_; ++k) {
        sum += scaled_phase_coeffs_(k, lc) * legendre_quad_(k, iq) * legendre_beam_(k);
      }
      zj(iq) = (2.0 - kronecker_mode_0) * direct_beam_flux * sum / (4.0 * M_PI);
    }

    // Beam transmission at top and bottom of layer lc (0-based)
    // chapman_tau_[lc]   = Chapman optical depth at bottom of layer (lc-1 in 1-based)
    // chapman_tau_[lc+1] = Chapman optical depth at bottom of layer lc
    // Wait: disortSet fills chapman_tau_[lc] = chapmanImpl(lc, 0.0, ...) for lc=1..ncut
    // So for 0-based lc: chapman_tau_[lc] = layer lc+1 bottom in 1-based = bottom of layer lc (0-based)
    // And chapman_tau_[0] = 0 (TOA)
    const double q0a = std::exp(-chapman_tau_[lc]);       // exp(-CHTAU(lc))   = top of lc (0-based)
    const double q2a = std::exp(-chapman_tau_[lc + 1]);   // exp(-CHTAU(lc+1)) = bottom of lc (0-based)

    // XBA(lc) = 1/CH(lc), with dithering if necessary
    // chapman_[lc] in OneBased corresponds to 1-based CH(lc+1) which is our 0-based layer lc
    double xba = 1.0 / chapman_[lc];
    if (std::abs(xba) > big && scaled_tau_cumulative_[lc + 1] > 1.0) xba = 0.0;
    if (std::abs(xba * scaled_tau_cumulative_[lc + 1]) > std::log(big)) xba = 0.0;
    // Dither if xba is close to a quadrature angle cosine
    if (std::abs(xba) > 1.0e-5) {
      for (int iq = 0; iq < nn_; ++iq) {
        if (std::abs((std::abs(xba) - 1.0 / quad_angle_(iq)) / xba) < 0.05) {
          xba *= 1.001;
          break;
        }
      }
    }
    beam_particular_atten_[lc] = xba;

    const double deltat = scaled_tau_cumulative_[lc + 1] - scaled_tau_cumulative_[lc];

    // XB0(iq,lc) and XB1(iq,lc): coefficients of linear-in-tau beam source expansion
    // XB1(iq) = (q2*exp(xba*taucpr[lc+1]) - q0*exp(xba*taucpr[lc])) / deltat
    // XB0(iq) = q0*exp(xba*taucpr[lc]) - XB1*taucpr[lc]
    Eigen::VectorXd xb0(nstr_), xb1(nstr_);
    for (int iq = 0; iq < nstr_; ++iq) {
      const double q0 = q0a * zj(iq);
      const double q2 = q2a * zj(iq);
      xb1(iq) = (q2 * std::exp(xba * scaled_tau_cumulative_[lc + 1])
           - q0 * std::exp(xba * scaled_tau_cumulative_[lc])) / deltat;
      xb0(iq) = q0 * std::exp(xba * scaled_tau_cumulative_[lc]) - xb1(iq) * scaled_tau_cumulative_[lc];
    }

    // ========================================================
    // c_upbeam_pseudo_spherical() equivalent
    // ========================================================

    // Build system [I + xba*mu*I - C] ZBS1 = XB1
    for (int iq = 0; iq < nstr_; ++iq) {
      for (int jq = 0; jq < nstr_; ++jq) {
        A(iq, jq) = -phase_matrix_[lc](iq, jq);
      }
      A(iq, iq) += 1.0 + xba * quad_angle_(iq);
    }
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);

    // Dither xba if matrix is near-singular
    const double rcond_approx = 1.0 / A.norm();  // rough estimate
    if (rcond_approx < 1.0e-4) {
      if (xba == 0.0) xba = 5.0e-9;
      xba *= 1.00000005;
      beam_particular_atten_[lc] = xba;
      for (int iq = 0; iq < nstr_; ++iq) {
        for (int jq = 0; jq < nstr_; ++jq) {
          A(iq, jq) = -phase_matrix_[lc](iq, jq);
        }
        A(iq, iq) += 1.0 + xba * quad_angle_(iq);
      }
      lu = Eigen::PartialPivLU<Eigen::MatrixXd>(A);
    }

    // Solve [I + xba*mu*I - C] ZBS1 = XB1 → ZBS1
    Eigen::VectorXd zbs1 = lu.solve(xb1);

    // ZBS0(iq) = XB0(iq) + CMU(iq)*ZBS1(iq)
    Eigen::VectorXd zbs0(nstr_);
    for (int iq = 0; iq < nstr_; ++iq) {
      zbs0(iq) = xb0(iq) + quad_angle_(iq) * zbs1(iq);
    }

    // Solve [I + xba*mu*I - C] ZBS0 = ZBS0_rhs → ZBS0
    zbs0 = lu.solve(zbs0);

    // Store with reordering (same as c_upbeam reordering):
    // ZBEAM0(iq+nn, lc) = ZBS0(iq)     for iq=1..nn  (downward streams)
    // ZBEAM0(nn-iq+1,lc) = ZBS0(iq+nn) for iq=1..nn  (upward streams reversed)
    // C++ 0-based: nn_+iq-1 → nn_+(iq-1), (nn_-1-(iq-1)) → (nn_-iq)
    for (int iq = 0; iq < nn_; ++iq) {
      beam_particular_[lc](nn_ + iq)    = zbs0(iq);        // ZBEAM0 downward
      beam_particular_[lc](nn_ - 1 - iq) = zbs0(iq + nn_); // ZBEAM0 upward (reversed)
      beam_particular_slope_[lc](nn_ + iq)   = zbs1(iq);        // ZBEAM1 downward
      beam_particular_slope_[lc](nn_ - 1 - iq) = zbs1(iq + nn_); // ZBEAM1 upward (reversed)
    }

    // Also save raw ZJ for interpSource (used in user intensity computation)
    if (!beam_source_raw_.empty()) {
      beam_source_raw_[lc] = zj;
    }
  }
}

void DisortSolver::computeIsotropicSource(int azimuth_mode) 
{
  /*
   * Compute isotropic (thermal) source particular solution for all layers.
   * Based on c_upisot() from cdisort.c:5503-5567
   *
   * Solves for Z₀ and Z₁ vectors representing thermal emission:
   *   [I - C] Z₁ = (1-ω₀) × b₁
   *   [I - C] Z₀ = (1-ω₀) × b₀ + μᵢ × Z₁
   *
   * where b₀, b₁ are linear expansion coefficients of Planck function
   *
   * Only computed for azimuth_mode == 0 (thermal source is azimuth-independent)
   */

  // Thermal sources only for azimuth mode 0
  if (azimuth_mode != 0 || !config_->flags.use_thermal_emission) {
    return;
  }

  // Compute Planck function at each level (0 to num_layers)
  std::vector<double> pkag(nlyr_ + 1);
  for (int lev = 0; lev <= nlyr_; ++lev) {
    // temperature is 1-based: temperature[1] = top, temperature[num_layers+1] = bottom
    pkag[lev] = planckFunction2(config_->wavenumber_low, config_->wavenumber_high, config_->temperature[lev]);
  }

  // Loop over all computational layers
  // Each layer has its own [I - C] matrix (stored per layer in phase_matrix_[lc])
  for (int lc = 0; lc < nlyr_; ++lc) {
    // Build per-layer coefficient matrix: A = I - C_lc  (vectorized, no heap alloc)
    isot_A_.noalias() = -phase_matrix_[lc];
    isot_A_.diagonal().array() += 1.0;
    // Factor once per layer; reuse for both Z₁ and Z₀ solves
    isot_lu_.compute(isot_A_);

    // Compute thermal expansion coefficients (mirrors C lines 811-815)
    // Linear approximation: B(τ) = xr0 + xr1 × τ
    double xr1 = 0.0;
    if (scaled_dtau_[lc] > 1e-4) {
      xr1 = (pkag[lc + 1] - pkag[lc]) / scaled_dtau_[lc];
    }
    double xr0 = pkag[lc] - xr1 * scaled_tau_cumulative_[lc];

    // Store for use in interpSource (thermal user-angle computation) and computeFluxes
    planck_intercept_[lc] = xr0;
    planck_slope_[lc] = xr1;

    // --- Solve for Z₁: b₁ is a constant vector ---
    isot_b_.fill((1.0 - scaled_ssa_[lc]) * xr1);
    isot_z1_.noalias() = isot_lu_.solve(isot_b_);

    // --- Solve for Z₀ ---
    isot_b_.array() = (1.0 - scaled_ssa_[lc]) * xr0 + quad_angle_.array() * isot_z1_.array();
    isot_z0_.noalias() = isot_lu_.solve(isot_b_);

    // Store solutions with angular reordering for symmetry
    // C code: ZPLK0(nn+iq, lc) = Z0(iq) for iq=1..nn
    //         ZPLK0(nn-iq+1, lc) = Z0(iq+nn) for iq=1..nn
    // Same pattern for Z1
    for (int iq = 0; iq < nn_; ++iq) {
      thermal_particular_z0_[lc](nn_ + iq)     = isot_z0_(iq);          // Downward streams
      thermal_particular_z0_[lc](nn_ - iq - 1) = isot_z0_(iq + nn_);    // Upward streams (reversed)

      thermal_particular_z1_[lc](nn_ + iq)     = isot_z1_(iq);
      thermal_particular_z1_[lc](nn_ - iq - 1) = isot_z1_(iq + nn_);
    }
  }
}

/**
 * @brief Convert LINPACK banded storage to dense matrix
 *
 * LINPACK banded storage format stores a band matrix in rectangular array.
 * For a matrix with ml subdiagonals and mu superdiagonals:
 * - Element A(i,j) is stored at cband(ml+mu+i-j, j) for 1-based indexing
 * - In 0-based C++: Element A(i,j) is at cband(ml+mu+i-j, j)
 *
 * @param cband Banded matrix in LINPACK format (lda × ncol)
 * @param ncol Number of columns (unknowns)
 * @param ml Number of subdiagonals
 * @param mu Number of superdiagonals
 * @return Dense matrix (ncol × ncol)
 */
// Band solver: LAPACK dgbtrf/dgbtrs wrappers
using detail::bandFactor;
using detail::bandSolve;


void DisortSolver::solve0(int azimuth_mode, double kronecker_mode_0) 
{
  /*
   * Solve boundary value problem for integration constants.
   * Based on c_solve0() from cdisort.c:4231-4553
   *
   * Constructs RHS from boundary conditions and solves:
   *   A * L = B
   * where A is the banded coefficient matrix (from setMatrix)
   *       B is the RHS vector (boundary conditions + particular solutions)
   *       L are the integration constants
   *
   * Phase 7: Now implements proper banded matrix solver.
   */

  (void)kronecker_mode_0;  // Reserved for future BRDF azimuthal correction

  const int ncol = nstr_ * ncut_;  // Total number of unknowns
  // Reuse pre-allocated RHS vector (solve0_b_ sized to nstr_*nlyr_ >= ncol)
  Eigen::VectorXd& b = solve0_b_;
  b.head(ncol).setZero();

  // ========================================================================
  // Construct RHS Vector from Boundary Conditions
  // ========================================================================

  const double isotropic_flux_top = config_->bc.isotropic_flux_top;
  const double direct_beam_flux = config_->bc.direct_beam_flux;
  const double direct_beam_mu = config_->bc.direct_beam_mu;

  if (azimuth_mode > 0) {
    // Azimuth-dependent case (direct_beam_flux > 0, plane-parallel geometry).
    // Mirrors c_solve0() from cdisort.c:4287-4392.
    //
    // For Lambertian surface (azimuth_mode > 0): BDR = 0 (no azimuthal dependence),
    // so only beam particular solution contributes to the boundaries.
    // Reference: STWJ eqs. 20a,c (lyrcut=FALSE, use_lambertian_surface=TRUE branch).
    if (direct_beam_flux > 0.0) {
      const int ncut = ncut_;

      // Top boundary: no downward diffuse radiation from above
      // C: B(iq) = -ZZ(nn+1-iq, 1)  for iq=1..nn
      for (int iq = 0; iq < nn_; ++iq) {
        b(iq) = -beam_particular_[0](nn_ - 1 - iq);
      }

      // Bottom boundary: C code STWJ(20c) for azimuth_mode>0
      // Lambertian: BDR=0 for azimuth_mode>0 → only particular solution subtracted.
      // BRDF: add diffuse+direct beam reflection. C: lines 4326-4369.
      for (int iq = 0; iq < nn_; ++iq) {
        int idx = ncol - nn_ + iq;
        if (lyrcut_ || config_->flags.use_lambertian_surface) {
          // Lambertian BDR=0 for azimuth_mode>0 (or truncated layer)
          if (config_->flags.use_spherical_beam) {
            // Spher: use ZBEAM0+ZBEAM1*tau expansion at bottom of ncut
            b(idx) = -std::exp(-beam_particular_atten_[ncut - 1] * scaled_tau_cumulative_[ncut])
                  * (beam_particular_[ncut - 1](nn_ + iq)
                   + beam_particular_slope_[ncut - 1](nn_ + iq) * scaled_tau_cumulative_[ncut]);
          } else {
            b(idx) = -beam_particular_[ncut - 1](nn_ + iq) * beam_transmission_[ncut];
          }
        } else {
          // BRDF: sum over diffuse streams + direct beam term
          // C: B(ncol-nn+iq) = sum_jq CWT(jq)*CMU(jq)*BDR(iq,jq)*ZZ(nn+1-jq,ncut)*EXPBEA(ncut)
          //                  + (BDR(iq,0)*direct_beam_mu*direct_beam_flux/PI - ZZ(iq+nn,ncut))*EXPBEA(ncut)
          // Note: use_spherical_beam+BRDF+azimuth_mode>0 not fully implemented in C (issues warning only).
          double sum = 0.0;
          for (int jq = 0; jq < nn_; ++jq) {
            sum += quad_weight_(jq) * quad_angle_(jq) * surface_refl_quad_(iq, jq + 1)
                 * beam_particular_[ncut - 1](nn_ - 1 - jq) * beam_transmission_[ncut];
          }
          b(idx) = sum + (surface_refl_quad_(iq, 0) * direct_beam_mu * direct_beam_flux / M_PI
                  - beam_particular_[ncut - 1](nn_ + iq)) * beam_transmission_[ncut];
        }
      }

      // Layer interface continuity: STWJ(20b)
      // C non-use_spherical_beam: B(++it) = (ZZ(iq,lc+1)-ZZ(iq,lc))*EXPBEA(lc)
      // C use_spherical_beam:     B(++it) = exp(-ZBEAMA(lc+1)*TAUCPR(lc))*(ZBEAM0(iq,lc+1)+ZBEAM1*TAUCPR(lc))
      //                      - exp(-ZBEAMA(lc)*TAUCPR(lc))*(ZBEAM0(iq,lc)+ZBEAM1*TAUCPR(lc))
      for (int lc = 1; lc < ncut; ++lc) {
        int ipnt = lc * nstr_ - nn_;
        for (int iq = 0; iq < nstr_; ++iq) {
          if (config_->flags.use_spherical_beam) {
            b(ipnt + iq) =
              std::exp(-beam_particular_atten_[lc] * scaled_tau_cumulative_[lc])
              * (beam_particular_[lc](iq) + beam_particular_slope_[lc](iq) * scaled_tau_cumulative_[lc])
              - std::exp(-beam_particular_atten_[lc - 1] * scaled_tau_cumulative_[lc])
              * (beam_particular_[lc - 1](iq) + beam_particular_slope_[lc - 1](iq) * scaled_tau_cumulative_[lc]);
          } else {
            b(ipnt + iq) = (beam_particular_[lc](iq) - beam_particular_[lc - 1](iq)) * beam_transmission_[lc];
          }
        }
      }
    }

    // else: direct_beam_flux == 0 → b stays all zeros
  } else {
    // Azimuth-independent case (azimuth_mode == 0)

    // --- Top Boundary Condition ---
    for (int iq = 0; iq < nn_; ++iq) {
      double sum = 0.0;

      // Beam source contribution
      if (direct_beam_flux > 0.0) {
        sum -= beam_particular_[0](nn_ - iq - 1);  // Top layer, upward streams
      }

      // Thermal source contribution
      if (config_->flags.use_thermal_emission) {
        sum -= thermal_particular_z0_[0](nn_ - iq - 1);
      }

      // Incident isotropic radiation + top boundary thermal emission
      sum += isotropic_flux_top + planck_top_;

      b(iq) = sum;
    }

    // --- Bottom Boundary Condition ---
    // STWJ(20c): B = 2*sum + (BDR(0)*direct_beam_mu*direct_beam_flux/pi - ZZ(iq+nn))*expbea - ZPLK + isotropic_flux_bottom
    // where sum = Σ_jq CWT(jq)*CMU(jq)*surface_albedo*(ZZ(nn-1-jq)*expbea + thermal_particular_z0_+thermal_particular_z1_*tau)
    // BDR(iq,jq)=surface_albedo for Lambertian; BDR(iq,0)=surface_albedo; /pi only for direct beam
    const int ncut = ncut_;  // Last layer (0-based)
    // Bottom boundary condition: STWJ(20c), C code lines 4494-4508.
    // Uses surface_refl_quad_(iq, jq+1) for diffuse, surface_refl_quad_(iq, 0) for direct beam, surface_emis_quad_(iq) for emission.
    // Both Lambertian and BRDF are handled via surface_refl_quad_/surface_emis_quad_ (filled by computeSurfaceBidir).
    for (int iq = 0; iq < nn_; ++iq) {
      int idx = ncol - nn_ + iq;  // Bottom BC starts at end of vector
      double sum = 0.0;

      // Diffuse particular solution reflection sum (iq-dependent for BRDF)
      if (!lyrcut_) {
        double refl_sum = 0.0;
        for (int jq = 0; jq < nn_; ++jq) {
          double ps_contrib = 0.0;
          if (direct_beam_flux > 0.0) {
            if (config_->flags.use_spherical_beam) {
              // Spher: exp(-ZBEAMA*TAUCPR)*(ZBEAM0+ZBEAM1*TAUCPR)
              ps_contrib += std::exp(-beam_particular_atten_[ncut - 1] * scaled_tau_cumulative_[ncut])
                      * (beam_particular_[ncut - 1](nn_ - 1 - jq)
                       + beam_particular_slope_[ncut - 1](nn_ - 1 - jq) * scaled_tau_cumulative_[ncut]);
            } else {
              ps_contrib += beam_particular_[ncut - 1](nn_ - 1 - jq) * beam_transmission_[ncut];
            }
          }
          if (config_->flags.use_thermal_emission) {
            ps_contrib += thermal_particular_z0_[ncut - 1](nn_ - 1 - jq)
                  + thermal_particular_z1_[ncut - 1](nn_ - 1 - jq) * scaled_tau_cumulative_[ncut];
          }
          refl_sum += quad_weight_(jq) * quad_angle_(jq) * surface_refl_quad_(iq, jq + 1) * ps_contrib;
        }
        sum += 2.0 * refl_sum;
      }

      // Beam source contribution at bottom
      if (direct_beam_flux > 0.0) {
        if (config_->flags.use_spherical_beam) {
          // Spher: exp(-ZBEAMA*TAUCPR)*(ZBEAM0+ZBEAM1*TAUCPR)
          sum -= std::exp(-beam_particular_atten_[ncut - 1] * scaled_tau_cumulative_[ncut])
               * (beam_particular_[ncut - 1](nn_ + iq)
                + beam_particular_slope_[ncut - 1](nn_ + iq) * scaled_tau_cumulative_[ncut]);
        } else {
          sum -= beam_particular_[ncut - 1](nn_ + iq) * beam_transmission_[ncut];
        }
        if (!lyrcut_) {
          // BDR(iq,0): direct beam reflection (same for use_spherical_beam and non-use_spherical_beam)
          sum += surface_refl_quad_(iq, 0) * direct_beam_mu * direct_beam_flux * beam_transmission_[ncut] / M_PI;
        }
      }

      // Thermal source at bottom
      if (config_->flags.use_thermal_emission) {
        sum -= thermal_particular_z0_[ncut - 1](nn_ + iq);
        sum -= thermal_particular_z1_[ncut - 1](nn_ + iq) * scaled_tau_cumulative_[ncut];
      }

      // Bottom boundary emission and fluorescence
      // BEM(iq) = 1 - surface_albedo for Lambertian; directional emissivity for BRDF
      sum += config_->bc.isotropic_flux_bottom;
      if (config_->flags.use_thermal_emission) {
        sum += surface_emis_quad_(iq) * planck_bottom_;
      }

      b(idx) = sum;
    }

    // --- Layer Interface Continuity ---
    for (int lc = 1; lc < ncut; ++lc) {
      int ipnt = lc * nstr_ - nn_;

      for (int iq = 0; iq < nstr_; ++iq) {
        double sum = 0.0;

        // Beam source contribution (difference across interface)
        if (direct_beam_flux > 0.0) {
          if (config_->flags.use_spherical_beam) {
            // Spher: use ZBEAM0+ZBEAM1*tau expansion at interface tau=TAUCPR(lc)
            // C: exp(-ZBEAMA(lc+1)*TAUCPR(lc))*(ZBEAM0(iq,lc+1)+ZBEAM1*TAUCPR(lc))
            //   -exp(-ZBEAMA(lc)  *TAUCPR(lc))*(ZBEAM0(iq,lc)  +ZBEAM1*TAUCPR(lc))
            sum += std::exp(-beam_particular_atten_[lc] * scaled_tau_cumulative_[lc])
                 * (beam_particular_[lc](iq) + beam_particular_slope_[lc](iq) * scaled_tau_cumulative_[lc]);
            sum -= std::exp(-beam_particular_atten_[lc - 1] * scaled_tau_cumulative_[lc])
                 * (beam_particular_[lc - 1](iq) + beam_particular_slope_[lc - 1](iq) * scaled_tau_cumulative_[lc]);
          } else {
            sum += beam_particular_[lc](iq) * beam_transmission_[lc];
            sum -= beam_particular_[lc - 1](iq) * beam_transmission_[lc];
          }
        }

        // Thermal source contribution
        if (config_->flags.use_thermal_emission) {
          sum += thermal_particular_z0_[lc](iq) - thermal_particular_z0_[lc - 1](iq);
          sum += (thermal_particular_z1_[lc](iq) - thermal_particular_z1_[lc - 1](iq)) * scaled_tau_cumulative_[lc];
        }

        b(ipnt + iq) = sum;
      }
    }
  }

  // ========================================================================
  // Solve Linear System: A * L = B
  // ========================================================================

  // Solve banded linear system using in-place LU factorization.
  // Based on c_solve0() calls to c_sgbco() and c_sgbsl() (cdisort.c:4525-4536)
  // This replaces the previous bandedToDense + PartialPivLU approach.

  const int ncd = 3 * nn_ - 1;  // Number of sub/super diagonals
  const int lda = band_matrix_.rows();

  // Factor band_matrix_ in-place (overwrites with LU); fills band_pivot_[0..ncol-1]
  bandFactor(band_matrix_.data(), lda, ncol, ncd, ncd, band_pivot_.data());

  // Solve: b.data()[0..ncol-1] is RHS on entry, solution on exit
  bandSolve(band_matrix_.data(), lda, ncol, ncd, ncd, band_pivot_.data(), b.data());

  // Extract integration constants from solution (b now holds X) with reordering.
  // Based on cdisort.c:4544-4550
  // LL(nn-iq+1, lc) = B(ipnt-iq+1)  for iq = 1 to nn
  // LL(nn+iq,   lc) = B(ipnt+iq)    for iq = 1 to nn

  integration_constants_.setZero();  // Initialize to zero first

  for (int lc = 0; lc < ncut_; ++lc) {
    // ipnt = lc * nstr_ - nn_ + 1 (C: ipnt = lc*num_streams-nn, 1-based)
    // But in 0-based C++: ipnt = (lc+1) * num_streams - nn - 1
    int ipnt = (lc + 1) * nstr_ - nn_ - 1;

    // Downward streams (iq = 0 to nn-1, corresponding to C's iq = 1 to nn)
    for (int iq = 0; iq < nn_; ++iq) {
      // C: LL(nn-iq+1, lc) = B(ipnt-iq+1)
      // C++ 0-based: integration_constants_(index) = b(ipnt - iq)
      int ll_idx = lc * nstr_ + (nn_ - iq - 1);  // Column-major: layer*num_streams + angle
      int b_idx = ipnt - iq;
      if (b_idx >= 0 && b_idx < ncol && ll_idx < integration_constants_.size()) {
        integration_constants_(ll_idx) = b(b_idx);
      }
    }

    // Upward streams (iq = 0 to nn-1, corresponding to C's iq = 1 to nn)
    for (int iq = 0; iq < nn_; ++iq) {
      // C: LL(nn+iq, lc) = B(ipnt+iq)
      // C++ 0-based: integration_constants_(index) = b(ipnt + iq + 1)
      int ll_idx = lc * nstr_ + (nn_ + iq);
      int b_idx = ipnt + iq + 1;
      if (b_idx >= 0 && b_idx < ncol && ll_idx < integration_constants_.size()) {
        integration_constants_(ll_idx) = b(b_idx);
      }
    }
  }

}

// ============================================================================
// Flux Computation
// ============================================================================

void DisortSolver::computeFluxes(DisortResult& result) 
{
  /*
   * Compute radiative fluxes and mean intensities at user output levels.
   * Based on c_fluxes() from cdisort.c:2307-2478
   *
   * Only called for azimuth_mode=0 (azimuthally averaged).
   */

  const double direct_beam_flux = config_->bc.direct_beam_flux;
  const double direct_beam_mu = config_->bc.direct_beam_mu;

  // Precomputed exp coefficients: exp_coeffs(jq) = LL(jq) * exp(...)
  // Factored out of the iq loop — only depends on eigenmode jq, not angle iq.
  Eigen::VectorXd exp_coeffs(nstr_);

  // Loop over user output levels
  for (int lu = 0; lu < ntau_; ++lu) {
    const int lyu = layer_of_user_level_[lu];

    // Check if this level is reachable
    if (lyrcut_ && lyu > ncut_ - 1) {
      result.flux_direct_beam[lu] = 0.0;
      result.flux_down[lu] = 0.0;
      result.flux_up[lu] = 0.0;
      result.flux_tau_divergence[lu] = 0.0;
      result.mean_intensity[lu] = 0.0;
      result.mean_intensity_down[lu] = 0.0;
      result.mean_intensity_up[lu] = 0.0;
      result.mean_intensity_direct_beam[lu] = 0.0;
      continue;
    }

    const double utaupr = scaled_tau_user_[lu];

    // Direct Beam Flux
    double fact = 0.0;
    double dirint = 0.0;
    double fldir = 0.0;
    double flux_direct_beam = 0.0;

    if (direct_beam_flux > 0.0) {
      if (config_->flags.use_spherical_beam) {
        const double ch_lyu = chapman_[lyu];
        fact   = std::exp(-utaupr / ch_lyu);
        flux_direct_beam = std::abs(direct_beam_mu) * direct_beam_flux * std::exp(-config_->tau_user[lu] / ch_lyu);
      } else {
        fact   = std::exp(-utaupr / direct_beam_mu);
        flux_direct_beam = direct_beam_mu * direct_beam_flux * std::exp(-config_->tau_user[lu] / direct_beam_mu);
      }
      dirint = direct_beam_flux * fact;
      fldir  = direct_beam_mu * direct_beam_flux * fact;
    }

    // Precompute exp() * integration_constants for all eigenmodes (once per level).
    // These only depend on jq and lyu, not on angle iq — saves nstr-1 redundant exp() per mode.
    const int ll_offset = lyu * nstr_;
    const double tau_bot = scaled_tau_cumulative_[lyu + 1];
    const double tau_top = scaled_tau_cumulative_[lyu];
    for (int jq = 0; jq < nn_; ++jq) {
      exp_coeffs(jq) = integration_constants_(jq + ll_offset)
                        * std::exp(-eigenvalues_[lyu](jq) * (utaupr - tau_bot));
    }
    for (int jq = nn_; jq < nstr_; ++jq) {
      exp_coeffs(jq) = integration_constants_(jq + ll_offset)
                        * std::exp(-eigenvalues_[lyu](jq) * (utaupr - tau_top));
    }

    // Compute diffuse intensity at all quadrature angles via matrix-vector product
    Eigen::VectorXd intensity = eigenvectors_[lyu] * exp_coeffs;

    // Add particular solutions
    if (direct_beam_flux > 0.0) {
      if (config_->flags.use_spherical_beam) {
        const double beam_exp = std::exp(-beam_particular_atten_[lyu] * utaupr);
        for (int iq = 0; iq < nstr_; ++iq)
          intensity(iq) += beam_exp * (beam_particular_[lyu](iq) + beam_particular_slope_[lyu](iq) * utaupr);
      } else {
        for (int iq = 0; iq < nstr_; ++iq)
          intensity(iq) += beam_particular_[lyu](iq) * fact;
      }
    }

    if (config_->flags.use_thermal_emission) {
      for (int iq = 0; iq < nstr_; ++iq)
        intensity(iq) += thermal_particular_z0_[lyu](iq) + thermal_particular_z1_[lyu](iq) * utaupr;
    }

    // Accumulate fluxes and mean intensities from quadrature
    double mean_intensity = 0.0;
    double mean_intensity_down = 0.0;
    double mean_intensity_up = 0.0;
    double fldn = 0.0;
    double flux_up = 0.0;

    // Downward angles (iq = 0 to nn-1)
    for (int iq = 0; iq < nn_; ++iq) {
      const int cwt_idx = nn_ - iq - 1;
      const double val = intensity(iq);
      const double wt = quad_weight_(cwt_idx);
      mean_intensity += wt * val;
      mean_intensity_down += wt * val;
      fldn += wt * val * quad_angle_(cwt_idx);
    }

    // Upward angles (iq = nn to num_streams-1)
    for (int iq = nn_; iq < nstr_; ++iq) {
      const int cwt_idx = iq - nn_;
      const double val = intensity(iq);
      const double wt = quad_weight_(cwt_idx);
      mean_intensity += wt * val;
      mean_intensity_up += wt * val;
      flux_up += wt * val * quad_angle_(cwt_idx);
    }

    // Finalize
    flux_up *= 2.0 * M_PI;
    fldn *= 2.0 * M_PI;

    const double fdntot = fldn + fldir;
    const double flux_down = fdntot - flux_direct_beam;

    mean_intensity = (2.0 * M_PI * mean_intensity + dirint) / (4.0 * M_PI);
    const double mean_intensity_direct_beam = dirint / (4.0 * M_PI);
    mean_intensity_down = (2.0 * M_PI * mean_intensity_down) / (4.0 * M_PI);
    mean_intensity_up = (2.0 * M_PI * mean_intensity_up) / (4.0 * M_PI);

    double plsorc = 0.0;
    if (config_->flags.use_thermal_emission) {
      plsorc = planck_intercept_[lyu] + planck_slope_[lyu] * utaupr;
    }

    const double flux_tau_divergence = (1.0 - config_->single_scat_albedo[lyu]) * 4.0 * M_PI * (mean_intensity - plsorc);

    // Store results
    result.flux_direct_beam[lu] = flux_direct_beam;
    result.flux_down[lu] = flux_down;
    result.flux_up[lu] = flux_up;
    result.flux_tau_divergence[lu] = flux_tau_divergence;
    result.mean_intensity[lu] = mean_intensity;
    result.mean_intensity_down[lu] = mean_intensity_down;
    result.mean_intensity_up[lu] = mean_intensity_up;
    result.mean_intensity_direct_beam[lu] = mean_intensity_direct_beam;
  }
}

// ============================================================================
// Chapman Function (Spherical Geometry)
// ============================================================================

// File-static helper so it can be called with either scaled or unscaled dtau.
// Port of c_chapman() from cdisort.c:9766-9840.
//
// Parameters:
//   lc         - Layer index (1-based: 1 = topmost layer)
//   tau_frac   - Fractional position within layer (0 = bottom, 0.5 = middle, 1 = top)
//   level_altitudes         - Altitude at level j+1: level_altitudes[1]=TOA, level_altitudes[num_layers+1]=0 (OneBased, 1-based)
//   dtau_lyr   - Per-layer optical depth (OneBased, 1-based, scaled or unscaled)
//   num_layers       - Number of layers
//   r          - Planet bottom_radius (same units as level_altitudes)
//   direct_beam_mu       - cos(solar zenith angle); direct_beam_mu > 0 means sun above horizon
//
// The function computes the integrated optical path length through the curved atmosphere
// from the observation point (at tau_frac in layer lc) to the top of the atmosphere.
// For zenith angles <= 90 deg this is a single path; for > 90 deg it accounts for
// the ray bending below the horizon.
static double chapmanImpl(int lc, double tau_frac,
              const std::vector<double>& level_altitudes,
              const std::vector<double>& dtau_lyr,
              int num_layers, double r, double direct_beam_mu)
{
  // Solar zenith angle in degrees: acos(direct_beam_mu)*180/pi
  // direct_beam_mu<0 → zenang > 90 (sun below horizon)
  const double DEG     = M_PI / 180.0;
  const double zenang  = std::acos(direct_beam_mu) * 180.0 / M_PI;   // degrees, 0..180
  const double zenrad  = zenang * DEG;

  // ZD(j) = altitude at level j (0-based: j=0 top of atmosphere, j=num_layers surface).
  // Altitude at top of 1-based layer lc: ZD(lc-1). Altitude at bottom: ZD(lc).
  auto ZD      = [&](int j) -> double { return level_altitudes[j]; };
  // DTAU_C(j) = dtau_lyr[j-1] (0-based std::vector, j iterates 1..id)
  auto DTAU_C  = [&](int j) -> double { return dtau_lyr[j - 1]; };

  // Radial distance at the evaluation point within layer lc
  // tau_frac=0 → bottom of lc (ZD(lc)), tau_frac=1 → top of lc (ZD(lc-1))
  const double xp     = r + ZD(lc) + (ZD(lc-1) - ZD(lc)) * tau_frac;
  const double xpsinz = xp * std::sin(zenrad);

  if (zenang > 90.0 && xpsinz < r) {
    return 1.0e+20;   // Below the horizon, effectively opaque
  }

  // Find the deepest layer that the ray passes through (relevant for zenang > 90)
  int id = lc;
  if (zenang > 90.0) {
    for (int j = lc; j <= num_layers; ++j) {
      if (xpsinz < (ZD(j-1) + r) && xpsinz >= (ZD(j) + r)) {
        id = j;
      }
    }
  }

  // Accumulate path length weighted by optical depth (eq. B2 in DS)
  double sum = 0.0;
  for (int j = 1; j <= id; ++j) {
    double fact  = 1.0;
    double fact2 = 1.0;
    // Factor 2 for second sum when sun is below horizon (zenang > 90)
    if (j > lc) {
      fact = 2.0;
    } else if (j == lc && lc == id && zenang > 90.0) {
      fact2 = -1.0;
    }

    const double rj   = r + ZD(j-1);
    double       rjp1 = r + ZD(j);
    if (j == lc && id == lc) {
      rjp1 = xp;   // Truncate at evaluation point
    }

    const double dhj = ZD(j-1) - ZD(j);   // Physical layer thickness
    double dsj;
    if (id > lc && j == id) {
      dsj = std::sqrt(rj*rj - xpsinz*xpsinz);
    } else {
      dsj = std::sqrt(rj*rj  - xpsinz*xpsinz)
        - fact2 * std::sqrt(rjp1*rjp1 - xpsinz*xpsinz);
    }
    sum += DTAU_C(j) * fact * dsj / dhj;
  }

  // Third term in eq. B2: partial path within layer lc when ray dips below lc
  if (id > lc) {
    const double dhj = ZD(lc-1) - ZD(lc);
    const double dsj = std::sqrt(xp*xp      - xpsinz*xpsinz)
             - std::sqrt((ZD(lc)+r)*(ZD(lc)+r) - xpsinz*xpsinz);
    sum += DTAU_C(lc) * dsj / dhj;
  }

  return sum;
}

double DisortSolver::chapmanFunction(int lc, double tau_frac) 
{
  // Port of c_chapman() using delta-M-scaled optical depths.
  return chapmanImpl(lc, tau_frac,
             config_->level_altitudes, scaled_dtau_,
             nlyr_, config_->bottom_radius, config_->bc.direct_beam_mu);
}

// ============================================================================
// User-Angle Intensity Output
// ============================================================================

void DisortSolver::interpEigenvec(int lc, int azimuth_mode) 
{
  /*
   * Interpolate eigenvectors from Gauss quadrature angles to user angles.
   * Port of c_interp_eigenvec() from cdisort.c:4766-4812.
   *
   * Rewritten as two DGEMM calls using pre-computed interp_ylmc_cwt_:
   *   W(l, iq)  = sum_jq [ ylmc_cwt(l,jq) * reduced_eigenvectors_(jq,iq) ]   (nact × nstr_)
   *             = interp_ylmc_cwt_.block(azimuth_mode,0,nact,nstr_) * reduced_eigenvectors_
   *   scale W row l by 0.5 * scaled_phase_coeffs_(l+azimuth_mode, lc)
   *   GU_tmp(iu, iq) = sum_l [ legendre_user_(l+azimuth_mode, iu) * W(l, iq) ] (numu_ × nstr_)
   *                  = legendre_user_.block(azimuth_mode,0,nact,numu_)^T * W
   * Then permute columns: eigenvec_user_[lc](iu, col) = GU_tmp(iu, iq)
   *
   * interp_ylmc_cwt_ is pre-computed once per azimuth_mode in solve() before the layer loop.
   */
  const int nact = nstr_ - azimuth_mode;  // Active Legendre orders (l = azimuth_mode..nstr_-1)

  // Step 1: W = ylmc_cwt_block * reduced_eigenvectors_  →  (nact × nstr_)
  interp_W_.topRows(nact).noalias() =
    interp_ylmc_cwt_.block(azimuth_mode, 0, nact, nstr_) * reduced_eigenvectors_;

  // Step 2: Scale row l (0-based offset from azimuth_mode) by 0.5 * scaled_phase_coeffs_(l+azimuth_mode, lc)
  for (int l = 0; l < nact; ++l)
    interp_W_.row(l) *= 0.5 * scaled_phase_coeffs_(l + azimuth_mode, lc);

  // Step 3: GU_tmp = ylmu_block^T * W  →  (numu_ × nstr_)
  interp_GU_tmp_.noalias() =
    legendre_user_.block(azimuth_mode, 0, nact, numu_).transpose() * interp_W_.topRows(nact);

  // Step 4: Permute columns into eigenvec_user_[lc] to match C storage convention:
  // C: iq<nn_ → col=nn_+iq; iq>=nn_ → col=nstr_-1-iq
  for (int iq = 0; iq < nstr_; ++iq) {
    const int col = (iq < nn_) ? (nn_ + iq) : (nstr_ - 1 - iq);
    eigenvec_user_[lc].col(col) = interp_GU_tmp_.col(iq);
  }
}

void DisortSolver::interpSource(int lc, int azimuth_mode, double kronecker_mode_0) 
{
  /*
   * Interpolate source functions to user angles (plane-parallel geometry).
   * Port of c_interp_source() from cdisort.c:4856-4974.
   *
   * Handles two source types:
   *   Beam:    fills beam_source_user_[lc]   (eq. SD(9), mirrors C lines 4910-4926)
   *   Thermal: fills thermal_source_z0_user_[lc], thermal_source_z1_user_[lc]  (STWJ 27c, mirrors C lines 4948-4971)
   */

  // --- Beam source (plane-parallel, direct_beam_flux > 0) ---
  if (config_->bc.direct_beam_flux > 0.0) {
    // Reuse pre-allocated scratch — avoids 96 Eigen heap allocations per solve
    Eigen::VectorXd& psi0 = interp_psi0_;
    psi0.setZero();

    // PSI0(l) = 0.5*GL(l,lc)*sum_jq(CWT(jq)*YLMC(l,jq)*ZJ(jq))
    for (int l = azimuth_mode; l < nstr_; ++l) {
      double psum = 0.0;
      for (int jq = 0; jq < nstr_; ++jq) {
        psum += quad_weight_(jq) * legendre_quad_(l, jq) * beam_source_raw_[lc](jq);
      }
      psi0(l) = 0.5 * scaled_phase_coeffs_(l, lc) * psum;  // scaled_phase_coeffs_ uses 1-based layer index
    }

    double fact = (2.0 - kronecker_mode_0) * config_->bc.direct_beam_flux / (4.0 * M_PI);

    for (int iu = 0; iu < numu_; ++iu) {
      double sum = 0.0;
      for (int l = azimuth_mode; l < nstr_; ++l) {
        sum += legendre_user_(l, iu) * (psi0(l) + fact * scaled_phase_coeffs_(l, lc) * legendre_beam_(l));
      }
      beam_source_user_[lc](iu) = sum;
    }
  }

  // --- Thermal source (azimuth_mode == 0, use_thermal_emission = true) ---
  // Mirrors c_interp_source() lines 4948-4971: STWJ(27c), STWL(31c)
  if (config_->flags.use_thermal_emission && azimuth_mode == 0 && !thermal_source_z0_user_.empty()) {
    Eigen::VectorXd psi0_th(nstr_), psi1_th(nstr_);
    psi0_th.setZero();
    psi1_th.setZero();

    // PSI0/PSI1 = 0.5*GL(l,lc)*sum_jq(CWT(jq)*YLMC(l,jq)*Z0/Z1(jq))
    // Z0(jq) is the unordered thermal solution (zee.zero in C).
    // In our storage, thermal_particular_z0_[lc] is the REORDERED version (ZPLK0 in C).
    // Inverse mapping from ZPLK0 → Z0 (0-based jq):
    //   jq = 0..nn_-1  → thermal_particular_z0_[lc][nn_ + jq]   (downward: ZPLK0(nn+jq+1))
    //   jq = nn_..nstr_-1 → thermal_particular_z0_[lc][nstr_-1-jq] (upward: ZPLK0(nn-(jq-nn_)))
    for (int l = azimuth_mode; l < nstr_; ++l) {
      double psum0 = 0.0, psum1 = 0.0;
      for (int jq = 0; jq < nstr_; ++jq) {
        double z0_jq = (jq < nn_) ? thermal_particular_z0_[lc](nn_ + jq) : thermal_particular_z0_[lc](nstr_ - 1 - jq);
        double z1_jq = (jq < nn_) ? thermal_particular_z1_[lc](nn_ + jq) : thermal_particular_z1_[lc](nstr_ - 1 - jq);
        psum0 += quad_weight_(jq) * legendre_quad_(l, jq) * z0_jq;
        psum1 += quad_weight_(jq) * legendre_quad_(l, jq) * z1_jq;
      }
      psi0_th(l) = 0.5 * scaled_phase_coeffs_(l, lc) * psum0;
      psi1_th(l) = 0.5 * scaled_phase_coeffs_(l, lc) * psum1;
    }

    // Z0U(iu, lc) = sum0 + (1 - OPRIM(lc)) * XR0(lc)
    // Z1U(iu, lc) = sum1 + (1 - OPRIM(lc)) * XR1(lc)
    for (int iu = 0; iu < numu_; ++iu) {
      double sum0 = 0.0, sum1 = 0.0;
      for (int l = azimuth_mode; l < nstr_; ++l) {
        sum0 += legendre_user_(l, iu) * psi0_th(l);
        sum1 += legendre_user_(l, iu) * psi1_th(l);
      }
      thermal_source_z0_user_[lc](iu) = sum0 + (1.0 - scaled_ssa_[lc]) * planck_intercept_[lc];
      thermal_source_z1_user_[lc](iu) = sum1 + (1.0 - scaled_ssa_[lc]) * planck_slope_[lc];
    }
  }
}

void DisortSolver::computeSurfaceBidir(double kronecker_mode_0, int azimuth_mode) 
{
  /*
   * Compute Fourier expansion coefficient of surface bidirectional reflectance.
   * Port of c_surface_bidir() from cdisort.c:4608-4754.
   *
   * Fills:
   *   surface_refl_quad_(iq, 0)       : direct-beam BDR at computational angle iq
   *   surface_refl_quad_(iq, jq+1)    : diffuse BDR (incoming stream jq, outgoing stream iq)
   *   surface_emis_quad_(iq)          : surface directional emissivity at computational angle iq
   *   surface_refl_user_(iu, iq)      : BDR at user outgoing angle iu, incoming stream iq
   *   surface_emis_user_(iu)          : emissivity at user angle iu
   *
   * Uses NMUG=200 Gauss-Legendre azimuthal quadrature with precomputed cosine series.
   */

  const double direct_beam_flux = config_->bc.direct_beam_flux;
  const double direct_beam_mu  = config_->bc.direct_beam_mu;

  surface_refl_quad_.setZero();
  surface_emis_quad_.setZero();

  if (config_->flags.use_lambertian_surface && azimuth_mode == 0) {
    // Lambertian: BDR = surface_albedo everywhere, BEM = 1 - surface_albedo
    for (int iq = 0; iq < nn_; ++iq) {
      surface_emis_quad_(iq) = 1.0 - config_->bc.surface_albedo;
      for (int jq = 0; jq <= nn_; ++jq) {
        surface_refl_quad_(iq, jq) = config_->bc.surface_albedo;
      }
    }
  } else if (!config_->flags.use_lambertian_surface) {
    // BRDF: compute Fourier coefficients via azimuthal quadrature.
    // Uses precomputed cosmp_brdf_(m, k) and integrates over positive half only (NMUG/2 points).
    // By symmetry: 0.5 * sum_{k=0}^{NMUG-1} = sum_{k=0}^{NMUG/2-1}.
    for (int iq = 0; iq < nn_; ++iq) {
      // Diffuse incoming streams (jq=0..nn-1 → columns 1..nn)
      for (int jq = 0; jq < nn_; ++jq) {
        double sum = 0.0;
        for (int k = 0; k < NMUG / 2; ++k) {
          sum += gwt_brdf_(k) *
               bidirReflectivity(quad_angle_(iq), quad_angle_(jq), M_PI * gmu_brdf_(k)) *
               cosmp_brdf_(azimuth_mode, k);
        }
        surface_refl_quad_(iq, jq + 1) = (2.0 - kronecker_mode_0) * sum;
      }
      // Direct beam (column 0)
      if (direct_beam_flux > 0.0) {
        double sum = 0.0;
        for (int k = 0; k < NMUG / 2; ++k) {
          sum += gwt_brdf_(k) *
               bidirReflectivity(quad_angle_(iq), direct_beam_mu, M_PI * gmu_brdf_(k)) *
               cosmp_brdf_(azimuth_mode, k);
        }
        surface_refl_quad_(iq, 0) = (2.0 - kronecker_mode_0) * sum;
      }
    }
    // Directional emissivity (only for azimuth_mode=0, C lines 4675-4693)
    if (azimuth_mode == 0) {
      for (int iq = 0; iq < nn_; ++iq) {
        double dref = 0.0;
        for (int jg = 0; jg < NMUG; ++jg) {
          double sum = 0.0;
          // Inner sum uses only positive half of quadrature (k=0..NMUG/2-1)
          for (int k = 0; k < NMUG / 2; ++k) {
            sum += gwt_brdf_(k) * gmu_brdf_(k) *
                 bidirReflectivity(quad_angle_(iq), gmu_brdf_(k), M_PI * gmu_brdf_(jg));
          }
          dref += gwt_brdf_(jg) * sum;
        }
        surface_emis_quad_(iq) = 1.0 - dref;
      }
    }
  }
  // else: Lambertian + azimuth_mode>0 → surface_refl_quad_ stays zero (correct: no azimuthal coupling)

  // Also fill surface_refl_user_ and surface_emis_user_ for user angles (needed for computeUserIntensities)
  if (!config_->flags.comp_only_fluxes && config_->flags.use_user_mu) {
    surface_refl_user_.setZero();
    surface_emis_user_.setZero();
    for (int iu = 0; iu < numu_; ++iu) {
      double umu_val = config_->mu_user[iu];  
      if (umu_val > 0.0) {
        if (config_->flags.use_lambertian_surface && azimuth_mode == 0) {
          for (int iq = 0; iq <= nn_; ++iq) {
            surface_refl_user_(iu, iq) = config_->bc.surface_albedo;
          }
          surface_emis_user_(iu) = 1.0 - config_->bc.surface_albedo;
        } else if (!config_->flags.use_lambertian_surface) {
          // Diffuse incoming streams
          for (int jq = 0; jq < nn_; ++jq) {
            double sum = 0.0;
            for (int k = 0; k < NMUG / 2; ++k) {
              sum += gwt_brdf_(k) *
                   bidirReflectivity(umu_val, quad_angle_(jq), M_PI * gmu_brdf_(k)) *
                   cosmp_brdf_(azimuth_mode, k);
            }
            surface_refl_user_(iu, jq + 1) = (2.0 - kronecker_mode_0) * sum;
          }
          // Direct beam
          if (direct_beam_flux > 0.0) {
            double sum = 0.0;
            for (int k = 0; k < NMUG / 2; ++k) {
              sum += gwt_brdf_(k) *
                   bidirReflectivity(umu_val, direct_beam_mu, M_PI * gmu_brdf_(k)) *
                   cosmp_brdf_(azimuth_mode, k);
            }
            surface_refl_user_(iu, 0) = (2.0 - kronecker_mode_0) * sum;
          }
          // Directional emissivity at user angles (azimuth_mode=0 only)
          if (azimuth_mode == 0) {
            double dref = 0.0;
            for (int jg = 0; jg < NMUG; ++jg) {
              double sum = 0.0;
              for (int k = 0; k < NMUG / 2; ++k) {
                sum += gwt_brdf_(k) * gmu_brdf_(k) *
                     bidirReflectivity(umu_val, gmu_brdf_(k),
                             M_PI * gmu_brdf_(jg));
              }
              dref += gwt_brdf_(jg) * sum;
            }
            surface_emis_user_(iu) = 1.0 - dref;
          }
        }
      }
    }
  }
}

double DisortSolver::bidirReflectivityHapke(double mu, double mup, double dphi) const 
{
  /*
   * Hapke BRDF model.
   * Port of BRDF_HAPKE from Fortran BDREF.f (Hapke 1993, Eq. 8.89).
   * Parameters are configurable via HapkeBrdfSpec; defaults: b0=1, hh=0.06, w=0.6.
   */
  double b0 = 1.0, hh = 0.06, w = 0.6;
  if (config_->brdf.hapke) {
    b0 = config_->brdf.hapke->b0;
    hh = config_->brdf.hapke->hh;
    w  = config_->brdf.hapke->w;
  }

  double ctheta = mu * mup + std::sqrt((1.0 - mu*mu) * (1.0 - mup*mup)) * std::cos(dphi);
  double thetah = std::acos(ctheta);
  double p      = 1.0 + 0.5 * ctheta;
  double b      = b0 * hh / (hh + std::tan(0.5 * thetah));
  double Xgamm  = std::sqrt(1.0 - w);
  double h0     = (1.0 + 2.0*mup) / (1.0 + 2.0*Xgamm*mup);
  double h      = (1.0 + 2.0*mu ) / (1.0 + 2.0*Xgamm*mu );
  return 0.25 * w * ((1.0 + b)*p + h0*h - 1.0) / (mu + mup);
}

double DisortSolver::bidirReflectivityRpv(const RpvBrdfSpec& brdf,
                      double mu1, double mu2,
                      double phi, double badmu) const 
{
  /*
   * RPV (Rahman-Pinty-Verstraete) BRDF model.
   * Port of c_bidir_reflectivity_rpv() from cdisort.c:1346-1409.
   * Note: azimuth convention converted (phi=0 = backward in RPV, forward in DISORT).
   */
  // Azimuth convention: DISORT phi=0 is forward, RPV phi=0 is backward
  phi = M_PI - phi;

  // Clamp mu values if badmu > 0
  if (badmu > 0.0) {
    if (mu1 < badmu) mu1 = badmu;
    if (mu2 < badmu) mu2 = badmu;
  }

  // Hot spot
  double hspot = brdf.rho0 * (std::pow(2.0*mu1*mu1*mu1, brdf.k - 1.0) *
           (1.0 - brdf.theta) / ((1.0 + brdf.theta) * (1.0 + brdf.theta)) *
           (2.0 - brdf.rho0) + brdf.sigma / mu1) *
           (brdf.t1 * std::exp(M_PI * brdf.t2) + 1.0);

  // Hot spot region (exact match)
  if (phi == 1.0e-4 && mu1 == mu2) {
    return hspot * brdf.scale;
  }

  double m      = std::pow(mu1 * mu2 * (mu1 + mu2), brdf.k - 1.0);
  double cosphi = std::cos(phi);
  double sin1   = std::sqrt(1.0 - mu1*mu1);
  double sin2   = std::sqrt(1.0 - mu2*mu2);
  double cosg   = mu1*mu2 + sin1*sin2*cosphi;
  double g      = std::acos(cosg);
  double f      = (1.0 - brdf.theta*brdf.theta) /
          std::pow(1.0 + 2.0*brdf.theta*cosg + brdf.theta*brdf.theta, 1.5);
  double tan1   = sin1 / mu1;
  double tan2   = sin2 / mu2;
  double capg   = std::sqrt(tan1*tan1 + tan2*tan2 - 2.0*tan1*tan2*cosphi);
  double hval   = 1.0 + (1.0 - brdf.rho0) / (1.0 + capg);
  double t      = 1.0 + brdf.t1 * std::exp(brdf.t2 * (M_PI - g));

  double ans = brdf.rho0 * (m * f * hval + brdf.sigma / mu1) * t * brdf.scale;
  if (ans < 0.0) ans = 0.0;
  return ans;
}

double DisortSolver::bidirReflectivityRossLi(double mu, double mup, double dphi) const 
{
  /*
   * Ross-Li BRDF model (Ross-Thick volumetric + Li-Sparse-Reciprocal geometric kernels).
   * Port of BRDF_ROSSLI from Fortran DISORT V3 (BDREF.f:266-370).
   * Parameters: iso, vol, geo from AmbralsBrdfSpec.
   * Reference: Roujean et al. (1992), Li & Strahler (1992).
   */
  const auto& spec = *config_->brdf.ambrals;
  constexpr double RATIO_HB = 2.0;   // Height-to-base ratio of crown
  constexpr double RATIO_BR = 1.0;   // Branch-to-radius ratio
  constexpr double ALPHA0 = 1.5 * M_PI / 180.0;  // Hot-spot angular width (1.5 degrees)

  double sin_i = std::sqrt(1.0 - mup * mup);
  double sin_r = std::sqrt(1.0 - mu * mu);
  double tan_i = (mup > 0.0) ? sin_i / mup : 0.0;
  double tan_r = (mu > 0.0)  ? sin_r / mu  : 0.0;

  double cos_alpha = mup * mu + sin_i * sin_r * std::cos(dphi);
  cos_alpha = std::clamp(cos_alpha, -1.0, 1.0);
  double alpha = std::acos(cos_alpha);
  double sin_alpha = std::sin(alpha);

  // Ross-Thick volumetric kernel
  double c = 1.0 + 1.0 / (1.0 + alpha / ALPHA0);
  double f_vol = 4.0 / (3.0 * M_PI) * (1.0 / (mup + mu))
               * ((M_PI / 2.0 - alpha) * cos_alpha + sin_alpha) * c - 1.0 / 3.0;

  // Li-Sparse-Reciprocal geometric kernel
  double tan_i1 = RATIO_BR * tan_i;
  double tan_r1 = RATIO_BR * tan_r;
  double cos_i1 = 1.0 / std::sqrt(1.0 + tan_i1 * tan_i1);
  double cos_r1 = 1.0 / std::sqrt(1.0 + tan_r1 * tan_r1);
  double sin_i1 = tan_i1 * cos_i1;
  double sin_r1 = tan_r1 * cos_r1;

  double cos_alpha1 = cos_i1 * cos_r1 + sin_i1 * sin_r1 * std::cos(dphi);
  double g_sq = tan_i1 * tan_i1 + tan_r1 * tan_r1
              - 2.0 * tan_i1 * tan_r1 * std::cos(dphi);
  if (g_sq < 0.0) g_sq = 0.0;

  double cos_t = RATIO_HB * (cos_i1 * cos_r1) / (cos_i1 + cos_r1)
               * std::sqrt(g_sq + std::pow(tan_i1 * tan_r1 * std::sin(dphi), 2));

  double t = 0.0;
  if (cos_t >= -1.0 && cos_t <= 1.0) {
    t = std::acos(cos_t);
  }

  double f_geo = (cos_i1 + cos_r1) / (M_PI * cos_i1 * cos_r1)
               * (t - std::sin(t) * std::cos(t) - M_PI)
               + (1.0 + cos_alpha1) / (2.0 * cos_i1 * cos_r1);

  return std::max(spec.iso + spec.vol * f_vol + spec.geo * f_geo, 0.0);
}

double DisortSolver::shadowEta(double cos_theta, double sigma_sq) 
{
  /*
   * Tsang shadowing function for Cox-Munk BRDF.
   * Port of SHADOW_ETA from Fortran BDREF.f:448-474.
   */
  double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
  if (sin_theta < 1e-15) return 0.0;
  double cot_val = cos_theta / sin_theta;
  double term1 = std::sqrt(sigma_sq / M_PI) / cot_val
               * std::exp(-cot_val * cot_val / sigma_sq);
  double term2 = std::erfc(cot_val / std::sqrt(sigma_sq));
  return 0.5 * (term1 - term2);
}

double DisortSolver::bidirReflectivityCoxMunk(double mu, double mup, double dphi) const 
{
  /*
   * Cox-Munk ocean BRDF: 1D Gaussian rough surface with Fresnel reflectance.
   * Port of OCEABRDF2 from Fortran DISORT V3 (BDREF.f:374-444).
   * Reference: Cox & Munk (1954).
   */
  const auto& spec = *config_->brdf.cox_munk;

  double sin_i = std::sqrt(1.0 - mup * mup);
  double sin_r = std::sqrt(1.0 - mu * mu);
  double cos_theta = -mup * mu + sin_i * sin_r * std::cos(dphi);

  // Guard against specular/degenerate geometry
  double one_minus_cos = 1.0 - cos_theta;
  if (one_minus_cos < 1e-15) return 0.0;

  double mu_n_sq = (mup + mu) * (mup + mu) / (2.0 * one_minus_cos);
  if (mu_n_sq < 1e-15) return 0.0;

  // Slope variance from Cox-Munk empirical relation
  double sigma_sq = 0.003 + 0.00512 * spec.u10;

  // Gaussian slope probability
  double p_slope = 1.0 / (M_PI * sigma_sq) * std::exp(-(1.0 - mu_n_sq) / (sigma_sq * mu_n_sq));

  // Fresnel reflectance
  double n_i = 1.0;
  double n_t = spec.refractive_index;
  double cos_li = std::sqrt(0.5 * (1.0 + cos_theta));
  double sin_li = std::sqrt(std::max(0.0, 1.0 - cos_li * cos_li));
  double sin_lt = n_i * sin_li / n_t;
  if (sin_lt >= 1.0) return 0.0;  // Total internal reflection guard
  double cos_lt = std::sqrt(1.0 - sin_lt * sin_lt);

  double r_s = (n_i * cos_li - n_t * cos_lt) / (n_i * cos_li + n_t * cos_lt);
  double r_p = (n_t * cos_li - n_i * cos_lt) / (n_i * cos_lt + n_t * cos_li);
  double r_fresnel = 0.5 * (r_s * r_s + r_p * r_p);

  double brdf = (p_slope * r_fresnel) / (4.0 * mup * mu * mu_n_sq * mu_n_sq);

  if (spec.do_shadow) {
    double shadow = 1.0 / (shadowEta(mup, sigma_sq) + shadowEta(mu, sigma_sq) + 1.0);
    brdf *= shadow;
  }

  return std::max(brdf, 0.0);
}

double DisortSolver::bidirReflectivity(double mu, double mup, double dphi) const 
{
  /*
   * Dispatch to appropriate BRDF model.
   * Port of c_bidir_reflectivity() from cdisort.c:1073-1244.
   */
  switch (config_->flags.brdf_type) {
  case BrdfType::Hapke:
    return bidirReflectivityHapke(mu, mup, dphi);

  case BrdfType::RPV:
    if (!config_->brdf.rpv) {
      throw std::runtime_error("bidirReflectivity: RPV BRDF selected but rpv spec is null");
    }
    // Note: badmu=0.0 (no limit); RPV clamping handled internally if needed.
    // For full support matching C code, badmu should be computed from dref scan,
    // but 0.0 is safe for typical parameter values.
    return bidirReflectivityRpv(*config_->brdf.rpv, mu, mup, dphi, 0.0);

  case BrdfType::CoxMunk:
    if (!config_->brdf.cox_munk) {
      throw std::runtime_error("bidirReflectivity: Cox-Munk BRDF selected but cox_munk spec is null");
    }
    return bidirReflectivityCoxMunk(mu, mup, dphi);

  case BrdfType::Ambrals:
    if (!config_->brdf.ambrals) {
      throw std::runtime_error("bidirReflectivity: Ross-Li/Ambrals BRDF selected but ambrals spec is null");
    }
    return bidirReflectivityRossLi(mu, mup, dphi);

  default:
    throw std::runtime_error("bidirReflectivity: unknown BRDF type");
  }
}

double DisortSolver::computeUserIntensities(int azimuth_mode, double kronecker_mode_0,
                       DisortResult& result,
                       double bplanck, double tplanck) 
{
  /*
   * Compute intensities at user angles and accumulate azimuthal Fourier modes.
   * Port of c_user_intensities() from cdisort.c:5660-5992.
   * Returns max_azimuth_contrib (max relative azimuthal contribution) for azimuth_mode > 0,
   * or 0.0 for azimuth_mode == 0.
   *
   * Algorithm:
   * 1. Incorporate integration constants: GU(iu,iq,lc) *= LL(iq,lc)
   * 2. For each output level lu and user angle iu:
   *    a. Integrate characteristic equation across layers entirely
   *       above (downward angles) or below (upward angles) the output level
   *    b. Add contribution from partial layer containing the output level
   *    c. Add boundary contribution (top for downward, bottom for upward)
   * 3. Accumulate into result.intensity using cos(azimuth_mode*(phi_user-direct_beam_phi))
   *
   * NOTE: eigenvec_user_[lc] is modified in-place (multiplied by integration_constants_).
   *       It will be recomputed by interpEigenvec in the next azimuth_mode pass.
   */
  const double direct_beam_mu  = config_->bc.direct_beam_mu;
  const double direct_beam_flux = config_->bc.direct_beam_flux;
  const double isotropic_flux_top = config_->bc.isotropic_flux_top;
  const double isotropic_flux_bottom = config_->bc.isotropic_flux_bottom;

  // Step 1: Incorporate integration constants into GU
  // GU(iu,iq,lc) *= LL(iq,lc) for all lc,iq,iu
  // integration_constants_ is stored as integration_constants_[iq + lc*nstr_] (both 0-based)
  for (int lc = 0; lc < ncut_; ++lc) {
    for (int iq = 0; iq < nstr_; ++iq) {
      double ll_val = integration_constants_(iq + lc * nstr_);
      for (int iu = 0; iu < numu_; ++iu) {
        eigenvec_user_[lc](iu, iq) *= ll_val;
      }
    }
  }

  // Step 2: Loop over output levels
  for (int lu = 0; lu < ntau_; ++lu) {
    double exp0 = 0.0;
    if (direct_beam_flux > 0.0) {
      exp0 = exp(-scaled_tau_user_[lu] / direct_beam_mu);
    }

    int lyu = layer_of_user_level_[lu];

    // Step 3: Loop over user polar angles
    for (int iu = 0; iu < numu_; ++iu) {
      double mu_user = config_->mu_user[iu];  

      // Skip truncated layers
      if (lyrcut_ && lyu >= ncut_) {
        intensity_azim_mode_(iu, lu) = 0.0;
        continue;
      }

      bool negumu = (mu_user < 0.0);
      int lyrstr, lyrend;
      double sgn;
      if (negumu) {
        lyrstr = 0;
        lyrend = lyu - 1;  // layers 0..lyu-1 are entirely above output level
        sgn    = -1.0;
      } else {
        lyrstr = lyu + 1;  // layers lyu+1..ncut_-1 are entirely below output level
        lyrend = ncut_ - 1;
        sgn    =  1.0;
      }

      double palint = 0.0;  // Scattering + beam contribution
      double plkint = 0.0;  // Planck/thermal contribution
      double genint = 0.0;  // General source contribution

      // --- Integrate across full layers ---
      for (int lc = lyrstr; lc <= lyrend; ++lc) {
        // scaled_tau_cumulative_: 0-based vector, scaled_tau_cumulative_[lc] = top, scaled_tau_cumulative_[lc+1] = bottom
        double dtau = scaled_dtau_[lc];
        double exp1 = exp((scaled_tau_user_[lu] - scaled_tau_cumulative_[lc    ]) / mu_user);
        double exp2 = exp((scaled_tau_user_[lu] - scaled_tau_cumulative_[lc + 1]) / mu_user);

        // Planck (thermal) contribution - STWL(36b,c)
        // Uses Z0U/Z1U (thermal source at user angles), mirrors C lines 5753-5761
        if (config_->flags.use_thermal_emission && azimuth_mode == 0) {
          double f0n = sgn * (exp1 - exp2);
          double f1n = sgn * ((scaled_tau_cumulative_[lc    ] + mu_user) * exp1
                    -(scaled_tau_cumulative_[lc + 1] + mu_user) * exp2);
          plkint += thermal_source_z0_user_[lc](iu) * f0n + thermal_source_z1_user_[lc](iu) * f1n;
        }

        // Beam contribution (plane-parallel geometry)
        if (direct_beam_flux > 0.0) {
          double denom = 1.0 + mu_user / direct_beam_mu;
          double expn;
          if (std::abs(denom) < 0.0001) {
            // L'Hospital limit
            expn = (dtau / direct_beam_mu) * exp0;
          } else {
            expn = (exp1 * beam_transmission_[lc] - exp2 * beam_transmission_[lc + 1]) * sgn / denom;
          }
          palint += beam_source_user_[lc](iu) * expn;
        }

        // Eigenvector contributions
        // Negative eigenvalues (iq=0..nn_-1): eigenvalues_[lc](iq) < 0
        Eigen::VectorXd& wk_neg = ci_wk_neg_;  // pre-allocated; exp(kk_neg * dtau)
        for (int iq = 0; iq < nn_; ++iq) {
          wk_neg(iq) = exp(eigenvalues_[lc](iq) * dtau);  // eigenvalues_ < 0, so this < 1
          double denom = 1.0 + mu_user * eigenvalues_[lc](iq);
          double expn;
          if (std::abs(denom) < 0.0001) {
            expn = (dtau / mu_user) * exp2;
          } else {
            expn = sgn * (exp1 * wk_neg(iq) - exp2) / denom;
          }
          palint += eigenvec_user_[lc](iu, iq) * expn;
        }

        // Positive eigenvalues (iq=nn_..nstr_-1): eigenvalues_[lc](iq) > 0
        for (int iq = nn_; iq < nstr_; ++iq) {
          double denom = 1.0 + mu_user * eigenvalues_[lc](iq);
          // WK(num_streams+1-iq) in C maps to wk_neg[nstr_-1-iq] in 0-based C++
          double expn;
          if (std::abs(denom) < 0.0001) {
            expn = -(dtau / mu_user) * exp1;
          } else {
            expn = sgn * (exp1 - exp2 * wk_neg(nstr_ - 1 - iq)) / denom;
          }
          palint += eigenvec_user_[lc](iu, iq) * expn;
        }
      }

      // --- Integrate partial layer containing the output level ---
      // dtau1 = distance from top of layer lyu to output level
      // dtau2 = distance from bottom of layer lyu to output level
      double dtau1 = scaled_tau_user_[lu] - scaled_tau_cumulative_[lyu    ];
      double dtau2 = scaled_tau_user_[lu] - scaled_tau_cumulative_[lyu + 1];

      if ((std::abs(dtau1) >= 1e-6 || !negumu) && (std::abs(dtau2) >= 1e-6 || negumu)) {
        double exp1_curr = 0.0, exp2_curr = 0.0;
        if (negumu) {
          exp1_curr = exp(dtau1 / mu_user);
        } else {
          exp2_curr = exp(dtau2 / mu_user);
        }

        // Beam contribution in current layer
        if (direct_beam_flux > 0.0) {
          double denom = 1.0 + mu_user / direct_beam_mu;
          double expn;
          if (std::abs(denom) < 0.0001) {
            expn = (dtau1 / direct_beam_mu) * exp0;
          } else if (negumu) {
            expn = (exp0 - beam_transmission_[lyu    ] * exp1_curr) / denom;
          } else {
            expn = (exp0 - beam_transmission_[lyu + 1] * exp2_curr) / denom;
          }
          palint += beam_source_user_[lyu](iu) * expn;
        }

        double dtau_lyu = scaled_dtau_[lyu];

        // Negative eigenvalues in current layer
        Eigen::VectorXd& wk_curr_neg = ci_wk_curr_neg_;  // pre-allocated
        for (int iq = 0; iq < nn_; ++iq) {
          double denom = 1.0 + mu_user * eigenvalues_[lyu](iq);
          double expn;
          if (std::abs(denom) < 0.0001) {
            expn = -dtau2 / mu_user * exp2_curr;
          } else if (negumu) {
            // exp(-KK*dtau2) = exp(-kk_neg*dtau2) — kk_neg<0, dtau2<=0 for negumu
            expn = (exp(-eigenvalues_[lyu](iq) * dtau2)
                 - exp(eigenvalues_[lyu](iq) * dtau_lyu) * exp1_curr) / denom;
          } else {
            expn = (exp(-eigenvalues_[lyu](iq) * dtau2) - exp2_curr) / denom;
          }
          wk_curr_neg(iq) = exp(eigenvalues_[lyu](iq) * dtau_lyu);  // for positive loop
          palint += eigenvec_user_[lyu](iu, iq) * expn;
        }

        // Positive eigenvalues in current layer
        for (int iq = nn_; iq < nstr_; ++iq) {
          double denom = 1.0 + mu_user * eigenvalues_[lyu](iq);
          double expn;
          if (std::abs(denom) < 0.0001) {
            expn = -(dtau1 / mu_user) * exp1_curr;
          } else if (negumu) {
            expn = (exp(-eigenvalues_[lyu](iq) * dtau1) - exp1_curr) / denom;
          } else {
            expn = (exp(-eigenvalues_[lyu](iq) * dtau1)
                 - exp(-eigenvalues_[lyu](iq) * dtau_lyu) * exp2_curr) / denom;
          }
          palint += eigenvec_user_[lyu](iu, iq) * expn;
        }

        // Planck contribution in current layer (mirrors C lines 5929-5938)
        // Uses Z0U/Z1U (thermal source at user angles)
        if (config_->flags.use_thermal_emission && azimuth_mode == 0) {
          double expn, fact;
          if (negumu) {
            expn = exp1_curr;
            fact = scaled_tau_cumulative_[lyu    ] + mu_user;
          } else {
            expn = exp2_curr;
            fact = scaled_tau_cumulative_[lyu + 1] + mu_user;
          }
          double f0n = 1.0 - expn;
          double f1n = scaled_tau_user_[lu] + mu_user - fact * expn;
          plkint += thermal_source_z0_user_[lyu](iu) * f0n + thermal_source_z1_user_[lyu](iu) * f1n;
        }
      }

      // --- Boundary contribution ---
      double bndint = 0.0;
      if (negumu && azimuth_mode == 0) {
        // Top boundary: isotropic illumination + thermal emission
        bndint = (isotropic_flux_top + tplanck) * exp(scaled_tau_user_[lu] / mu_user);

      } else if (!negumu) {
        if (lyrcut_ || (config_->flags.use_lambertian_surface && azimuth_mode > 0)) {
          // No bottom-boundary contribution (lyrcut or higher Lambertian modes)
          intensity_azim_mode_(iu, lu) = palint + plkint;
          continue;
        }

        // Bottom-boundary Lambertian reflection contribution
        // Precompute exp(-kk_pos * dtau) for last layer
        Eigen::VectorXd& wk_bnd = ci_wk_bnd_;  // pre-allocated
        for (int jq = nn_; jq < nstr_; ++jq) {
          wk_bnd(jq - nn_) = exp(-eigenvalues_[nlyr_ - 1](jq) * scaled_dtau_[nlyr_ - 1]);
        }

        double bnddfu = 0.0;
        // Sum over first nn rows of GC at bottom layer (Lambertian reflection)
        // C: for(iq=nn; iq>=1; iq--) → C++: for(iq=nn_-1; iq>=0; --iq)
        for (int iq = nn_ - 1; iq >= 0; --iq) {
          double dfuint = 0.0;

          // Negative eigenvector contributions (jq=0..nn_-1)
          for (int jq = 0; jq < nn_; ++jq) {
            dfuint += eigenvectors_[nlyr_ - 1](iq, jq) * integration_constants_(jq + (nlyr_ - 1) * nstr_);
          }
          // Positive eigenvector contributions (jq=nn_..nstr_-1)
          for (int jq = nn_; jq < nstr_; ++jq) {
            dfuint += eigenvectors_[nlyr_ - 1](iq, jq)
                * integration_constants_(jq + (nlyr_ - 1) * nstr_)
                * wk_bnd(jq - nn_);
          }
          // Beam contribution to downward flux at bottom
          if (direct_beam_flux > 0.0) {
            dfuint += beam_particular_[nlyr_ - 1](iq) * beam_transmission_[nlyr_];
          }
          // Thermal contribution at bottom
          dfuint += kronecker_mode_0 * (thermal_particular_z0_[nlyr_ - 1](iq)
                   + thermal_particular_z1_[nlyr_ - 1](iq) * scaled_tau_cumulative_[nlyr_]);

          // RMU, CMU, CWT indices:
          // C: iq_C = iq+1 goes from nn..1
          //    RMU(iu, nn+1-iq_C) = surface_refl_user_(iu, nn_-iq)
          //    CMU(nn+1-iq_C) = quad_angle_(nn_-iq-1)
          //    CWT(nn+1-iq_C) = quad_weight_(nn_-iq-1)
          bnddfu += (1.0 + kronecker_mode_0) * surface_refl_user_(iu, nn_ - iq)
              * quad_angle_(nn_ - iq - 1) * quad_weight_(nn_ - iq - 1) * dfuint;
        }

        // Direct-beam reflection at bottom
        double bnddir = 0.0;
        if (direct_beam_flux > 0.0 || direct_beam_mu > 0.0) {
          bnddir = direct_beam_mu * direct_beam_flux / M_PI * surface_refl_user_(iu, 0) * beam_transmission_[nlyr_];
        }

        bndint = (bnddfu + bnddir + kronecker_mode_0 * surface_emis_user_(iu) * bplanck + isotropic_flux_bottom)
             * exp((scaled_tau_user_[lu] - scaled_tau_cumulative_[nlyr_]) / mu_user);
      }

      intensity_azim_mode_(iu, lu) = palint + plkint + bndint + genint;
    }
  }

  // Step 4: Accumulate azimuthal contributions into result.intensity / result.intensity_azimuth_avg
  if (azimuth_mode == 0) {
    // Initialize intensities and intensitiesAzimuthAvg with azimuth_mode=0 component
    for (int lu = 0; lu < ntau_; ++lu) {
      for (int iu = 1; iu <= numu_; ++iu) {  // iu 1-based for result accessor
        result.intensitiesAzimuthAvg(iu - 1, lu) = intensity_azim_mode_(iu - 1, lu);
        for (int j = 1; j <= nphi_; ++j) {
          result.intensities(iu - 1, lu, j - 1) = intensity_azim_mode_(iu - 1, lu);
        }
      }
    }
    return 0.0;
  } else {
    // Increment intensities by current azimuthal Fourier component (eq. SD(2))
    // and compute convergence metric (c_ratio of azimuthal contribution
    // to accumulated intensity) — mirrors C code lines 920-930.
    double max_azimuth_contrib = 0.0;
    for (int j = 1; j <= nphi_; ++j) {
      double phirad = (config_->phi_user[j - 1] - config_->bc.direct_beam_phi) * M_PI / 180.0;
      double cosphi = std::cos(static_cast<double>(azimuth_mode) * phirad);
      for (int lu = 0; lu < ntau_; ++lu) {
        for (int iu = 1; iu <= numu_; ++iu) {
          double azterm = intensity_azim_mode_(iu - 1, lu) * cosphi;
          result.intensities(iu - 1, lu, j - 1) += azterm;
          // Safe ratio |azterm| / |intensities|: mirrors c_ratio()
          double uu_abs = std::abs(result.intensities(iu - 1, lu, j - 1));
          double ratio;
          if (uu_abs == 0.0) {
            ratio = 1.0 + std::abs(azterm);  // denominator zero
          } else if (std::abs(azterm) == 0.0) {
            ratio = 0.0;
          } else {
            ratio = std::abs(azterm) / uu_abs;
          }
          max_azimuth_contrib = std::max(max_azimuth_contrib, ratio);
        }
      }
    }
    return max_azimuth_contrib;
  }
}

// ============================================================================
// Intensity Correction: Nakajima-Tanaka TMS/IMS Algorithm
// ============================================================================

double DisortSolver::xiFunc(double umu1, double umu2, double tau) const 
{
  // Eq. STWL (72): xi function for IMS correction.
  // Converts c_xi_func() from cdisort.c:5996-6031.
  double x1   = (umu2 - umu1) / (umu2 * umu1);
  double exp1  = std::exp(-tau / umu1);
  if (x1 != 0.0) {
    return ((tau * x1 - 1.0) * std::exp(-tau / umu2) + exp1) / (x1 * x1 * umu1 * umu2);
  } else {
    return tau * tau * exp1 / (2.0 * umu1 * umu2);
  }
}

double DisortSolver::singleScat(double dither, int layru, int num_layers,
                const std::vector<double>& phase_function,
                const std::vector<double>& omega,
                const std::vector<double>& tau,
                double mu_user, double direct_beam_mu,
                double tau_user, double direct_beam_flux) const 
{
  // Single-scattered intensity eqs. STWL (65b,d,e).
  // Converts c_single_scat() from cdisort.c:3936-4013.
  // phase_function[lc], omega[lc], tau[lc] are all 1-based (index 0..num_layers).
  constexpr double four_pi = 4.0 * M_PI;
  double ans  = 0.0;
  double exp0 = std::exp(-tau_user / direct_beam_mu);

  if (std::abs(mu_user + direct_beam_mu) <= dither) {
    // Downward when mu_user == -direct_beam_mu, eq. STWL (65e)
    for (int lyr = 1; lyr <= layru - 1; ++lyr) {
      ans += omega[lyr] * phase_function[lyr] * (tau[lyr] - tau[lyr - 1]);
    }
    ans = direct_beam_flux / (four_pi * direct_beam_mu) * exp0 *
        (ans + omega[layru] * phase_function[layru] * (tau_user - tau[layru - 1]));
    return ans;
  }

  if (mu_user > 0.0) {
    // Upward intensity eq. STWL (65b)
    for (int lyr = layru; lyr <= num_layers; ++lyr) {
      double exp1 = std::exp(-((tau[lyr] - tau_user) / mu_user + tau[lyr] / direct_beam_mu));
      ans  += omega[lyr] * phase_function[lyr] * (exp0 - exp1);
      exp0  = exp1;
    }
  } else {
    // Downward intensity eq. STWL (65d)
    for (int lyr = layru; lyr >= 1; --lyr) {
      double exp1 = std::exp(-((tau[lyr - 1] - tau_user) / mu_user + tau[lyr - 1] / direct_beam_mu));
      ans  += omega[lyr] * phase_function[lyr] * (exp0 - exp1);
      exp0  = exp1;
    }
  }
  ans *= direct_beam_flux / (four_pi * (1.0 + mu_user / direct_beam_mu));
  return ans;
}

double DisortSolver::secondaryScat(int iu, int lu, double ctheta, int layru) const 
{
  // IMS correction term eq. STWL (A7).
  // Converts c_secondary_scat() from cdisort.c:2668-2776.
  // Uses unscaled optical depths tau_cumulative_ and config_ arrays.
  constexpr double tiny = 1.0e-4;
  const DisortConfig& cfg = *config_;
  const int num_phase_func_moments  = cfg.num_phase_func_moments;
  const int num_streams  = nstr_;
  const double utau_lu = cfg.tau_user[lu - 1];
  const double umu_iu  = cfg.mu_user[iu - 1];

  // Vertically-averaged single-scatter surface_albedo (wbar) and fraction f (fbar),
  // eq. STWL (A.15): averaged from top to layer layru.
  double dtau  = utau_lu - tau_cumulative_[layru];
  double wbar  = cfg.single_scat_albedo[layru] * dtau;
  double fbar  = forward_scatter_frac_[layru] * wbar;
  double stau  = dtau;
  for (int lyr = 0; lyr < layru; ++lyr) {
    wbar += cfg.delta_tau[lyr] * cfg.single_scat_albedo[lyr];
    fbar += cfg.delta_tau[lyr] * cfg.single_scat_albedo[lyr] * forward_scatter_frac_[lyr];
    stau += cfg.delta_tau[lyr];
  }

  if (wbar <= tiny || fbar <= tiny || stau <= tiny || cfg.bc.direct_beam_flux <= tiny) {
    return 0.0;
  }

  fbar /= wbar;
  wbar /= stau;

  // Compute pspike = sum of (2*p'' - p''^2) weighted Legendre terms
  // where p'' = gbar*(2-gbar) for each moment.
  double pspike = 1.0;
  double gbar   = 1.0;
  double pl, plm1 = 1.0, plm2 = 0.0;

  // Moments k = 1 to num_streams-1 (using gbar = 1 for the residual)
  for (int k = 1; k <= num_streams - 1; ++k) {
    pl    = ((2 * k - 1) * ctheta * plm1 - (k - 1) * plm2) / k;
    plm2  = plm1;
    plm1  = pl;
    pspike += gbar * (2.0 - gbar) * (double)(2 * k + 1) * pl;
  }

  // Moments k = num_streams to num_phase_func_moments (using layer-weighted gbar)
  for (int k = num_streams; k <= num_phase_func_moments; ++k) {
    pl   = ((2 * k - 1) * ctheta * plm1 - (k - 1) * plm2) / k;
    plm2 = plm1;
    plm1 = pl;
    // Recompute dtau and gbar for this moment
    dtau = utau_lu - tau_cumulative_[layru];
    gbar = cfg.phaseFunctionMoments(k, layru) * cfg.single_scat_albedo[layru] * dtau;
    for (int lyr = 0; lyr < layru; ++lyr) {
      gbar += cfg.phaseFunctionMoments(k, lyr) * cfg.single_scat_albedo[lyr] * cfg.delta_tau[lyr];
    }
    double tmp = fbar * wbar * stau;
    if (tmp <= tiny) {
      gbar = 0.0;
    } else {
      gbar /= tmp;
    }
    pspike += gbar * (2.0 - gbar) * (double)(2 * k + 1) * pl;
  }

  double umu0p = cfg.bc.direct_beam_mu / (1.0 - fbar * wbar);

  // IMS correction term, eq. STWL (A.13)
  return cfg.bc.direct_beam_flux / (4.0 * M_PI) *
       (fbar * wbar) * (fbar * wbar) / (1.0 - fbar * wbar) *
       pspike * xiFunc(-umu_iu, umu0p, utau_lu);
}

void DisortSolver::applyIntensityCorrection(DisortResult& result) 
{
  // Nakajima-Tanaka intensity correction (TMS + IMS).
  // Converts c_intensity_correction() from cdisort.c:2502-2664.
  //
  // Must only be called when corint=true:
  //   deltam_ && config_->bc.direct_beam_flux > 0 && scat_yes_ && !config_->flags.comp_only_fluxes
  //
  // For each (iu, jp) pair, computes:
  //   TMS: intensities += singleScat(phast, single_scat_albedo, taucpr) - singleScat(phasm, oprim, taucpr)
  //   IMS: intensities -= secondaryScat(...)   (only for downward angles near aureole)
  const DisortConfig& cfg = *config_;
  const double dither   = 100.0 * std::numeric_limits<double>::epsilon();
  const double dtheta   = 10.0;  // aureole half-width in degrees
  const double deg      = std::acos(-1.0) / 180.0;  // π/180

  // Phase functions per layer — reuse pre-allocated members (avoid 6 allocs per solve)
  std::vector<double>& phasa = ic_phasa_;  // exact phase function
  std::vector<double>& phasm = ic_phasm_;  // delta-M-scaled phase function
  std::vector<double>& phast = ic_phast_;  // TMS phase function

  // omega arrays for singleScat (single_scat_albedo and oprim, 1-based)
  std::vector<double>& ssalb_vec = ic_ssalb_;
  std::vector<double>& oprim_vec = ic_oprim_;
  for (int lc = 1; lc <= ncut_; ++lc) {
    ssalb_vec[lc] = cfg.single_scat_albedo[lc - 1];
    oprim_vec[lc] = scaled_ssa_[lc - 1];
  }

  // Precompute phirad: phi_user angles minus direct_beam_phi, in radians
  std::vector<double>& phirad = ic_phirad_;
  for (int j = 1; j <= nphi_; ++j) {
    phirad[j] = (cfg.phi_user[j - 1] - cfg.bc.direct_beam_phi) * deg;
  }

  // -----------------------------------------------------------------------
  // Loop over user angles
  // -----------------------------------------------------------------------
  for (int iu = 1; iu <= numu_; ++iu) {
    double theta0 = 0.0, thetap = 0.0;
    if (cfg.mu_user[iu - 1] < 0.0) {
      theta0 = std::acos(-cfg.bc.direct_beam_mu) / deg;
      thetap = std::acos(cfg.mu_user[iu - 1])  / deg;
    }

    // Loop over azimuth angles
    for (int jp = 1; jp <= nphi_; ++jp) {
      // Cosine of scattering angle, eq. STWL (4)
      double ctheta = -cfg.bc.direct_beam_mu * cfg.mu_user[iu - 1] +
              std::sqrt((1.0 - cfg.bc.direct_beam_mu * cfg.bc.direct_beam_mu) *
                    (1.0 - cfg.mu_user[iu - 1] * cfg.mu_user[iu - 1])) *
              std::cos(phirad[jp]);

      // Initialize phase functions
      for (int lc = 1; lc <= ncut_; ++lc) {
        phasa[lc] = 1.0;
        phasm[lc] = 1.0;
      }

      // Legendre recurrence to build exact (phasa) and delta-M (phasm) phase functions
      double pl, plm1 = 1.0, plm2 = 0.0;
      for (int k = 1; k <= cfg.num_phase_func_moments; ++k) {
        pl   = ((2 * k - 1) * ctheta * plm1 - (k - 1) * plm2) / k;
        plm2 = plm1;
        plm1 = pl;

        // Exact phase function (all num_phase_func_moments moments)
        for (int lc = 1; lc <= ncut_; ++lc) {
          phasa[lc] += (double)(2 * k + 1) * pl * cfg.phaseFunctionMoments(k, lc - 1);
        }

        // Delta-M-scaled phase function (moments 1..num_streams-1 only)
        if (k <= nstr_ - 1) {
          for (int lc = 1; lc <= ncut_; ++lc) {
            double fl = forward_scatter_frac_[lc - 1];
            phasm[lc] += (double)(2 * k + 1) * pl *
                   (cfg.phaseFunctionMoments(k, lc - 1) - fl) / (1.0 - fl);
          }
        }
      }

      // TMS phase function: phast = phasa / (1 - flyr*single_scat_albedo), eq. STWL (68)
      for (int lc = 1; lc <= ncut_; ++lc) {
        phast[lc] = phasa[lc] / (1.0 - forward_scatter_frac_[lc - 1] * cfg.single_scat_albedo[lc - 1]);
      }

      // ---------------------------------------------------------------
      // TMS correction: replace delta-M single-scatter with exact
      // ---------------------------------------------------------------
      for (int lu = 1; lu <= ntau_; ++lu) {
        if (!lyrcut_ || layer_of_user_level_[lu - 1] < ncut_) {
          double ussndm = singleScat(dither, layer_of_user_level_[lu - 1] + 1, ncut_,
                         phast, ssalb_vec, scaled_tau_cumulative_,
                         cfg.mu_user[iu - 1], cfg.bc.direct_beam_mu,
                         scaled_tau_user_[lu - 1], cfg.bc.direct_beam_flux);
          double ussp   = singleScat(dither, layer_of_user_level_[lu - 1] + 1, ncut_,
                         phasm, oprim_vec, scaled_tau_cumulative_,
                         cfg.mu_user[iu - 1], cfg.bc.direct_beam_mu,
                         scaled_tau_user_[lu - 1], cfg.bc.direct_beam_flux);
          result.intensities(iu - 1, lu - 1, jp - 1) += ussndm - ussp;
        }
      }

      // ---------------------------------------------------------------
      // IMS correction: secondary scattering near aureole
      // Only for downward angles (mu_user < 0) within dtheta of the beam
      // ---------------------------------------------------------------
      if (cfg.mu_user[iu - 1] < 0.0 && std::abs(theta0 - thetap) <= dtheta) {
        int ltau = 1;
        if (cfg.tau_user[0] <= dither) {
          ltau = 2;
        }
        for (int lu = ltau; lu <= ntau_; ++lu) {
          if (!lyrcut_ || layer_of_user_level_[lu - 1] < ncut_) {
            double duims = secondaryScat(iu, lu, ctheta, layer_of_user_level_[lu - 1]);
            result.intensities(iu - 1, lu - 1, jp - 1) -= duims;
          }
        }
      }
    }  // end loop over azimuth angles
  }  // end loop over user angles
}

// ============================================================================
// New Intensity Correction (Buras-Emde algorithm)
// Converts c_new_intensity_correction(), prep_double_scat_integr(),
// c_new_secondary_scat(), calc_phase_squared() from cdisort.c:2846-3472.
// ============================================================================

void DisortSolver::applyNewIntensityCorrection(DisortResult& result) 
{
  // Buras-Emde intensity correction (TMS + IMS using tabulated phase function).
  // Converts c_new_intensity_correction() from cdisort.c:2846-3095.
  // Must only be called when corint=true and intensity_corr_nakajima=false.
  const DisortConfig& cfg = *config_;
  const double dither = 100.0 * std::numeric_limits<double>::epsilon();
  const double dtheta = 10.0;   // degrees defining aureole region
  const double tiny   = 1.0e-4;
  const int    nf     = 100;    // integration grid points
  const int    num_phase_func_angles = cfg.num_phase_func_angles;
  const double deg    = M_PI / 180.0;

  // 1-based flat-array accessors for phas2 [num_phase_func_angles × ntau_]
  auto PHAS2 = [&](int it, int lu, const std::vector<double>& p) -> double {
    return p[(it-1) + (lu-1)*num_phase_func_angles];
  };
  auto PHAS2w = [&](int it, int lu, std::vector<double>& p) -> double& {
    return p[(it-1) + (lu-1)*num_phase_func_angles];
  };

  // Determine if secondary scattering is needed
  // (any downward user angle within dtheta of beam direction)
  bool need_secondary_scattering = false;
  for (int iu = 1; iu <= numu_; ++iu) {
    if (cfg.mu_user[iu - 1] < 0.0) {
      double theta0  = std::acos(-cfg.bc.direct_beam_mu) / deg;
      double thetap  = std::acos(cfg.mu_user[iu - 1])  / deg;
      if (std::fabs(theta0 - thetap) <= dtheta) {
        need_secondary_scattering = true;
        break;
      }
    }
  }

  // Arrays for secondary scattering (only allocated if needed)
  std::vector<double> mu_eq, norm_phas;
  std::vector<int>    neg_phas;
  std::vector<double> phas2(num_phase_func_angles * ntau_, 0.0);

  if (need_secondary_scattering) {
    mu_eq.resize(nf * ntau_, 0.0);
    norm_phas.resize(ntau_, 0.0);
    neg_phas.resize(nf * ntau_, 0);

    // phasr: delta-M-truncated (not divided by 1-f) phase function per layer
    // PHASR(lc) = sum_{k=0}^{num_streams-1} (2k+1)*P_k(ctheta)*(pmom_k - f)
    //           = (1-f) + sum_{k=1}^{num_streams-1} (2k+1)*P_k*(pmom_k - f)
    std::vector<double> phasr(nlyr_ + 1, 0.0);  // 1-based lc

    for (int it = 1; it <= num_phase_func_angles; ++it) {
      double ctheta_it = cfg.mu_phase_function[it - 1];

      for (int lc = 1; lc <= nlyr_; ++lc)
        phasr[lc] = 1.0 - forward_scatter_frac_[lc - 1];

      double plm1 = 1.0, plm2 = 0.0;
      for (int k = 1; k <= nstr_ - 1; ++k) {
        double pl = ((2*k - 1) * ctheta_it * plm1 - (k - 1) * plm2) / k;
        plm2 = plm1;
        plm1 = pl;
        for (int lc = 1; lc <= nlyr_; ++lc)
          phasr[lc] += (2*k + 1) * pl * (cfg.phaseFunctionMoments(k, lc - 1) - forward_scatter_frac_[lc - 1]);
      }

      // phas2(it,lu) = sum over layers of (dsphase - phasr) * single_scat_albedo * delta_tau
      for (int lu = 1; lu <= ntau_; ++lu) {
        PHAS2w(it, lu, phas2) = 0.0;
        int llay = layer_of_user_level_[lu - 1];
        for (int lyr = 1; lyr <= llay; ++lyr) {
          // lyr is 1-based (1..llay); llay is 0-based, so 1-based lyr covers 0-based layers 0..llay-1
          PHAS2w(it, lu, phas2) +=
            (cfg.phase_function[lyr - 1][it - 1] - phasr[lyr]) *
            cfg.single_scat_albedo[lyr - 1] * cfg.delta_tau[lyr - 1];
        }
        int lyr = llay;  // 0-based layer index
        PHAS2w(it, lu, phas2) +=
          (cfg.phase_function[lyr][it - 1] - phasr[lyr + 1]) *
          cfg.single_scat_albedo[lyr] * (cfg.tau_user[lu - 1] - tau_cumulative_[lyr]);
      }
    }  // end for it

    // Normalize phas2: divide by fbar = sum of flyr*single_scat_albedo*delta_tau
    for (int lu = 1; lu <= ntau_; ++lu) {
      int llay = layer_of_user_level_[lu - 1];
      double fbar = forward_scatter_frac_[llay] * cfg.single_scat_albedo[llay] *
              (cfg.tau_user[lu - 1] - tau_cumulative_[llay]);
      for (int lyr = 0; lyr < llay; ++lyr)
        fbar += cfg.single_scat_albedo[lyr] * cfg.delta_tau[lyr] * forward_scatter_frac_[lyr];

      if (fbar <= tiny || cfg.bc.direct_beam_flux <= tiny) {
        for (int it = 1; it <= num_phase_func_angles; ++it)
          PHAS2w(it, lu, phas2) = 0.0;
      } else {
        fbar = 1.0 / fbar;
        for (int it = 1; it <= num_phase_func_angles; ++it)
          PHAS2w(it, lu, phas2) *= fbar;
      }

      // Further normalize phas2 to integrate to 2.0
      double f_phas2 = 0.0;
      for (int it = 2; it <= num_phase_func_angles; ++it) {
        f_phas2 += (cfg.mu_phase_function[it - 1] - cfg.mu_phase_function[it - 2]) * 0.5 *
               (PHAS2(it, lu, phas2) + PHAS2(it-1, lu, phas2));
      }
      if (f_phas2 != 0.0) {
        double norm = 2.0 / f_phas2;
        for (int it = 1; it <= num_phase_func_angles; ++it)
          PHAS2w(it, lu, phas2) *= norm;
      }
    }  // end for lu

    prepDoubleScatIntegr(num_phase_func_angles, nf, cfg.mu_phase_function, phas2,
               mu_eq, neg_phas, norm_phas);
  }  // end if need_secondary_scattering

  // Pre-compute phi_user angles in radians
  std::vector<double> phirad(nphi_ + 1, 0.0);
  for (int jp = 1; jp <= nphi_; ++jp)
    phirad[jp] = (cfg.phi_user[jp - 1] - cfg.bc.direct_beam_phi) * deg;

  // Phase function vectors (1-based, size nlyr_+1)
  std::vector<double> phasm(nlyr_ + 1, 0.0);
  std::vector<double> phast(nlyr_ + 1, 0.0);

  // single_scat_albedo and oprim vectors for singleScat calls (1-based, size nlyr_+1)
  std::vector<double> ssalb_vec(nlyr_ + 1, 0.0);
  std::vector<double> oprim_vec(nlyr_ + 1, 0.0);
  for (int lc = 1; lc <= ncut_; ++lc) {
    ssalb_vec[lc] = cfg.single_scat_albedo[lc - 1];
    oprim_vec[lc] = scaled_ssa_[lc - 1];
  }

  // -----------------------------------------------------------------------
  // Loop over user angles and azimuth angles
  // -----------------------------------------------------------------------
  for (int iu = 1; iu <= numu_; ++iu) {
    double theta0 = 0.0, thetap = 0.0;
    if (cfg.mu_user[iu - 1] < 0.0) {
      theta0 = std::acos(-cfg.bc.direct_beam_mu) / deg;
      thetap = std::acos(cfg.mu_user[iu - 1])  / deg;
    }

    for (int jp = 1; jp <= nphi_; ++jp) {
      // Cosine of scattering angle, eq. STWL(4)
      double ctheta = -cfg.bc.direct_beam_mu * cfg.mu_user[iu - 1] +
              std::sqrt((1.0 - cfg.bc.direct_beam_mu*cfg.bc.direct_beam_mu) *
                    (1.0 - cfg.mu_user[iu - 1]*cfg.mu_user[iu - 1])) *
              std::cos(phirad[jp]);

      // Initialize delta-M-scaled phase function
      for (int lc = 1; lc <= ncut_; ++lc)
        phasm[lc] = 1.0;

      // Interpolate exact phase from tabulated cfg.phase_function[] and compute phast
      // locate() returns 0-based index; +1 gives 1-based it
      int it = locate(cfg.mu_phase_function, ctheta) + 1;
      for (int lc = 1; lc <= ncut_; ++lc) {
        double phasa = cfg.phase_function[lc - 1][it - 1] +
          (ctheta - cfg.mu_phase_function[it - 1]) /
          (cfg.mu_phase_function[it] - cfg.mu_phase_function[it - 1]) *
          (cfg.phase_function[lc - 1][it] - cfg.phase_function[lc - 1][it - 1]);
        phast[lc] = phasa / (1.0 - forward_scatter_frac_[lc - 1] * cfg.single_scat_albedo[lc - 1]);
      }

      // Build delta-M-scaled phase function (phasm) via Legendre recurrence
      double pl, plm1 = 1.0, plm2 = 0.0;
      for (int k = 1; k <= nstr_ - 1; ++k) {
        pl   = ((2*k - 1) * ctheta * plm1 - (k - 1) * plm2) / k;
        plm2 = plm1;
        plm1 = pl;
        for (int lc = 1; lc <= ncut_; ++lc)
          phasm[lc] += (2.0*k + 1) * pl *
                 (cfg.phaseFunctionMoments(k, lc - 1) - forward_scatter_frac_[lc - 1]) / (1.0 - forward_scatter_frac_[lc - 1]);
      }

      // -----------------------------------------------------------
      // TMS correction: replace delta-M single-scatter with exact
      // -----------------------------------------------------------
      for (int lu = 1; lu <= ntau_; ++lu) {
        if (!lyrcut_ || layer_of_user_level_[lu - 1] < ncut_) {
          double ussndm = singleScat(dither, layer_of_user_level_[lu - 1] + 1, ncut_,
                         phast, ssalb_vec, scaled_tau_cumulative_,
                         cfg.mu_user[iu - 1], cfg.bc.direct_beam_mu,
                         scaled_tau_user_[lu - 1], cfg.bc.direct_beam_flux);
          double ussp   = singleScat(dither, layer_of_user_level_[lu - 1] + 1, ncut_,
                         phasm, oprim_vec, scaled_tau_cumulative_,
                         cfg.mu_user[iu - 1], cfg.bc.direct_beam_mu,
                         scaled_tau_user_[lu - 1], cfg.bc.direct_beam_flux);
          result.intensities(iu - 1, lu - 1, jp - 1) += ussndm - ussp;
        }
      }

      // -----------------------------------------------------------
      // IMS correction: secondary scattering for angles in aureole
      // Only for downward angles (mu_user < 0) within dtheta of beam
      // -----------------------------------------------------------
      if (cfg.mu_user[iu - 1] < 0.0 && std::fabs(theta0 - thetap) <= dtheta) {
        int ltau = 1;
        if (cfg.tau_user[0] <= dither)
          ltau = 2;
        for (int lu = ltau; lu <= ntau_; ++lu) {
          if (!lyrcut_ || layer_of_user_level_[lu - 1] < ncut_) {
            double duims = newSecondaryScat(iu, lu, it, ctheta, layer_of_user_level_[lu - 1],
                             nf, num_phase_func_angles,
                             phas2, mu_eq, neg_phas,
                             norm_phas[lu-1]);
            result.intensities(iu - 1, lu - 1, jp - 1) -= duims;
          }
        }
      }
    }  // end jp loop
  }  // end iu loop
}

// ============================================================================
// Analytic BRDF Intensity Correction (Fortran V3 INTCOR_BEAM_REFLEC)
// ============================================================================

void DisortSolver::applyBrdfIntensityCorrection(DisortResult& result) 
{
  /*
   * Port of INTCOR_BEAM_REFLEC from Fortran DISORT V3 (DISORT.f:2461-2577).
   *
   * Corrects reflected beam intensity at user angles by comparing the exact
   * (analytic) BRDF evaluation with the Fourier-series reconstruction.
   * The Fourier expansion has limited angular resolution (nstr modes), so
   * the difference (analytic - approximate) is added as a correction.
   *
   * Correction per (iu, lu, j):
   *   USS = mu0 * fbeam * (rho_analytic - rho_approx)
   *       * exp(-tau_bottom / mu0)
   *       * exp((tau_user - tau_bottom) / mu_user)
   */
  const DisortConfig& cfg = *config_;
  const double fbeam = cfg.bc.direct_beam_flux;
  const double umu0  = cfg.bc.direct_beam_mu;
  const double deg   = M_PI / 180.0;

  for (int iu = 0; iu < numu_; ++iu) {
    double mu_user = cfg.mu_user[iu];
    if (mu_user <= 0.0) continue;  // Only upward angles see surface reflection

    for (int j = 0; j < nphi_; ++j) {
      double dphi_deg = cfg.phi_user[j] - cfg.bc.direct_beam_phi;
      double dphi_rad = dphi_deg * deg;

      // Exact BRDF at this user angle/azimuth geometry
      double rho_analytic = bidirReflectivity(mu_user, umu0, dphi_rad);
      if (rho_analytic < 0.0) continue;

      // Fourier-reconstructed approximate BRDF
      // rho_approx = sum_{m=0}^{nstr-1} brdf_fourier_user_beam_(iu, m) * cos(m * dphi_rad)
      double rho_approx = 0.0;
      for (int m = 0; m < nstr_; ++m) {
        rho_approx += brdf_fourier_user_beam_(iu, m)
                    * std::cos(static_cast<double>(m) * dphi_rad);
      }

      double drho = rho_analytic - rho_approx;
      if (std::abs(drho) < 1e-15) continue;

      // Beam transmission to bottom boundary
      double beam_to_bottom = std::exp(-scaled_tau_cumulative_[ncut_] / umu0);

      // Apply correction to each output level
      for (int lu = 0; lu < ntau_; ++lu) {
        if (lyrcut_ && layer_of_user_level_[lu] >= ncut_) continue;

        double uss = umu0 * fbeam * drho * beam_to_bottom
                   * std::exp((scaled_tau_user_[lu] - scaled_tau_cumulative_[ncut_]) / mu_user);

        result.intensities(iu, lu, j) += uss;
      }
    }
  }
}

// ----------------------------------------------------------------------------

void DisortSolver::prepDoubleScatIntegr(int num_phase_func_angles, int nf,
                    const std::vector<double>& mu_phase_function,
                    const std::vector<double>& phas2,
                    std::vector<double>& mu_eq,
                    std::vector<int>&    neg_phas,
                    std::vector<double>& norm_phas) const 
{
  // Converts prep_double_scat_integr() from cdisort.c:3133-3210.
  // Builds equidistant-in-|phas2| integration grid (mu_eq), sign array
  // (neg_phas), and normalization (norm_phas).

  auto MUP      = [&](int it) -> double { return mu_phase_function[it-1]; };
  auto PHAS2    = [&](int it, int lu) -> double { return phas2[(it-1) + (lu-1)*num_phase_func_angles]; };
  auto MU_EQ    = [&](int i, int lu) -> double& { return mu_eq[(i-1) + (lu-1)*nf]; };
  auto NEG_PHAS = [&](int i, int lu) -> int&    { return neg_phas[(i-1) + (lu-1)*nf]; };

  std::vector<double> f_phas2_abs(num_phase_func_angles, 0.0);  // 1-based it = 1..num_phase_func_angles
  auto F_PHAS2_ABS = [&](int it) -> double& { return f_phas2_abs[it-1]; };

  for (int lu = 1; lu <= ntau_; ++lu) {
    // Compute cumulative integral of |phas2| over angle grid
    F_PHAS2_ABS(1) = 0.0;
    for (int it = 2; it <= num_phase_func_angles; ++it) {
      F_PHAS2_ABS(it) = F_PHAS2_ABS(it-1) +
        (MUP(it) - MUP(it-1)) * 0.5 *
        (std::fabs(PHAS2(it, lu)) + std::fabs(PHAS2(it-1, lu)));
    }

    // Build mu_eq grid equidistant in F_PHAS2_ABS
    double f_phas2 = 0.0;
    double df = F_PHAS2_ABS(num_phase_func_angles) / (nf - 1);
    MU_EQ(1, lu) = -1.0;
    NEG_PHAS(1, lu) = (PHAS2(1, lu) > 0.0) ? 0 : 1;

    int it = 1;
    for (int i = 2; i <= nf - 1; ++i) {
      f_phas2 += df;
      while (F_PHAS2_ABS(it + 1) < f_phas2)
        ++it;

      MU_EQ(i, lu) = MUP(it) +
        (f_phas2 - F_PHAS2_ABS(it)) /
        (F_PHAS2_ABS(it + 1) - F_PHAS2_ABS(it)) *
        (MUP(it + 1) - MUP(it));

      if (PHAS2(it, lu) > 0.0 && PHAS2(it+1, lu) > 0.0) {
        NEG_PHAS(i, lu) = 0;
      } else if (PHAS2(it, lu) < 0.0 && PHAS2(it+1, lu) < 0.0) {
        NEG_PHAS(i, lu) = 1;
      } else {
        double val = PHAS2(it, lu) +
          (f_phas2 - F_PHAS2_ABS(it)) /
          (F_PHAS2_ABS(it + 1) - F_PHAS2_ABS(it)) *
          (PHAS2(it+1, lu) - PHAS2(it, lu));
        NEG_PHAS(i, lu) = (val > 0.0) ? 0 : 1;
      }
    }

    MU_EQ(nf, lu) = 1.0;
    NEG_PHAS(nf, lu) = (PHAS2(num_phase_func_angles, lu) > 0.0) ? 0 : 1;
    norm_phas[lu-1] = F_PHAS2_ABS(num_phase_func_angles) / ((nf - 1) * M_PI);
  }
}

// ----------------------------------------------------------------------------

double DisortSolver::newSecondaryScat(int iu, int lu, int it, double ctheta,
                    int layru, int nf, int num_phase_func_angles,
                    const std::vector<double>& phas2,
                    const std::vector<double>& mu_eq,
                    const std::vector<int>&    neg_phas,
                    double norm_phas) const 
{
  // Converts c_new_secondary_scat() from cdisort.c:3255-3318.
  // Computes IMS correction: direct_beam_flux/(4π) * (f̄·ω̄)²/(1-f̄·ω̄) * pspike * ξ(...)
  const DisortConfig& cfg = *config_;
  const double tiny = 1.0e-4;

  auto MUP   = [&](int i) -> double { return cfg.mu_phase_function[i - 1]; };  // i is 1-based
  auto PHAS2 = [&](int i, int lv) -> double { return phas2[(i-1) + (lv-1)*num_phase_func_angles]; };

  // Vertically averaged single_scat_albedo (wbar) and separated fraction (fbar), eq. STWL (A.15)
  // layru is 0-based (0..nlyr_-1); lu and iu are 1-based from caller
  double dtau = cfg.tau_user[lu - 1] - tau_cumulative_[layru];
  double wbar = cfg.single_scat_albedo[layru] * dtau;
  double fbar = forward_scatter_frac_[layru] * wbar;
  double stau = dtau;
  for (int lyr = 0; lyr < layru; ++lyr) {
    wbar += cfg.delta_tau[lyr] * cfg.single_scat_albedo[lyr];
    fbar += cfg.delta_tau[lyr] * cfg.single_scat_albedo[lyr] * forward_scatter_frac_[lyr];
    stau += cfg.delta_tau[lyr];
  }

  if (wbar <= tiny || fbar <= tiny || stau <= tiny || cfg.bc.direct_beam_flux <= tiny)
    return 0.0;

  fbar /= wbar;
  wbar /= stau;

  // pspike1 = P" at the scattering angle (residual phase function interpolated)
  double pspike1 = PHAS2(it, lu) +
    (ctheta - MUP(it)) / (MUP(it+1) - MUP(it)) *
    (PHAS2(it+1, lu) - PHAS2(it, lu));

  // pspike2 = P"*P" convolution integral
  double pspike2 = calcPhaseSquared(num_phase_func_angles, lu, ctheta, nf,
                    cfg.mu_phase_function, phas2, mu_eq, neg_phas,
                    norm_phas);

  double pspike = 2.0 * pspike1 - pspike2;
  double umu0p  = cfg.bc.direct_beam_mu / (1.0 - fbar * wbar);

  // IMS correction term, eq. STWL (A.13)
  return cfg.bc.direct_beam_flux / (4.0 * M_PI) *
       (fbar * wbar) * (fbar * wbar) / (1.0 - fbar * wbar) *
       pspike * xiFunc(-cfg.mu_user[iu - 1], umu0p, cfg.tau_user[lu - 1]);
}

// ----------------------------------------------------------------------------

double DisortSolver::calcPhaseSquared(int num_phase_func_angles, int lu, double ctheta, int nf,
                    const std::vector<double>& mu_phase_function,
                    const std::vector<double>& phas2,
                    const std::vector<double>& mu_eq,
                    const std::vector<int>&    neg_phas,
                    double norm_phas) const 
{
  // Converts calc_phase_squared() from cdisort.c:3354-3470.
  // Computes the azimuthal integral of phas2(mu_1) * phas2(theta_12) over mu_1,
  // using the equidistant mu_eq grid. Result = p"*p" for IMS correction.

  auto MUP      = [&](int i)         -> double { return mu_phase_function[i-1]; };
  auto PHAS2    = [&](int i, int lv) -> double { return phas2[(i-1) + (lv-1)*num_phase_func_angles]; };
  auto MU_EQ    = [&](int i, int lv) -> double { return mu_eq[(i-1) + (lv-1)*nf]; };
  auto NEG_PHAS = [&](int i, int lv) -> int    { return neg_phas[(i-1) + (lv-1)*nf]; };

  double pspike2 = 0.0;
  double stheta  = std::sqrt(1.0 - ctheta * ctheta);

  for (int j = 1; j <= nf; ++j) {
    double phint;

    if (ctheta == 1.0 || MU_EQ(j, lu) == 1.0) {
      // Special case: second scattering angle independent of azimuth
      int it2 = locate(mu_phase_function, MU_EQ(j, lu) * ctheta) + 1;
      phint = M_PI * (PHAS2(it2, lu) +
        (MU_EQ(j, lu) * ctheta - MUP(it2)) /
        (MUP(it2 + 1) - MUP(it2)) *
        (PHAS2(it2 + 1, lu) - PHAS2(it2, lu)));
      if (ctheta == 1.0)
        phint /= 2.0;
    } else {
      phint = 0.0;
      double smueq = std::sqrt(1.0 - MU_EQ(j, lu) * MU_EQ(j, lu));
      double mumin  = ctheta * MU_EQ(j, lu) - stheta * smueq;
      double mumax  = ctheta * MU_EQ(j, lu) + stheta * smueq;

      bool cutting = false;
      if (MU_EQ(j, lu) < mumax) {
        mumax   = MU_EQ(j, lu);
        cutting = true;
      }

      if (mumin < mumax) {
        int imin = locate(mu_phase_function, mumin) + 1;
        int imax = locate(mu_phase_function, mumax) + 1;

        int k = imin;
        double D = (PHAS2(k+1, lu) - PHAS2(k, lu)) / (MUP(k+1) - MUP(k));
        double C = PHAS2(k, lu) - MUP(k) * D;
        phint += (D * ctheta * MU_EQ(j, lu) + C) * M_PI / 2.0;

        for (k = imin + 1; k <= imax; ++k) {
          double Dp = (PHAS2(k+1, lu) - PHAS2(k, lu)) / (MUP(k+1) - MUP(k));
          double Cp = PHAS2(k, lu) - MUP(k) * Dp;

          phint +=
            (Dp - D) * std::sqrt(1.0 - ctheta*ctheta
                   - MU_EQ(j,lu)*MU_EQ(j,lu)
                   + 2.0*ctheta*MU_EQ(j,lu)*MUP(k)
                   - MUP(k)*MUP(k))
            + ((Dp - D)*ctheta*MU_EQ(j,lu) + Cp - C) *
              std::asin((ctheta*MU_EQ(j,lu) - MUP(k)) /
                  (smueq * stheta));
          D = Dp;
          C = Cp;
        }

        if (cutting) {
          phint += -D * std::sqrt(1.0 - ctheta*ctheta
                  + 2.0*MU_EQ(j,lu)*MU_EQ(j,lu)*(ctheta - 1.0))
               - (D*ctheta*MU_EQ(j,lu) + C) *
                 std::asin((ctheta - 1.0)*MU_EQ(j,lu) /
                     (smueq * stheta));
        } else {
          phint += (D*ctheta*MU_EQ(j,lu) + C) * M_PI / 2.0;
        }
      }
    }

    if (j == 1 || j == nf)
      pspike2 += (NEG_PHAS(j, lu) ? -0.5 : +0.5) * phint;
    else
      pspike2 += (NEG_PHAS(j, lu) ? -1.0 : +1.0) * phint;
  }

  return pspike2 * norm_phas;
}

// ============================================================================
// SPECIAL_BC: Medium Albedo and Transmissivity
// ============================================================================

void DisortSolver::computeAlbTrans(DisortConfig& config, DisortResult& result) 
{
  /*
   * Compute medium surface_albedo and transmissivity for SPECIAL_BC mode.
   * Based on c_albtrans() from cdisort.c:7453-7632.
   *
   * Uses the reciprocity principle:
   *   surface_albedo(theta)  = upward azim-avg intensity at TOP under isotropic top illumination
   *   trans(theta)   = upward azim-avg intensity at TOP under isotropic bottom illumination
   *
   * For single layer, transmissivity is obtained from the BOTTOM intensity under top illumination.
   */

  // Force special-case settings (mirrors c_albtrans() lines 7492-7497)
  ncut_   = nlyr_;
  lyrcut_ = false;
  config.bc.isotropic_flux_top    = 1.0;
  config.bc.isotropic_flux_bottom    = 0.0;
  config.flags.use_lambertian_surface = true;   // Force Lambertian (BDR=0 in base computation)

  const int numu_half = numu_ / 2;  // original user num_user_mu (numu_ already doubled)

  // Zero out BDR: ALBEDO=0 base case for reciprocity
  // (mirrors c_albtrans() line 7520: memset(bdr, 0, ...))
  surface_refl_quad_.setZero();
  surface_emis_quad_.setZero();

  // Compute Legendre polynomials at user angles (azimuth_mode=0)
  // numu_ = 2*original, mu_user contains both negative (lower half) and positive (upper half)
  {
    std::vector<double> umu_vec(numu_);
    for (int iu = 0; iu < numu_; ++iu) {
      umu_vec[iu] = config.mu_user[iu];
    }
    legendrePolynomialsFlat(numu_, 0, numu_ - 1, nstr_ - 1, umu_vec.data(), legendre_user_.data());
  }

  // Pre-compute legendre_quad_ and interp_ylmc_cwt_ for azimuth_mode=0 before calling solveEigen/interpEigenvec.
  // (In the normal code path these are computed in the main azimuthal loop in solve().)
  legendrePolynomialsFlat(nstr_, 0, nstr_ - 1, nstr_ - 1, cmu_vec_.data(), legendre_quad_.data());
  for (int l = 0; l < nstr_; ++l)
    for (int jq = 0; jq < nstr_; ++jq)
      interp_ylmc_cwt_(l, jq) = legendre_quad_(l, jq) * quad_weight_(jq);

  // Solve eigenvalue problem and interpolate eigenvectors to user angles for each layer
  for (int lc = 0; lc < nlyr_; ++lc) {
    solveEigen(lc, 0);
    interpEigenvec(lc, 0);
  }

  // Build coefficient matrix (kronecker_mode_0=1 for azimuth_mode=0, lyrcut=false)
  setMatrix(1.0, false);

  // LU-decompose the banded matrix once; reused for TOP_ILLUM and BOT_ILLUM
  // (mirrors c_albtrans() c_sgbco() at line 7547)
  // bandFactor factors band_matrix_ in-place; band_pivot_ holds pivot indices.
  // bandSolve (called in solve1) uses the factored band_matrix_ and is safe to call twice.
  const int ncol = nstr_ * ncut_;
  const int ncd  = 3 * nn_ - 1;
  bandFactor(band_matrix_.data(), band_matrix_.rows(), ncol, ncd, ncd, band_pivot_.data());

  // Workspace for intensity results (numu_ × 2)
  // col 0: top output level (utaupr=0), col 1: bottom output level (utaupr=taucpr[num_layers])
  Eigen::MatrixXd u0u_top(numu_, 2);

  // === Illuminate from TOP ===
  // (mirrors c_albtrans() line 7556: c_solve1(..., TOP_ILLUM, ...))
  solve1(0 /* TOP_ILLUM */, ncol);
  albTransIntensity(u0u_top);

  // Extract beam-incidence albedos from reciprocity principle
  // albedo_medium(iu) = intensitiesAzimuthAvg(iu + numu_half, 1) for iu = 1..numu_half
  // (upward intensity at TOP under top illumination = surface_albedo)
  for (int iu = 1; iu <= numu_half; ++iu) {
    result.albedo_medium[iu - 1] = u0u_top(iu + numu_half - 1, 0);
  }

  if (nlyr_ == 1) {
    // Single layer: transmissivity from TOP illumination (col 1 = bottom level)
    // transmissivity_medium(iu) = intensitiesAzimuthAvg(numu_half+1-iu, 2) + exp(-taucpr[num_layers]/UMU(iu+numu_half))
    // The intensitiesAzimuthAvg index numu_half+1-iu (1-based) → 0-based: numu_half-iu
    for (int iu = 1; iu <= numu_half; ++iu) {
      double mu = config.mu_user[iu + numu_half - 1];  // positive user angle UMU(iu+numu_half)
      result.transmissivity_medium[iu - 1] = u0u_top(numu_half - iu, 1)
                + std::exp(-scaled_tau_cumulative_[nlyr_] / mu);
    }
  } else {
    // Multi-layer: illuminate from BOTTOM to get transmissivity
    Eigen::MatrixXd u0u_bot(numu_, 2);
    solve1(1 /* BOT_ILLUM */, ncol);
    albTransIntensity(u0u_bot);

    // transmissivity_medium(iu) = intensitiesAzimuthAvg(iu + numu_half, 1) + exp(-taucpr[num_layers]/UMU(iu+numu_half))
    // (upward intensity at TOP under bottom illumination = transmissivity)
    for (int iu = 1; iu <= numu_half; ++iu) {
      double mu = config.mu_user[iu + numu_half - 1];
      result.transmissivity_medium[iu - 1] = u0u_bot(iu + numu_half - 1, 0)
                + std::exp(-scaled_tau_cumulative_[nlyr_] / mu);
    }
  }

  // === Spherical surface_albedo correction (if bc.surface_albedo > 0) ===
  // (mirrors c_albtrans() lines 7594-7613)
  if (config.bc.surface_albedo > 0.0) {
    double sflup, sfldn;
    // integration_constants_ currently holds: TOP_ILLUM solution for num_layers=1, BOT_ILLUM for num_layers>1
    albTransSpherical(sflup, sfldn);

    double sphalb, sphtrn;
    if (nlyr_ == 1) {
      // sflup = upward at top under TOP illumination = sphalb
      // sfldn = downward at bottom under TOP illumination = sphtrn
      sphalb = sflup;
      sphtrn = sfldn;
    } else {
      // integration_constants_ holds BOT_ILLUM: sflup = upward at top under BOT illumination = sphtrn
      // sfldn = downward at bottom under BOT illumination = sphalb
      sphtrn = sflup;
      sphalb = sfldn;
    }

    double correction = config.bc.surface_albedo / (1.0 - config.bc.surface_albedo * sphalb);
    for (int iu = 1; iu <= numu_half; ++iu) {
      result.albedo_medium[iu - 1] += correction * sphtrn * result.transmissivity_medium[iu - 1];
      result.transmissivity_medium[iu - 1] += correction * sphalb * result.transmissivity_medium[iu - 1];
    }

    result.spherical_albedo = sphalb;
    result.spherical_transmissivity = sphtrn;
  }

  // Restore numu_ and mu_user to positive-only values
  // (mirrors c_albtrans() lines 7617-7620)
  numu_ = numu_half;
  config.num_user_mu = numu_half;
  for (int iu = 1; iu <= numu_half; ++iu) {
    config.mu_user[iu] = config.mu_user[iu + numu_half];
  }
}

void DisortSolver::solve1(int ihom, int ncol) 
{
  /*
   * Solve for integration constants under isotropic illumination.
   * Based on c_solve1() from cdisort.c:7826-7871.
   *
   * Simpler than solve0(): only top-boundary (isotropic_flux_top) or bottom-boundary (isotropic_flux_top)
   * forcing; no beam or thermal particular solutions.
   *
   * ihom = 0 (TOP_ILLUM): B[0..nn-1] = isotropic_flux_top, B[ncol-nn..ncol-1] = 0
   * ihom = 1 (BOT_ILLUM): B[0..nn-1] = 0,     B[ncol-nn..ncol-1] = isotropic_flux_top
   *
   * Requires band_matrix_ to already be LU-factored by bandFactor (and band_pivot_ set).
   */
  const double isotropic_flux_top = config_->bc.isotropic_flux_top;

  // Use solve0_b_ scratch space (already sized nstr_*nlyr_ >= ncol)
  Eigen::VectorXd& b = solve0_b_;
  b.head(ncol).setZero();

  if (ihom == 0 /* TOP_ILLUM */) {
    for (int i = 0; i < nn_; ++i) {
      b(i) = isotropic_flux_top;
    }
  } else /* BOT_ILLUM */ {
    for (int i = 0; i < nn_; ++i) {
      b(ncol - nn_ + i) = isotropic_flux_top;
    }
  }

  // Solve using already-factored banded matrix; b.data() overwritten with solution
  bandSolve(band_matrix_.data(), band_matrix_.rows(), ncol, nn_ * 3 - 1, nn_ * 3 - 1,
      band_pivot_.data(), b.data());

  // Reorder solution into integration_constants_ — same pattern as solve0()
  // (b now holds the solution X after sgbsl)
  integration_constants_.setZero();
  for (int lc = 0; lc < ncut_; ++lc) {
    int ipnt = (lc + 1) * nstr_ - nn_ - 1;

    for (int iq = 0; iq < nn_; ++iq) {
      int ll_idx = lc * nstr_ + (nn_ - iq - 1);
      int b_idx  = ipnt - iq;
      if (b_idx >= 0 && b_idx < ncol && ll_idx < (int)integration_constants_.size()) {
        integration_constants_(ll_idx) = b(b_idx);
      }
    }
    for (int iq = 0; iq < nn_; ++iq) {
      int ll_idx = lc * nstr_ + (nn_ + iq);
      int b_idx  = ipnt + iq + 1;
      if (b_idx >= 0 && b_idx < ncol && ll_idx < (int)integration_constants_.size()) {
        integration_constants_(ll_idx) = b(b_idx);
      }
    }
  }
}

void DisortSolver::albTransIntensity(Eigen::MatrixXd& u0u_out) 
{
  /*
   * Compute azimuthally-averaged intensity at top (lu=0) and bottom (lu=1).
   * Based on c_albtrans_intensity() from cdisort.c:7678-7757.
   *
   * Does NOT modify eigenvec_user_ in-place; computes gu*ll on the fly so it can be
   * called twice with different integration_constants_ (for TOP_ILLUM and BOT_ILLUM).
   *
   * Output levels:
   *   lu=0: tau=0 (top),   iumin=numu_/2, iumax=numu_-1, sgn=+1
   *   lu=1: tau=taucpr[num_layers] (bottom), iumin=0, iumax=numu_/2-1, sgn=-1
   */
  const double utaupr_top = 0.0;
  const double utaupr_bot = scaled_tau_cumulative_[nlyr_];

  for (int lu = 0; lu < 2; ++lu) {
    int    iumin, iumax;
    double sgn, utaupr_lu;
    if (lu == 0) {
      iumin     = numu_ / 2;     // upward angles at top: upper half of doubled mu_user
      iumax     = numu_ - 1;
      sgn       = 1.0;
      utaupr_lu = utaupr_top;
    } else {
      iumin     = 0;             // downward angles at bottom: lower half
      iumax     = numu_ / 2 - 1;
      sgn       = -1.0;
      utaupr_lu = utaupr_bot;
    }

    for (int iu = iumin; iu <= iumax; ++iu) {
      double mu   = config_->mu_user[iu];  
      double palint = 0.0;

      for (int lc = 0; lc < nlyr_; ++lc) {
        double dtau = scaled_tau_cumulative_[lc + 1] - scaled_tau_cumulative_[lc];
        double exp1 = std::exp((utaupr_lu - scaled_tau_cumulative_[lc    ]) / mu);
        double exp2 = std::exp((utaupr_lu - scaled_tau_cumulative_[lc + 1]) / mu);

        // Negative eigenvalues: eigenvalues_[lc](iq) for iq=0..nn_-1
        Eigen::VectorXd wk_neg(nn_);
        for (int iq = 0; iq < nn_; ++iq) {
          wk_neg(iq) = std::exp(eigenvalues_[lc](iq) * dtau);
          double denom = 1.0 + mu * eigenvalues_[lc](iq);
          double expn;
          if (std::abs(denom) < 0.0001) {
            expn = dtau / mu * exp2;  // L'Hospital limit
          } else {
            expn = (exp1 * wk_neg(iq) - exp2) * sgn / denom;
          }
          // GU(iu,iq,lc)*LL(iq,lc): eigenvec_user_[lc](iu, iq) * integration_constants_(iq + lc*nstr_)
          palint += eigenvec_user_[lc](iu, iq) * integration_constants_(iq + lc * nstr_) * expn;
        }

        // Positive eigenvalues: eigenvalues_[lc](iq) for iq=nn_..nstr_-1
        for (int iq = nn_; iq < nstr_; ++iq) {
          double denom = 1.0 + mu * eigenvalues_[lc](iq);
          double expn;
          if (std::abs(denom) < 0.0001) {
            expn = -dtau / mu * exp1;  // L'Hospital limit
          } else {
            // WK(num_streams+1-iq) in C = wk_neg(nstr_-1-iq) in 0-based C++
            expn = (exp1 - exp2 * wk_neg(nstr_ - 1 - iq)) * sgn / denom;
          }
          palint += eigenvec_user_[lc](iu, iq) * integration_constants_(iq + lc * nstr_) * expn;
        }
      }
      u0u_out(iu, lu) = palint;
    }
  }
}

void DisortSolver::albTransSpherical(double& sflup, double& sfldn) 
{
  /*
   * Compute spherical surface_albedo and transmissivity from current integration_constants_ and eigenvectors_.
   * Based on c_albtrans_spherical() from cdisort.c:7907-7951.
   *
   * sflup: upward flux at TOP boundary (× 2)
   * sfldn: downward flux at BOTTOM boundary (× 2)
   */

  // Upward flux at TOP: integrate upward streams (iq=nn_..nstr_-1) at tau=0
  // C: for iq=nn+1..num_streams, zint = sum_jq GC(iq,jq,1)*LL(jq,1)*WK(jq) + GC(iq,jq,1)*LL(jq,1)
  //    sflup += CWT(iq-nn)*CMU(iq-nn)*zint
  // Where WK(jq) = exp(KK(jq,1)*TAUCPR(1)) for jq=1..nn (negative kk)
  sflup = 0.0;
  {
    double dtau0 = scaled_tau_cumulative_[1] - scaled_tau_cumulative_[0];  // = scaled_tau_cumulative_[1] since scaled_tau_cumulative_[0]=0
    for (int iq = nn_; iq < nstr_; ++iq) {
      // C iq = nn+1..num_streams, 0-based: iq = nn_..nstr_-1
      // C CWT(iq-nn) = quad_weight_ at index iq-nn_
      // eigenvectors_[0](iq, jq) ↔ C GC(iq+1, jq+1, 1)
      double zint = 0.0;
      for (int jq = 0; jq < nn_; ++jq) {
        // Negative kk: WK(jq) = exp(KK(jq,1)*dtau0) = exp(eigenvalues_[0](jq)*dtau0)
        double wk_jq = std::exp(eigenvalues_[0](jq) * dtau0);
        zint += eigenvectors_[0](iq, jq) * integration_constants_(jq + 0 * nstr_) * wk_jq;
      }
      for (int jq = nn_; jq < nstr_; ++jq) {
        // Positive kk: no exponential (they appear at tau=0)
        zint += eigenvectors_[0](iq, jq) * integration_constants_(jq + 0 * nstr_);
      }
      sflup += quad_weight_(iq - nn_) * quad_angle_(iq - nn_) * zint;
    }
  }

  // Downward flux at BOTTOM: integrate downward streams (iq=0..nn_-1) at tau=taucpr[num_layers]
  // C: for iq=1..nn, zint = sum_jq GC(iq,jq,num_layers)*LL(jq,num_layers) + *exp(...)
  //    sfldn += CWT(nn+1-iq)*CMU(nn+1-iq)*zint
  sfldn = 0.0;
  {
    int last_lc = nlyr_ - 1;
    double dtau_last = scaled_tau_cumulative_[nlyr_] - scaled_tau_cumulative_[nlyr_ - 1];
    for (int iq = 0; iq < nn_; ++iq) {
      // C iq = 1..nn, 0-based: iq = 0..nn_-1
      // C CWT(nn+1-iq) = quad_weight_ at index nn_-1-iq (0-based)
      double zint = 0.0;
      for (int jq = 0; jq < nn_; ++jq) {
        // Negative kk at bottom: no exponential (appear at bottom naturally)
        zint += eigenvectors_[last_lc](iq, jq) * integration_constants_(jq + last_lc * nstr_);
      }
      for (int jq = nn_; jq < nstr_; ++jq) {
        // Positive kk at bottom: WK = exp(-eigenvalues_[last_lc](jq)*dtau_last)
        // C: exp(-KK(jq,num_layers)*(TAUCPR(num_layers)-TAUCPR(num_layers-1)))
        // eigenvalues_[last_lc](jq) > 0, so exp is < 1
        double wk_jq = std::exp(-eigenvalues_[last_lc](jq) * dtau_last);
        zint += eigenvectors_[last_lc](iq, jq) * integration_constants_(jq + last_lc * nstr_) * wk_jq;
      }
      sfldn += quad_weight_(nn_ - 1 - iq) * quad_angle_(nn_ - 1 - iq) * zint;
    }
  }

  sflup *= 2.0;
  sfldn *= 2.0;
}

} // namespace disortpp
