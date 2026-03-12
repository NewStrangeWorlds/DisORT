#include "FluxSolver.hpp"
#include "BandSolver.hpp"
#include "IndexConversion.hpp"
#include "Planck.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace disortpp {

// Forward declaration of file-static Chapman function helper
static double chapmanImpl(int lc, double tau_frac,
              const std::vector<double>& level_altitudes,
              const std::vector<double>& dtau_lyr,
              int num_layers, double r, double direct_beam_mu);

// ============================================================================
// Main Solve Method
// ============================================================================

template<int NStr>
FluxResult DisortFluxSolver<NStr>::solve(const DisortFluxConfig& config_in)
{
  // If index_from_bottom, work on a reversed local copy
  const bool reverse_index = config_in.index_from_bottom;
  DisortFluxConfig local_config;
  if (reverse_index) {
    local_config = config_in;
    reverseInputArrays(local_config);
  }
  const DisortFluxConfig& config = reverse_index ? local_config : config_in;

  config_ = &config;

  config.validate();

  // Validate NStr matches config
  if (config.num_streams != NStr) {
    throw std::invalid_argument(
      "DisortFluxSolver<" + std::to_string(NStr) + ">::solve: config.num_streams="
      + std::to_string(config.num_streams) + " does not match template NStr");
  }

  nlyr_ = config.num_layers;

  // Determine if scattering is present
  scat_yes_ = false;
  for (int lc = 0; lc < nlyr_; ++lc) {
    if (config.single_scat_albedo[lc] > 0.0) {
      scat_yes_ = true;
      break;
    }
  }

  // Determine if delta-M transformation is needed
  deltam_ = false;
  for (int lc = 0; lc < nlyr_; ++lc) {
    if (config.phaseFunctionMoments(NStr, lc) != 0.0) {
      deltam_ = true;
      break;
    }
  }

  // Determine if Delta-M+ should be used (Lin et al. 2018)
  deltam_plus_ = config.use_delta_m_plus;
  if (deltam_plus_) {
    // Global fallback: check ALL layers (Fortran DISORT lines 482-498)
    for (int lc = 0; lc < nlyr_; ++lc) {
      double pm  = config.phaseFunctionMoments(NStr, lc);
      double pm1 = config.phaseFunctionMoments(NStr + 1, lc);
      if (pm < 1e-4 || pm1 < 0.7 * pm) {
        deltam_plus_ = false;
        break;
      }
    }
    if (deltam_plus_) {
      deltam_ = false;  // Delta-M+ replaces standard Delta-M
    }
    // If fallback: deltam_ retains its value from above
  }

  // Allocate working arrays if dimensions changed
  if (nlyr_ != alloc_nlyr_) {
    allocateWorkingArrays();
    result_.allocate(nlyr_);
    alloc_nlyr_ = nlyr_;
  } else {
    // Zero result in-place
    std::fill(result_.flux_direct_beam.begin(), result_.flux_direct_beam.end(), 0.0);
    std::fill(result_.flux_down.begin(), result_.flux_down.end(), 0.0);
    std::fill(result_.flux_up.begin(), result_.flux_up.end(), 0.0);
    std::fill(result_.flux_tau_divergence.begin(), result_.flux_tau_divergence.end(), 0.0);
    std::fill(result_.mean_intensity.begin(), result_.mean_intensity.end(), 0.0);
    std::fill(result_.mean_intensity_down.begin(), result_.mean_intensity_down.end(), 0.0);
    std::fill(result_.mean_intensity_up.begin(), result_.mean_intensity_up.end(), 0.0);
    std::fill(result_.mean_intensity_direct_beam.begin(), result_.mean_intensity_direct_beam.end(), 0.0);
  }

  // Setup: delta-M scaling, quadrature, beam attenuation
  disortSet(config, deltam_);

  // Precompute Planck functions for thermal emission boundaries
  planck_bottom_ = 0.0;
  planck_bottom_deriv_ = 0.0;
  planck_top_ = 0.0;
  if (config.use_thermal_emission) {
    planck_top_ = planckFunction2(config.wavenumber_low, config.wavenumber_high,
                    config.temperature_top) * config.emissivity_top;
    planck_bottom_ = planckFunction2(config.wavenumber_low, config.wavenumber_high,
                      config.temperature_bottom);

    // Compute dB/dτ at bottom for diffusion lower BC
    if (config.use_diffusion_lower_bc) {
      double B_layer = planckFunction2(config.wavenumber_low, config.wavenumber_high,
                        config.temperature[nlyr_ - 1]);
      double dtau_last = scaled_dtau_[nlyr_ - 1];
      if (dtau_last > 0.0) {
        planck_bottom_deriv_ = (planck_bottom_ - B_layer) / dtau_last;
      }
    }
  }

  // Only azimuth mode m=0 is needed for fluxes
  // Compute Legendre polynomials at computational angles
  legendrePolynomialsFlat(NStr, 0, NStr - 1, NStr - 1, cmu_vec_.data(), legendre_quad_.data());

  // Eigenvalue problem for each layer
  for (int lc = 0; lc < nlyr_; ++lc) {
    solveEigen(lc);
  }

  // Particular solutions
  if (config.direct_beam_flux > 0.0) {
    if (config.use_spherical_beam) {
      computeBeamSourceSpherical();
    } else {
      computeBeamSource();
    }
  }

  if (config.use_thermal_emission) {
    computeIsotropicSource();
  }

  // Compute Lambertian surface reflectivity (trivial for azimuth_mode=0)
  surface_refl_quad_.setZero();
  surface_emis_quad_.setZero();
  for (int iq = 0; iq < NN; ++iq) {
    surface_emis_quad_(iq) = 1.0 - config.surface_albedo;
    for (int jq = 0; jq <= NN; ++jq) {
      surface_refl_quad_(iq, jq) = config.surface_albedo;
    }
  }

  // Boundary value problem
  setMatrix();
  solve0();

  // Compute fluxes at all layer boundaries
  computeFluxes(result_);

  if (reverse_index) {
    reverseOutputArrays(result_);
  }

  return result_;
}

// ============================================================================
// Memory Allocation
// ============================================================================

template<int NStr>
void DisortFluxSolver<NStr>::allocateWorkingArrays() 
{
  // Delta-M scaled arrays
  scaled_dtau_.resize(nlyr_);
  scaled_tau_cumulative_.resize(nlyr_ + 1);
  scaled_ssa_.resize(nlyr_);
  forward_scatter_frac_.resize(nlyr_);

  scaled_phase_coeffs_.resize(NStr, nlyr_);

  // Beam attenuation
  chapman_.resize(nlyr_);
  chapman_tau_.resize(2 * nlyr_ + 2);
  beam_transmission_.resize(nlyr_ + 1);

  // Per-layer eigensystem
  eigenvalues_.resize(nlyr_);
  eigenvectors_.resize(nlyr_);
  phase_matrix_.resize(nlyr_);
  for (int lc = 0; lc < nlyr_; ++lc) {
    eigenvalues_[lc].setZero();
    eigenvectors_[lc].setZero();
    phase_matrix_[lc].setZero();
  }

  // Particular solutions
  beam_particular_.resize(nlyr_);
  beam_particular_slope_.resize(nlyr_);
  beam_particular_atten_.assign(nlyr_, 0.0);
  thermal_z0_.resize(nlyr_);
  thermal_z1_.resize(nlyr_);
  for (int lc = 0; lc < nlyr_; ++lc) {
    beam_particular_[lc].setZero();
    beam_particular_slope_[lc].setZero();
    thermal_z0_[lc].setZero();
    thermal_z1_[lc].setZero();
  }

  // Thermal expansion coefficients
  planck_intercept_.assign(nlyr_, 0.0);
  planck_slope_.assign(nlyr_, 0.0);

  // Band matrix
  const int mi9m2 = 9 * NN - 2;
  const int nnlyri = NStr * nlyr_;
  band_matrix_.resize(mi9m2, nnlyri);
  band_pivot_.resize(nnlyri);
  integration_constants_.resize(nnlyri);
  solve0_b_.resize(nnlyri);

  // Surface reflectivity
  surface_refl_quad_.resize(NN, NN + 1);
  surface_emis_quad_.setZero();

}

// ============================================================================
// Gaussian Quadrature
// ============================================================================

template<int NStr>
void DisortFluxSolver<NStr>::gaussianQuadrature(int n) 
{
  const double eps = 3.0e-14;
  const int max_iter = 10;

  // Compute for positive angles (upward)
  for (int i = 1; i <= n; ++i) {
    double z = std::cos(M_PI * (i - 0.25) / (n + 0.5));
    double z1, pp;
    int iter = 0;
    do {
      double p1 = 1.0, p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        double p3 = p2; p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      z1 = z;
      z = z1 - p1 / pp;
      if (++iter > max_iter) {
        throw std::runtime_error("gaussianQuadrature: Failed to converge");
      }
    } while (std::abs(z - z1) > eps);

    quad_angle_(i - 1) = 0.5 * (1.0 + z);
    quad_weight_(i - 1) = 1.0 / ((1.0 - z * z) * pp * pp);
  }

  // Downward (negative) angles
  for (int i = 0; i < n; ++i) {
    quad_angle_(n + i) = -quad_angle_(i);
    quad_weight_(n + i) = quad_weight_(i);
  }
}

// ============================================================================
// Chapman Function
// ============================================================================

template<int NStr>
double DisortFluxSolver<NStr>::chapmanFunction(int lc, double tau_frac) 
{
  return chapmanImpl(lc, tau_frac,
             config_->level_altitudes, scaled_dtau_,
             nlyr_, config_->bottom_radius, config_->direct_beam_mu);
}

// ============================================================================
// Setup and Delta-M Transformation
// ============================================================================

template<int NStr>
void DisortFluxSolver<NStr>::disortSet(const DisortFluxConfig& config, bool deltam)
{
  const double abscut = 10.0;

  // Cumulative optical depths (unscaled)
  std::vector<double> tauc(nlyr_ + 1);
  tauc[0] = 0.0;
  for (int lc = 0; lc < nlyr_; ++lc) {
    tauc[lc + 1] = tauc[lc] + config.delta_tau[lc];
  }

  // Apply delta-M scaling
  scaled_tau_cumulative_[0] = 0.0;
  double abstau = 0.0;
  ncut_ = nlyr_;

  for (int lc = 0; lc < nlyr_; ++lc) {

    if (abstau < abscut) {
      ncut_ = lc + 1;
    }
    abstau += (1.0 - config.single_scat_albedo[lc]) * config.delta_tau[lc];

    double f;
    if (!deltam && !deltam_plus_) {
      scaled_ssa_[lc] = config.single_scat_albedo[lc];
      scaled_dtau_[lc] = config.delta_tau[lc];
      scaled_tau_cumulative_[lc + 1] = tauc[lc + 1];
      for (int k = 0; k < NStr; ++k) {
        scaled_phase_coeffs_(k, lc) = (2.0 * k + 1.0) * scaled_ssa_[lc] * config.phaseFunctionMoments(k, lc);
      }
      f = 0.0;
    } else if (!deltam_plus_ ||
               config.phaseFunctionMoments(NStr - 1, lc) == config.phaseFunctionMoments(NStr, lc)) {
      // Standard Delta-M (also per-layer fallback when consecutive moments M-1 and M are equal)
      f = config.phaseFunctionMoments(NStr, lc);
      scaled_ssa_[lc] = config.single_scat_albedo[lc] * (1.0 - f)
               / (1.0 - f * config.single_scat_albedo[lc]);
      scaled_dtau_[lc] = (1.0 - f * config.single_scat_albedo[lc]) * config.delta_tau[lc];
      scaled_tau_cumulative_[lc + 1] = scaled_tau_cumulative_[lc] + scaled_dtau_[lc];
      for (int k = 0; k < NStr; ++k) {
        scaled_phase_coeffs_(k, lc) = (2.0 * k + 1.0) * scaled_ssa_[lc]
                       * (config.phaseFunctionMoments(k, lc) - f) / (1.0 - f);
      }
    } else {
      // Delta-M+ transformation (Lin et al. 2018)
      double pm  = config.phaseFunctionMoments(NStr, lc);
      double pm1 = config.phaseFunctionMoments(NStr + 1, lc);

      if (pm == pm1) {
        // Degenerate case: fall back to standard Delta-M for this layer
        f = pm;
        scaled_ssa_[lc] = config.single_scat_albedo[lc] * (1.0 - f)
                 / (1.0 - f * config.single_scat_albedo[lc]);
        scaled_dtau_[lc] = (1.0 - f * config.single_scat_albedo[lc]) * config.delta_tau[lc];
        scaled_tau_cumulative_[lc + 1] = scaled_tau_cumulative_[lc] + scaled_dtau_[lc];
        for (int k = 0; k < NStr; ++k) {
          scaled_phase_coeffs_(k, lc) = (2.0 * k + 1.0) * scaled_ssa_[lc]
                         * (config.phaseFunctionMoments(k, lc) - f) / (1.0 - f);
        }
      } else {
        double M = static_cast<double>(NStr);
        double sigma_sq = ((M + 1.0) * (M + 1.0) - M * M) /
                          (std::log(pm * pm) - std::log(pm1 * pm1));
        double c = std::exp(M * M / (2.0 * sigma_sq));
        f = c * pm;

        scaled_ssa_[lc] = config.single_scat_albedo[lc] * (1.0 - f)
                 / (1.0 - f * config.single_scat_albedo[lc]);
        scaled_dtau_[lc] = (1.0 - f * config.single_scat_albedo[lc]) * config.delta_tau[lc];
        scaled_tau_cumulative_[lc + 1] = scaled_tau_cumulative_[lc] + scaled_dtau_[lc];

        for (int k = 0; k < NStr; ++k) {
          double kd = static_cast<double>(k);
          scaled_phase_coeffs_(k, lc) = (2.0 * k + 1.0) * scaled_ssa_[lc] *
                 (config.phaseFunctionMoments(k, lc) - f * std::exp(-kd * kd / (2.0 * sigma_sq))) / (1.0 - f);
        }
      }
    }
    forward_scatter_frac_[lc] = f;
  }

  // Beam attenuation
  if (config.direct_beam_flux > 0.0) {
    chapman_tau_[0] = 0.0;
    beam_transmission_[0] = 1.0;

    if (config.use_spherical_beam) {
      if (config.direct_beam_mu < 0.0) {
        beam_transmission_[0] = std::exp(-chapmanImpl(1, 0.0,
                             config.level_altitudes, config.delta_tau,
                             nlyr_, config.bottom_radius,
                             config.direct_beam_mu));
      }
      for (int lc = 0; lc < ncut_; ++lc) {
        const double taup_mid = scaled_tau_cumulative_[lc] + scaled_dtau_[lc] / 2.0;
        const double chtau_mid = chapmanImpl(lc + 1, 0.5,
                           config.level_altitudes, scaled_dtau_,
                           nlyr_, config.bottom_radius,
                           config.direct_beam_mu);
        chapman_[lc] = taup_mid / chtau_mid;
        chapman_tau_[lc + 1] = chapmanImpl(lc + 1, 0.0,
                          config.level_altitudes, scaled_dtau_,
                          nlyr_, config.bottom_radius,
                          config.direct_beam_mu);
        beam_transmission_[lc + 1] = std::exp(-chapman_tau_[lc + 1]);
      }
    } else {
      for (int lc = 0; lc < ncut_; ++lc) {
        chapman_[lc] = config.direct_beam_mu;
        beam_transmission_[lc + 1] = std::exp(-scaled_tau_cumulative_[lc + 1] / config.direct_beam_mu);
      }
    }
  } else {
    for (int lc = 0; lc < ncut_; ++lc) {
      beam_transmission_[lc + 1] = 0.0;
    }
  }

  // Layer cutoff
  lyrcut_ = false;
  if (abstau >= abscut && !config.use_thermal_emission && nlyr_ > 1) {
    lyrcut_ = true;
  }
  if (!lyrcut_) {
    ncut_ = nlyr_;
  }

  // Compute Gaussian quadrature
  gaussianQuadrature(NN);

  // Precompute similarity transform vectors for eigenvalue symmetrization:
  // D = diag(sqrt(cwt_i * mu_i)), D^{-1} = diag(1 / sqrt(cwt_i * mu_i))
  for (int iq = 0; iq < NN; ++iq) {
    similarity_d_(iq) = std::sqrt(quad_weight_(iq) * quad_angle_(iq));
    similarity_d_inv_(iq) = 1.0 / similarity_d_(iq);
  }

  // Populate cmu_vec_ for legendrePolynomialsFlat
  for (int i = 0; i < NStr; ++i) cmu_vec_[i] = quad_angle_(i);

  // Note: beam angle matching a quadrature angle is handled by per-layer
  // eigenvalue-based dithering in computeBeamSource() (Fortran DISORT V3 approach).
}

// ============================================================================
// Eigenvalue Problem
// ============================================================================

template<int NStr>
void DisortFluxSolver<NStr>::solveEigen(int lc) 
{
  // Build coefficient matrix CC for this layer using DGEMM-like operations
  // cc_top(iq, jq) = 0.5 * cwt(jq) * sum_l { gl(l,lc) * ylmc(l,iq) * ylmc(l,jq) }

  // Step 1: Scale Legendre polynomials by phase coefficients
  // gl_ylmc(l, iq) = scaled_phase_coeffs_(l, lc) * legendre_quad_(l, iq) for l=0..NStr-1
  Eigen::Matrix<double, NStr, NN> cc_gl_ylmc;
  for (int l = 0; l < NStr; ++l) {
    for (int iq = 0; iq < NN; ++iq) {
      cc_gl_ylmc(l, iq) = scaled_phase_coeffs_(l, lc) * legendre_quad_(l, iq);
    }
  }

  // Step 2: cc_top(iq, jq) = sum_l { cc_gl_ylmc(l,iq) * legendre_quad_(l,jq) }
  Eigen::Matrix<double, NN, NStr> cc_top;
  cc_top.noalias() = cc_gl_ylmc.transpose() * legendre_quad_.template topRows<NStr>();
  cc_top *= 0.5;
  cc_top.array().rowwise() *= quad_weight_.transpose().array();

  // Step 3: Fill phase_matrix_[lc] using symmetry
  phase_matrix_[lc].topRows(NN) = cc_top;
  phase_matrix_[lc].template bottomLeftCorner<NN, NN>() = cc_top.rightCols(NN);
  phase_matrix_[lc].template bottomRightCorner<NN, NN>() = cc_top.leftCols(NN);

  // Step 4: Build alpha_minus_beta_ and alpha_plus_beta_
  for (int iq = 0; iq < NN; ++iq) {
    const double inv_cmu = 1.0 / quad_angle_(iq);
    alpha_minus_beta_.row(iq).noalias() = inv_cmu * (cc_top.row(iq).template head<NN>()
                           - cc_top.row(iq).template tail<NN>());
    alpha_plus_beta_.row(iq).noalias() = inv_cmu * (cc_top.row(iq).template head<NN>()
                          + cc_top.row(iq).template tail<NN>());
    alpha_minus_beta_(iq, iq) -= inv_cmu;
    alpha_plus_beta_(iq, iq) -= inv_cmu;
  }

  // Symmetrize the eigenvalue problem via similarity transform.
  // The matrices alpha_plus_beta_ and alpha_minus_beta_ have the form:
  //   M(i,j) = (cwt_j / mu_i) * f_sym(i,j) - delta(i,j) / mu_i
  // where f_sym is symmetric in the angle arguments.
  // The similarity D = diag(sqrt(cwt_i * mu_i)) symmetrizes both matrices:
  //   A_s = D * alpha_plus_beta_ * D^{-1}  (symmetric)
  //   B_s = D * alpha_minus_beta_ * D^{-1}  (symmetric, negative definite for omega<1)
  //
  // We want eigenvalues/vectors of M = (alpha+beta)(alpha-beta).
  // D*M*D^{-1} = A_s * B_s has the same eigenvalues.
  // Since -B_s is SPD (for omega < 1), factor -B_s = L*L^T (Cholesky).
  // Then eigenvalues of A_s*B_s = eigenvalues of S = -L^T * A_s * L (symmetric).
  // Eigenvectors of M: u = D^{-1} * L^{-T} * w  (where S*w = lambda*w).
  HalfMat A_s, neg_B_s;
  for (int i = 0; i < NN; ++i) {
    for (int j = 0; j < NN; ++j) {
      A_s(i, j) = similarity_d_(i) * alpha_plus_beta_(i, j) * similarity_d_inv_(j);
      neg_B_s(i, j) = -similarity_d_(i) * alpha_minus_beta_(i, j) * similarity_d_inv_(j);
    }
  }

  HalfVec eigenvalues_tmp;
  HalfMat eigenvectors_tmp;

  // For non-conservative scattering (omega < 1), -B_s is SPD — use fast Cholesky path.
  // For conservative scattering (omega = 1), -B_s is only PSD — fall back to general solver.
  bool use_spd_path = (scaled_ssa_[lc] < 1.0 - 1e-12);
  if (use_spd_path) {
    cholesky_.compute(neg_B_s);
    use_spd_path = (cholesky_.info() == Eigen::Success);
  }
  if (use_spd_path) {
    // Fast symmetric path: -B_s is SPD
    // Form S = -L^T * A_s * L = L^T * (-A_s) * L  (symmetric, eigenvalues = k^2 > 0)
    const auto& L = cholesky_.matrixL();
    eigen_product_.noalias() = -A_s * L;
    HalfMat S;
    S.noalias() = L.transpose() * eigen_product_;

    sym_eig_solver_.compute(S);
    eigenvalues_tmp = sym_eig_solver_.eigenvalues();

    // Recover eigenvectors of M: u = D^{-1} * L^{-T} * w
    // L^T is upper triangular; solve L^T * v = w for each eigenvector w
    HalfMat Lt = HalfMat(L.transpose());
    for (int i = 0; i < NN; ++i) {
      HalfVec v = Lt.template triangularView<Eigen::Upper>().solve(
                    sym_eig_solver_.eigenvectors().col(i));
      for (int j = 0; j < NN; ++j) {
        eigenvectors_tmp(j, i) = similarity_d_inv_(j) * v(j);
      }
    }
  } else {
    // Fallback for conservative scattering (omega=1) or other degenerate cases:
    // -B_s is not SPD, use general eigensolver on the product matrix
    eigen_product_.noalias() = alpha_plus_beta_ * alpha_minus_beta_;
    Eigen::EigenSolver<HalfMat> eig(eigen_product_, true);
    for (int i = 0; i < NN; ++i) {
      eigenvalues_tmp(i) = eig.eigenvalues()(i).real();
      for (int j = 0; j < NN; ++j) {
        eigenvectors_tmp(j, i) = eig.eigenvectors()(j, i).real();
      }
    }
  }

  // Extract eigenvalues: k = sqrt(|lambda|)
  HalfVec reduced_eigenvalues;
  for (int iq = 0; iq < NN; ++iq) {
    reduced_eigenvalues(iq) = std::sqrt(std::abs(eigenvalues_tmp(iq)));
    eigenvalues_[lc](iq + NN) = reduced_eigenvalues(iq);
    eigenvalues_[lc](NN - 1 - iq) = -reduced_eigenvalues(iq);
  }

  // Compute eigenvectors (G+) + (G-) and store in alpha_plus_beta_ temporarily
  // alpha_plus_beta_ = (alpha_minus_beta_ * eigenvectors_tmp) / diag(reduced_eigenvalues)
  alpha_plus_beta_.noalias() = alpha_minus_beta_ * eigenvectors_tmp;
  alpha_plus_beta_.array().rowwise() /= reduced_eigenvalues.transpose().array();

  // Recover eigenvectors G+, G- from sum and difference
  for (int jq = 0; jq < NN; ++jq) {
    for (int iq = 0; iq < NN; ++iq) {
      double gpplgm = alpha_plus_beta_(iq, jq);
      double gpmigm = eigenvectors_tmp(iq, jq);

      // Store in eigenvectors_ with symmetry reordering
      eigenvectors_[lc](NN + iq, NN + jq) = 0.5 * (gpplgm + gpmigm);
      eigenvectors_[lc](NN - iq - 1, NN + jq) = 0.5 * (gpplgm - gpmigm);

      gpplgm *= -1.0;
      eigenvectors_[lc](NN + iq, NN - jq - 1) = 0.5 * (gpplgm + gpmigm);
      eigenvectors_[lc](NN - iq - 1, NN - jq - 1) = 0.5 * (gpplgm - gpmigm);
    }
  }
}

// ============================================================================
// Beam Source (Plane-Parallel)
// ============================================================================

template<int NStr>
void DisortFluxSolver<NStr>::computeBeamSource() 
{
  const double direct_beam_mu = config_->direct_beam_mu;
  const double direct_beam_flux = config_->direct_beam_flux;

  // Legendre polynomials at beam angle (azimuth_mode=0, kronecker_mode_0=1)
  umu0_vec_[0] = -direct_beam_mu;
  legendrePolynomialsFlat(1, 0, 0, NStr - 1, umu0_vec_.data(), legendre_beam_.data());

  constexpr double dither_threshold = 1000.0 * std::numeric_limits<double>::epsilon();

  for (int lc = 0; lc < nlyr_; ++lc) {
    // Per-layer eigenvalue-based beam dithering (Fortran DISORT V3 approach):
    // If |1 - mu0^2 * kk^2| is near zero for any eigenvalue, the beam source
    // matrix becomes near-singular. Dither mu0 by 0.1% for this layer only.
    double beam_mu = direct_beam_mu;
    const double mu0sq = direct_beam_mu * direct_beam_mu;
    for (int iq = 0; iq < NN; ++iq) {
      double kk = eigenvalues_[lc](iq + NN);  // positive eigenvalue
      if (std::abs(1.0 - mu0sq * kk * kk) < dither_threshold) {
        beam_mu = 0.999 * direct_beam_mu;
        break;
      }
    }

    // Build A = I + diag(mu/mu0) - C
    beam_A_.noalias() = -phase_matrix_[lc];
    beam_A_.diagonal().array() += 1.0 + quad_angle_.array() / beam_mu;

    // Build RHS: direct_beam_flux * sum_k[GL*YLMC*YLM0] / (4*pi)
    for (int iq = 0; iq < NStr; ++iq) {
      double sum = 0.0;
      for (int k = 0; k < NStr; ++k) {
        sum += scaled_phase_coeffs_(k, lc) * legendre_quad_(k, iq) * legendre_beam_(k);
      }
      beam_b_(iq) = direct_beam_flux * sum / (4.0 * M_PI);
    }

    // Solve using fixed-size LU
    Eigen::PartialPivLU<FullMat> lu(beam_A_);
    FullVec x = lu.solve(beam_b_);

    // Store with angular reordering
    for (int iq = 0; iq < NN; ++iq) {
      beam_particular_[lc](NN + iq) = x(iq);
      beam_particular_[lc](NN - iq - 1) = x(iq + NN);
    }
  }
}

// ============================================================================
// Beam Source (Pseudo-Spherical)
// ============================================================================

template<int NStr>
void DisortFluxSolver<NStr>::computeBeamSourceSpherical() 
{
  const double direct_beam_flux = config_->direct_beam_flux;
  const double big = std::sqrt(std::numeric_limits<double>::max()) / 1.0e10;

  // Legendre polynomials at beam angle
  umu0_vec_[0] = -config_->direct_beam_mu;
  legendrePolynomialsFlat(1, 0, 0, NStr - 1, umu0_vec_.data(), legendre_beam_.data());

  for (int lc = 0; lc < nlyr_; ++lc) {
    // Compute ZJ: source function coefficients
    for (int iq = 0; iq < NStr; ++iq) {
      double sum = 0.0;
      for (int k = 0; k < NStr; ++k) {
        sum += scaled_phase_coeffs_(k, lc) * legendre_quad_(k, iq) * legendre_beam_(k);
      }
      spher_zj_(iq) = direct_beam_flux * sum / (4.0 * M_PI);
    }

    // Beam transmission at top and bottom of layer
    const double q0a = std::exp(-chapman_tau_[lc]);
    const double q2a = std::exp(-chapman_tau_[lc + 1]);

    // XBA = 1/CH(lc), with dithering
    double xba = 1.0 / chapman_[lc];
    if (std::abs(xba) > big && scaled_tau_cumulative_[lc + 1] > 1.0) xba = 0.0;
    if (std::abs(xba * scaled_tau_cumulative_[lc + 1]) > std::log(big)) xba = 0.0;
    if (std::abs(xba) > 1.0e-5) {
      for (int iq = 0; iq < NN; ++iq) {
        if (std::abs((std::abs(xba) - 1.0 / quad_angle_(iq)) / xba) < 0.05) {
          xba *= 1.001;
          break;
        }
      }
    }
    beam_particular_atten_[lc] = xba;

    const double deltat = scaled_tau_cumulative_[lc + 1] - scaled_tau_cumulative_[lc];

    // Linear-in-tau expansion coefficients
    FullVec xb0, xb1;
    for (int iq = 0; iq < NStr; ++iq) {
      const double q0 = q0a * spher_zj_(iq);
      const double q2 = q2a * spher_zj_(iq);
      xb1(iq) = (q2 * std::exp(xba * scaled_tau_cumulative_[lc + 1])
            - q0 * std::exp(xba * scaled_tau_cumulative_[lc])) / deltat;
      xb0(iq) = q0 * std::exp(xba * scaled_tau_cumulative_[lc])
            - xb1(iq) * scaled_tau_cumulative_[lc];
    }

    // Build system [I + xba*mu*I - C]
    spher_A_.noalias() = -phase_matrix_[lc];
    for (int iq = 0; iq < NStr; ++iq) {
      spher_A_(iq, iq) += 1.0 + xba * quad_angle_(iq);
    }
    Eigen::PartialPivLU<FullMat> lu(spher_A_);

    // Dither if near-singular
    const double rcond_approx = 1.0 / spher_A_.norm();
    if (rcond_approx < 1.0e-4) {
      if (xba == 0.0) xba = 5.0e-9;
      xba *= 1.00000005;
      beam_particular_atten_[lc] = xba;
      spher_A_.noalias() = -phase_matrix_[lc];
      for (int iq = 0; iq < NStr; ++iq) {
        spher_A_(iq, iq) += 1.0 + xba * quad_angle_(iq);
      }
      lu = Eigen::PartialPivLU<FullMat>(spher_A_);
    }

    // Solve for ZBS1
    FullVec zbs1 = lu.solve(xb1);

    // ZBS0(iq) = XB0(iq) + CMU(iq)*ZBS1(iq)
    FullVec zbs0;
    for (int iq = 0; iq < NStr; ++iq) {
      zbs0(iq) = xb0(iq) + quad_angle_(iq) * zbs1(iq);
    }
    zbs0 = lu.solve(zbs0);

    // Store with reordering
    for (int iq = 0; iq < NN; ++iq) {
      beam_particular_[lc](NN + iq) = zbs0(iq);
      beam_particular_[lc](NN - 1 - iq) = zbs0(iq + NN);
      beam_particular_slope_[lc](NN + iq) = zbs1(iq);
      beam_particular_slope_[lc](NN - 1 - iq) = zbs1(iq + NN);
    }
  }
}

// ============================================================================
// Isotropic (Thermal) Source
// ============================================================================

template<int NStr>
void DisortFluxSolver<NStr>::computeIsotropicSource() 
{
  // Compute Planck function at each level
  std::vector<double> pkag(nlyr_ + 1);
  for (int lev = 0; lev <= nlyr_; ++lev) {
    pkag[lev] = planckFunction2(config_->wavenumber_low, config_->wavenumber_high,
                   config_->temperature[lev]);
  }

  for (int lc = 0; lc < nlyr_; ++lc) {
    // Build A = I - C
    isot_A_.noalias() = -phase_matrix_[lc];
    isot_A_.diagonal().array() += 1.0;
    Eigen::PartialPivLU<FullMat> lu(isot_A_);

    // Thermal expansion: B(tau) = xr0 + xr1 * tau
    double xr1 = 0.0;
    if (scaled_dtau_[lc] > 1e-4) {
      xr1 = (pkag[lc + 1] - pkag[lc]) / scaled_dtau_[lc];
    }
    double xr0 = pkag[lc] - xr1 * scaled_tau_cumulative_[lc];

    planck_intercept_[lc] = xr0;
    planck_slope_[lc] = xr1;

    // Solve for Z1
    isot_b_.setConstant((1.0 - scaled_ssa_[lc]) * xr1);
    isot_z1_.noalias() = lu.solve(isot_b_);

    // Solve for Z0
    isot_b_.array() = (1.0 - scaled_ssa_[lc]) * xr0 + quad_angle_.array() * isot_z1_.array();
    FullVec isot_z0 = lu.solve(isot_b_);

    // Store with angular reordering
    for (int iq = 0; iq < NN; ++iq) {
      thermal_z0_[lc](NN + iq) = isot_z0(iq);
      thermal_z0_[lc](NN - iq - 1) = isot_z0(iq + NN);
      thermal_z1_[lc](NN + iq) = isot_z1_(iq);
      thermal_z1_[lc](NN - iq - 1) = isot_z1_(iq + NN);
    }
  }
}

// ============================================================================
// Band Matrix Construction
// ============================================================================

template<int NStr>
void DisortFluxSolver<NStr>::setMatrix() 
{
  const int mi9m2 = 9 * NN - 2;
  const int nnlyri = NStr * nlyr_;
  const int ncd = 3 * NN - 1;
  const int lda = 3 * ncd + 1;
  const int nshift = lda - 2 * NStr + 1;

  band_matrix_.resize(mi9m2, nnlyri);
  band_matrix_.setZero();

  int ncol = 0;

  // Layer continuity
  for (int lc = 0; lc < ncut_; ++lc) {
    // Exponential scaling factors
    work_vec_.setZero();
    for (int iq = 0; iq < NN; ++iq) {
      work_vec_(iq) = std::exp(eigenvalues_[lc](iq) * scaled_dtau_[lc]);
    }

    int jcol = 0;

    // Downward streams
    for (int iq = 0; iq < NN; ++iq) {
      int irow = nshift - jcol - 1;
      for (int jq = 0; jq < NStr; ++jq) {
        if (irow + NStr >= 0 && irow + NStr < band_matrix_.rows() && ncol < band_matrix_.cols()) {
          band_matrix_(irow + NStr, ncol) = eigenvectors_[lc](jq, iq);
        }
        if (irow >= 0 && irow < band_matrix_.rows() && ncol < band_matrix_.cols()) {
          band_matrix_(irow, ncol) = -eigenvectors_[lc](jq, iq) * work_vec_(iq);
        }
        irow++;
      }
      ncol++;
      jcol++;
    }

    // Upward streams
    for (int iq = NN; iq < NStr; ++iq) {
      int irow = nshift - jcol - 1;
      for (int jq = 0; jq < NStr; ++jq) {
        band_matrix_(irow + NStr, ncol) = eigenvectors_[lc](jq, iq) * work_vec_(NStr - iq - 1);
        band_matrix_(irow, ncol) = -eigenvectors_[lc](jq, iq);
        irow++;
      }
      ncol++;
      jcol++;
    }
  }

  // Top boundary condition: no downward diffuse radiation
  int jcol = 0;
  for (int iq = 0; iq < NN; ++iq) {
    double expa = std::exp(eigenvalues_[0](iq) * scaled_tau_cumulative_[1]);
    int irow = nshift - jcol + NN - 1;
    for (int jq = NN - 1; jq >= 0; --jq) {
      band_matrix_(irow, jcol) = eigenvectors_[0](jq, iq) * expa;
      irow++;
    }
    jcol++;
  }
  for (int iq = NN; iq < NStr; ++iq) {
    int irow = nshift - jcol + NN - 1;
    for (int jq = NN - 1; jq >= 0; --jq) {
      band_matrix_(irow, jcol) = eigenvectors_[0](jq, iq);
      irow++;
    }
    jcol++;
  }

  // Bottom boundary condition: Lambertian surface reflection
  int nncol = ncol - NStr - 1;
  jcol = 0;
  const int ncut = ncut_;
  const double surface_albedo = config_->surface_albedo;

  // Downward streams at bottom
  for (int iq = 0; iq < NN; ++iq) {
    nncol++;
    int irow = nshift - jcol + NStr - 1;
    for (int jq = NN; jq < NStr; ++jq) {
      if (lyrcut_ || config_->use_diffusion_lower_bc) {
        band_matrix_(irow, nncol) = eigenvectors_[ncut - 1](jq, iq);
      } else {
        // Lambertian + azimuth_mode=0: surface reflection coupling
        double refl_sum = 0.0;
        for (int k = 0; k < NN; ++k) {
          refl_sum += quad_weight_(k) * quad_angle_(k) * surface_albedo
                * eigenvectors_[ncut - 1](NN - 1 - k, iq);
        }
        band_matrix_(irow, nncol) = eigenvectors_[ncut - 1](jq, iq) - 2.0 * refl_sum;
      }
      irow++;
    }
    jcol++;
  }

  // Upward streams at bottom
  for (int iq = NN; iq < NStr; ++iq) {
    nncol++;
    int irow = nshift - jcol + NStr - 1;
    // Use negative eigenvalue index (NStr-iq-1) for stability, matching DisortSolver's
    // work_vec_(nstr_-iq-1) convention from the continuity loop.
    double expa = std::exp(eigenvalues_[ncut - 1](NStr - iq - 1) * scaled_dtau_[ncut - 1]);
    for (int jq = NN; jq < NStr; ++jq) {
      if (lyrcut_ || config_->use_diffusion_lower_bc) {
        band_matrix_(irow, nncol) = eigenvectors_[ncut - 1](jq, iq) * expa;
      } else {
        double refl_sum = 0.0;
        for (int k = 0; k < NN; ++k) {
          refl_sum += quad_weight_(k) * quad_angle_(k) * surface_albedo
                * eigenvectors_[ncut - 1](NN - 1 - k, iq);
        }
        band_matrix_(irow, nncol) = (eigenvectors_[ncut - 1](jq, iq) - 2.0 * refl_sum) * expa;
      }
      irow++;
    }
    jcol++;
  }
}

// ============================================================================
// Solve Boundary Value Problem
// ============================================================================

template<int NStr>
void DisortFluxSolver<NStr>::solve0() 
{
  const int ncol = NStr * ncut_;
  Eigen::VectorXd& b = solve0_b_;
  b.head(ncol).setZero();

  const double isotropic_flux_top = config_->isotropic_flux_top;
  const double direct_beam_flux = config_->direct_beam_flux;
  const double direct_beam_mu = config_->direct_beam_mu;
  const int ncut = ncut_;

  // --- Top Boundary Condition ---
  for (int iq = 0; iq < NN; ++iq) {
    double sum = 0.0;

    if (direct_beam_flux > 0.0) {
      sum -= beam_particular_[0](NN - iq - 1);
    }
    if (config_->use_thermal_emission) {
      sum -= thermal_z0_[0](NN - iq - 1);
    }
    sum += isotropic_flux_top + planck_top_;

    b(iq) = sum;
  }

  // --- Bottom Boundary Condition ---
  if (config_->use_diffusion_lower_bc) {
    // Diffusion approximation: I_up(μ_i) = B(T_bottom) + μ_i × dB/dτ
    for (int iq = 0; iq < NN; ++iq) {
      int idx = ncol - NN + iq;
      double sum = 0.0;

      // Beam particular solution at bottom
      if (direct_beam_flux > 0.0) {
        if (config_->use_spherical_beam) {
          sum -= std::exp(-beam_particular_atten_[ncut - 1] * scaled_tau_cumulative_[ncut])
               * (beam_particular_[ncut - 1](NN + iq)
                + beam_particular_slope_[ncut - 1](NN + iq) * scaled_tau_cumulative_[ncut]);
        } else {
          sum -= beam_particular_[ncut - 1](NN + iq) * beam_transmission_[ncut];
        }
      }

      // Thermal particular solution at bottom
      if (config_->use_thermal_emission) {
        sum -= thermal_z0_[ncut - 1](NN + iq);
        sum -= thermal_z1_[ncut - 1](NN + iq) * scaled_tau_cumulative_[ncut];
      }

      // Diffusion source: B(T_bottom) + μ_i × dB/dτ
      sum += planck_bottom_ + quad_angle_(iq) * planck_bottom_deriv_;

      b(idx) = sum;
    }
  } else {
  for (int iq = 0; iq < NN; ++iq) {
    int idx = ncol - NN + iq;
    double sum = 0.0;

    // Diffuse particular solution reflection sum
    if (!lyrcut_) {
      double refl_sum = 0.0;
      for (int jq = 0; jq < NN; ++jq) {
        double ps_contrib = 0.0;
        if (direct_beam_flux > 0.0) {
          if (config_->use_spherical_beam) {
            ps_contrib += std::exp(-beam_particular_atten_[ncut - 1] * scaled_tau_cumulative_[ncut])
                    * (beam_particular_[ncut - 1](NN - 1 - jq)
                     + beam_particular_slope_[ncut - 1](NN - 1 - jq) * scaled_tau_cumulative_[ncut]);
          } else {
            ps_contrib += beam_particular_[ncut - 1](NN - 1 - jq) * beam_transmission_[ncut];
          }
        }
        if (config_->use_thermal_emission) {
          ps_contrib += thermal_z0_[ncut - 1](NN - 1 - jq)
                + thermal_z1_[ncut - 1](NN - 1 - jq) * scaled_tau_cumulative_[ncut];
        }
        refl_sum += quad_weight_(jq) * quad_angle_(jq) * surface_refl_quad_(iq, jq + 1) * ps_contrib;
      }
      sum += 2.0 * refl_sum;
    }

    // Beam source at bottom
    if (direct_beam_flux > 0.0) {
      if (config_->use_spherical_beam) {
        sum -= std::exp(-beam_particular_atten_[ncut - 1] * scaled_tau_cumulative_[ncut])
             * (beam_particular_[ncut - 1](NN + iq)
              + beam_particular_slope_[ncut - 1](NN + iq) * scaled_tau_cumulative_[ncut]);
      } else {
        sum -= beam_particular_[ncut - 1](NN + iq) * beam_transmission_[ncut];
      }
      if (!lyrcut_) {
        sum += surface_refl_quad_(iq, 0) * direct_beam_mu * direct_beam_flux * beam_transmission_[ncut] / M_PI;
      }
    }

    // Thermal source at bottom
    if (config_->use_thermal_emission) {
      sum -= thermal_z0_[ncut - 1](NN + iq);
      sum -= thermal_z1_[ncut - 1](NN + iq) * scaled_tau_cumulative_[ncut];
    }

    // Bottom boundary emission
    sum += config_->isotropic_flux_bottom;
    if (config_->use_thermal_emission) {
      sum += surface_emis_quad_(iq) * planck_bottom_;
    }

    b(idx) = sum;
  }
  } // end standard bottom BC

  // --- Layer Interface Continuity ---
  for (int lc = 1; lc < ncut; ++lc) {
    int ipnt = lc * NStr - NN;
    for (int iq = 0; iq < NStr; ++iq) {
      double sum = 0.0;

      if (direct_beam_flux > 0.0) {
        if (config_->use_spherical_beam) {
          sum += std::exp(-beam_particular_atten_[lc] * scaled_tau_cumulative_[lc])
               * (beam_particular_[lc](iq) + beam_particular_slope_[lc](iq) * scaled_tau_cumulative_[lc]);
          sum -= std::exp(-beam_particular_atten_[lc - 1] * scaled_tau_cumulative_[lc])
               * (beam_particular_[lc - 1](iq) + beam_particular_slope_[lc - 1](iq) * scaled_tau_cumulative_[lc]);
        } else {
          sum += beam_particular_[lc](iq) * beam_transmission_[lc];
          sum -= beam_particular_[lc - 1](iq) * beam_transmission_[lc];
        }
      }

      if (config_->use_thermal_emission) {
        sum += thermal_z0_[lc](iq) - thermal_z0_[lc - 1](iq);
        sum += (thermal_z1_[lc](iq) - thermal_z1_[lc - 1](iq)) * scaled_tau_cumulative_[lc];
      }

      b(ipnt + iq) = sum;
    }
  }

  // Solve banded linear system
  const int ncd = 3 * NN - 1;
  const int lda = band_matrix_.rows();
  detail::bandFactor(band_matrix_.data(), lda, ncol, ncd, ncd, band_pivot_.data());
  detail::bandSolve(band_matrix_.data(), lda, ncol, ncd, ncd, band_pivot_.data(), b.data());

  // Extract integration constants with reordering
  integration_constants_.setZero();
  for (int lc = 0; lc < ncut_; ++lc) {
    int ipnt = (lc + 1) * NStr - NN - 1;
    for (int iq = 0; iq < NN; ++iq) {
      int ll_idx = lc * NStr + (NN - iq - 1);
      int b_idx = ipnt - iq;
      if (b_idx >= 0 && b_idx < ncol && ll_idx < integration_constants_.size()) {
        integration_constants_(ll_idx) = b(b_idx);
      }
    }
    for (int iq = 0; iq < NN; ++iq) {
      int ll_idx = lc * NStr + (NN + iq);
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

template<int NStr>
void DisortFluxSolver<NStr>::computeFluxes(FluxResult& result) 
{
  const double direct_beam_flux = config_->direct_beam_flux;
  const double direct_beam_mu = config_->direct_beam_mu;

  // Precomputed exp coefficients: exp_coeffs(jq) = LL(jq) * exp(...)
  // Factored out of the iq loop — only depends on eigenmode jq, not angle iq.
  FullVec exp_coeffs;

  // Unscaled cumulative optical depth, computed incrementally
  double tauc_lu = 0.0;

  // Output at all layer boundaries (num_layers+1 levels)
  for (int lu = 0; lu <= nlyr_; ++lu) {
    // Determine which layer this level belongs to.
    int lyu = (lu == 0) ? 0 : lu - 1;

    // Check if level is beyond the cutoff BEFORE capping lyu
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

    // Cap lyu to valid range (after lyrcut check)
    if (lyu >= ncut_) lyu = ncut_ - 1;

    // Scaled optical depth at this level
    const double utaupr = scaled_tau_cumulative_[lu];

    // Direct beam flux
    double fact = 0.0;
    double dirint = 0.0;
    double fldir = 0.0;
    double flux_direct_beam_val = 0.0;

    // tauc_lu is the unscaled cumulative optical depth at level lu,
    // computed incrementally (updated at end of loop body)

    if (direct_beam_flux > 0.0) {
      if (config_->use_spherical_beam) {
        const double ch_lyu = chapman_[lyu];
        fact = std::exp(-utaupr / ch_lyu);
        flux_direct_beam_val = std::abs(direct_beam_mu) * direct_beam_flux * std::exp(-tauc_lu / ch_lyu);
      } else {
        fact = std::exp(-utaupr / direct_beam_mu);
        flux_direct_beam_val = direct_beam_mu * direct_beam_flux * std::exp(-tauc_lu / direct_beam_mu);
      }
      dirint = direct_beam_flux * fact;
      fldir = direct_beam_mu * direct_beam_flux * fact;
    }

    // Precompute exp() * integration_constants for all eigenmodes (once per level).
    // These only depend on jq and lyu, not on iq — saves NStr-1 redundant exp() per mode.
    const int ll_offset = lyu * NStr;
    const double tau_bot = scaled_tau_cumulative_[lyu + 1];
    const double tau_top = scaled_tau_cumulative_[lyu];
    for (int jq = 0; jq < NN; ++jq) {
      exp_coeffs(jq) = integration_constants_(jq + ll_offset)
                        * std::exp(-eigenvalues_[lyu](jq) * (utaupr - tau_bot));
    }
    for (int jq = NN; jq < NStr; ++jq) {
      exp_coeffs(jq) = integration_constants_(jq + ll_offset)
                        * std::exp(-eigenvalues_[lyu](jq) * (utaupr - tau_top));
    }

    // Compute diffuse intensity at all quadrature angles via matrix-vector product:
    //   intensity = eigenvectors * exp_coeffs
    // Then add particular solutions (beam, thermal).
    FullVec intensity = eigenvectors_[lyu] * exp_coeffs;

    if (direct_beam_flux > 0.0) {
      if (config_->use_spherical_beam) {
        const double beam_exp = std::exp(-beam_particular_atten_[lyu] * utaupr);
        intensity += beam_exp * (beam_particular_[lyu] + beam_particular_slope_[lyu] * utaupr);
      } else {
        intensity += beam_particular_[lyu] * fact;
      }
    }

    if (config_->use_thermal_emission) {
      intensity += thermal_z0_[lyu] + thermal_z1_[lyu] * utaupr;
    }

    // Accumulate fluxes and mean intensities from quadrature
    double mean_intensity = 0.0;
    double mean_intensity_down = 0.0;
    double mean_intensity_up = 0.0;
    double fldn = 0.0;
    double flux_up = 0.0;

    // Downward angles (iq = 0 to NN-1)
    for (int iq = 0; iq < NN; ++iq) {
      const int cwt_idx = NN - iq - 1;
      const double val = intensity(iq);
      const double wt = quad_weight_(cwt_idx);
      mean_intensity += wt * val;
      mean_intensity_down += wt * val;
      fldn += wt * val * quad_angle_(cwt_idx);
    }

    // Upward angles (iq = NN to NStr-1)
    for (int iq = NN; iq < NStr; ++iq) {
      const int cwt_idx = iq - NN;
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
    const double flux_down = fdntot - flux_direct_beam_val;

    mean_intensity = (2.0 * M_PI * mean_intensity + dirint) / (4.0 * M_PI);
    const double mean_intensity_direct_beam = dirint / (4.0 * M_PI);
    mean_intensity_down = (2.0 * M_PI * mean_intensity_down) / (4.0 * M_PI);
    mean_intensity_up = (2.0 * M_PI * mean_intensity_up) / (4.0 * M_PI);

    // Flux divergence
    double plsorc = 0.0;
    if (config_->use_thermal_emission) {
      plsorc = planck_intercept_[lyu] + planck_slope_[lyu] * utaupr;
    }
    const double flux_tau_divergence = (1.0 - config_->single_scat_albedo[lyu])
                       * 4.0 * M_PI * (mean_intensity - plsorc);

    result.flux_direct_beam[lu] = flux_direct_beam_val;
    result.flux_down[lu] = flux_down;
    result.flux_up[lu] = flux_up;
    result.flux_tau_divergence[lu] = flux_tau_divergence;
    result.mean_intensity[lu] = mean_intensity;
    result.mean_intensity_down[lu] = mean_intensity_down;
    result.mean_intensity_up[lu] = mean_intensity_up;
    result.mean_intensity_direct_beam[lu] = mean_intensity_direct_beam;

    // Advance unscaled cumulative optical depth for next level
    if (lu < nlyr_) tauc_lu += config_->delta_tau[lu];
  }
}

// ============================================================================
// Chapman Function (file-static helper, shared with DisortSolver)
// ============================================================================

static double chapmanImpl(int lc, double tau_frac,
              const std::vector<double>& level_altitudes,
              const std::vector<double>& dtau_lyr,
              int num_layers, double r, double direct_beam_mu)
{
  const double DEG = M_PI / 180.0;
  const double zenang = std::acos(direct_beam_mu) * 180.0 / M_PI;
  const double zenrad = zenang * DEG;

  auto ZD = [&](int j) -> double { return level_altitudes[j]; };
  auto DTAU_C = [&](int j) -> double { return dtau_lyr[j - 1]; };

  const double xp = r + ZD(lc) + (ZD(lc - 1) - ZD(lc)) * tau_frac;
  const double xpsinz = xp * std::sin(zenrad);

  if (zenang > 90.0 && xpsinz < r) {
    return 1.0e+20;
  }

  int id = lc;
  if (zenang > 90.0) {
    for (int j = lc; j <= num_layers; ++j) {
      if (xpsinz < (ZD(j - 1) + r) && xpsinz >= (ZD(j) + r)) {
        id = j;
      }
    }
  }

  double sum = 0.0;
  for (int j = 1; j <= id; ++j) {
    double fact = 1.0;
    double fact2 = 1.0;
    if (j > lc) {
      fact = 2.0;
    } else if (j == lc && lc == id && zenang > 90.0) {
      fact2 = -1.0;
    }

    const double rj = r + ZD(j - 1);
    double rjp1 = r + ZD(j);
    if (j == lc && id == lc) {
      rjp1 = xp;
    }

    const double dhj = ZD(j - 1) - ZD(j);
    double dsj;
    if (id > lc && j == id) {
      dsj = std::sqrt(rj * rj - xpsinz * xpsinz);
    } else {
      dsj = std::sqrt(rj * rj - xpsinz * xpsinz)
        - fact2 * std::sqrt(rjp1 * rjp1 - xpsinz * xpsinz);
    }
    sum += DTAU_C(j) * fact * dsj / dhj;
  }

  if (id > lc) {
    const double dhj = ZD(lc - 1) - ZD(lc);
    const double dsj = std::sqrt(xp * xp - xpsinz * xpsinz)
             - std::sqrt((ZD(lc) + r) * (ZD(lc) + r) - xpsinz * xpsinz);
    sum += DTAU_C(lc) * dsj / dhj;
  }

  return sum;
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template class DisortFluxSolver<4>;
template class DisortFluxSolver<6>;
template class DisortFluxSolver<8>;
template class DisortFluxSolver<10>;
template class DisortFluxSolver<12>;
template class DisortFluxSolver<14>;
template class DisortFluxSolver<16>;
template class DisortFluxSolver<32>;
template class DisortFluxSolver<64>;

} // namespace disortpp
