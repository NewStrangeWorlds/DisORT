#pragma once

#include "DisortConfig.hpp"
#include "DisortResult.hpp"
#include "LinearAlgebra.hpp"
#include "Legendre.hpp"
#include "Utils.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace disortpp {

// ============================================================================
// Contiguous per-layer storage wrappers
// ============================================================================

/// Stores nlyr column vectors of length `rows` in a single contiguous MatrixXd.
/// Access via operator[] returns an Eigen column expression (zero-copy view).
struct LayerVectors {
  Eigen::MatrixXd data_;
  void resize(int rows, int ncols) { data_.resize(rows, ncols); data_.setZero(); }
  bool empty() const { return data_.size() == 0; }
  auto operator[](int lc)       { return data_.col(lc); }
  auto operator[](int lc) const { return data_.col(lc); }
};

/// Stores nlyr matrices of size rows×cols in a single contiguous MatrixXd.
/// Access via operator[] returns an Eigen block expression (zero-copy view).
struct LayerMatrices {
  Eigen::MatrixXd data_;
  int cols_per_layer_ = 0;
  void resize(int rows, int cols, int nlyr) {
    cols_per_layer_ = cols;
    data_.resize(rows, cols * nlyr);
    data_.setZero();
  }
  auto operator[](int lc)       { return data_.middleCols(lc * cols_per_layer_, cols_per_layer_); }
  auto operator[](int lc) const { return data_.middleCols(lc * cols_per_layer_, cols_per_layer_); }
};

/**
 * @brief Main DISORT radiative transfer solver
 *
 * This class implements the discrete ordinates method for solving the radiative
 * transfer equation in plane-parallel atmospheres. It converts the original
 * C implementation to modern C++17.
 *
 * **Algorithm Overview:**
 * 1. Setup: Delta-M transformation, quadrature angles (disortSet)
 * 2. For each azimuthal mode:
 *    a. Solve eigenvalue problem for each layer (solveEigen)
 *    b. Build coefficient matrix for boundary conditions (setMatrix)
 *    c. Compute particular solutions (beam and thermal sources)
 *    d. Solve boundary value problem
 *    e. Compute intensities and fluxes
 *
 * **Key Features:**
 * - Delta-M scaling for forward-peaked phase functions
 * - Multiple scattering with arbitrary phase functions
 * - Thermal emission (Planck function)
 * - Lambertian or BRDF surface reflection
 * - User-specified output levels and angles
 *
 * **References:**
 * - Stamnes et al. (1988) "Numerically stable algorithm for discrete-ordinate-method
 *   radiative transfer in multiple scattering and emitting layered media"
 */
class DisortSolver {
public:
  /**
   * @brief Construct solver (lightweight, no allocation)
   */
  DisortSolver() = default;

  /**
   * @brief Solve the radiative transfer problem
   *
   * Main entry point for DISORT algorithm.
   *
   * @param config Input configuration (layers, optical properties, boundary conditions)
   *               Note: config may be modified to set automatic dimensions (num_user_tau, num_user_mu)
   * @return DisortResult containing intensities, fluxes, and other outputs
   * @throws std::invalid_argument if configuration is invalid
   * @throws std::runtime_error if numerical problems occur
   */
  DisortResult solve(DisortConfig& config);

private:
  // ========================================================================
  // Setup and Initialization
  // ========================================================================

  /**
   * @brief Setup computational arrays and apply delta-M transformation
   *
   * Converts c_disort_set() from original code.
   *
   * Sets up:
   * - Output optical depth levels (utaupr)
   * - Delta-M scaled optical properties (dtaucpr, oprim, gl)
   * - Chapman function for beam attenuation (ch, expbea)
   * - Layer cutoff for absorption (lyrcut, ncut)
   * - Quadrature angles and weights (cmu, cwt)
   * - User angle mapping (layru)
   *
   * @param config Input configuration (may be modified to set output levels)
   * @param deltam Whether to apply delta-M transformation
   */
  void disortSet(DisortConfig& config, bool deltam);

  /**
   * @brief Compute Gaussian quadrature angles and weights
   *
   * Computes Gauss-Legendre quadrature on interval (0,1).
   *
   * @param n Number of quadrature points
   * @param cmu [out] Cosines of quadrature angles (size 2n)
   * @param cwt [out] Quadrature weights (size 2n)
   */
  void gaussianQuadrature(int n, Eigen::VectorXd& cmu, Eigen::VectorXd& cwt);

  /**
   * @brief Compute Chapman function for spherical geometry
   *
   * Chapman function describes path length through curved atmosphere.
   * Used for accurate beam attenuation in spherical geometry.
   *
   * @param lc Layer index
   * @param tau_frac Fractional depth within layer (0.0 = top, 1.0 = bottom)
   * @return Chapman function value
   */
  double chapmanFunction(int lc, double tau_frac);

  // ========================================================================
  // Eigenvalue Problem
  // ========================================================================

  /**
   * @brief Solve eigenvalue problem for a single layer
   *
   * Converts c_solve_eigen() from original code.
   *
   * Computes eigenvalues (k) and eigenvectors (G) for the homogeneous
   * solution of the radiative transfer equation in a single layer.
   *
   * **Algorithm (Stamnes et al. 1988, Sec. 2.3):**
   * 1. Build coefficient matrix CC from phase function moments
   * 2. Form reduced eigenvalue problem: (α+β)(α-β)
   * 3. Solve for eigenvalues λ (using Eigen)
   * 4. Compute k = sqrt(λ)
   * 5. Recover eigenvectors G+, G- from reduced eigenvectors
   *
   * @param lc Layer index (0-based)
   * @param azimuth_mode Azimuthal mode number
   */
  void solveEigen(int lc, int azimuth_mode);

  /**
   * @brief Core eigenvalue computation with compile-time NN for fixed-size Eigen types
   *
   * Performs steps 4–7 of solveEigen (alpha/beta construction, symmetrization,
   * Cholesky + eigensolve, eigenvector recovery) using stack-allocated fixed-size
   * matrices. Called after solveEigen computes cc_top_half_ and phase_matrix_.
   *
   * @tparam NN Half the number of streams (compile-time constant)
   * @param lc Layer index
   */
  template<int NN>
  void solveEigenCore(int lc);

  /// Dynamic-size fallback for solveEigenCore when NN is not in {2, 4, 8, 16}
  void solveEigenCoreDynamic(int lc);

  /**
   * @brief Build coefficient matrix for boundary conditions
   *
   * Converts c_set_matrix() from original code.
   *
   * Constructs the banded coefficient matrix for the system of equations
   * arising from:
   * - Continuity of intensity at layer interfaces
   * - Top boundary condition (no downward diffuse radiation)
   * - Bottom boundary condition (Lambertian or BRDF surface)
   *
   * Uses scaling transformation for numerical stability.
   *
   * @param kronecker_mode_0 Kronecker delta (1 if azimuth_mode=0, 0 otherwise)
   * @param lyrcut Whether bottom layers are truncated
   */
  void setMatrix(double kronecker_mode_0, bool lyrcut);

  // ========================================================================
  // Particular Solutions (Source Terms)
  // ========================================================================

  /**
   * @brief Compute beam source particular solution
   *
   * Converts c_upbeam() from original code.
   * Computes particular solution for beam source (solar/stellar radiation).
   */
  void computeBeamSource(int azimuth_mode);

  /**
   * @brief Compute pseudo-spherical beam source (use_spherical_beam=true)
   *
   * Converts c_set_coefficients_beam_source() + c_upbeam_pseudo_spherical().
   * Populates beam_particular_ (ZBEAM0), beam_particular_slope_ (ZBEAM1), beam_particular_atten_ (ZBEAMA=XBA).
   */
  void computeBeamSourceSpherical(int azimuth_mode);

  /**
   * @brief Compute isotropic source particular solution
   *
   * Converts c_upisot() from original code.
   * Computes particular solution for thermal emission (Planck source).
   */
  void computeIsotropicSource(int azimuth_mode);

  /**
   * @brief Solve boundary value problem
   *
   * Converts c_solve0() from original code.
   * Constructs RHS vector from boundary conditions and solves the banded
   * linear system to find integration constants.
   *
   * @param azimuth_mode Azimuthal mode number
   * @param kronecker_mode_0 Kronecker delta (1 if azimuth_mode=0, 0 otherwise)
   */
  void solve0(int azimuth_mode, double kronecker_mode_0);

  // ========================================================================
  // Flux and Intensity Computation
  // ========================================================================

  /**
   * @brief Compute radiative fluxes and mean intensities
   *
   * Converts c_fluxes() from original code.
   * Computes upward and downward fluxes, mean intensities, and flux divergence
   * at user output levels. Only called for azimuth_mode=0 (azimuthally averaged).
   *
   * @param result Reference to result structure where fluxes will be stored
   */
  void computeFluxes(DisortResult& result);

  /**
   * @brief Interpolate eigenvectors to user angles
   *
   * Converts c_interp_eigenvec() from original code.
   * Projects eigenvectors (at computational angles) onto user angles via
   * Legendre expansion. Fills eigenvec_user_[lc].
   *
   * @param lc Layer index (0-based)
   * @param azimuth_mode Azimuthal mode number
   */
  void interpEigenvec(int lc, int azimuth_mode);

  /**
   * @brief Interpolate source functions to user angles
   *
   * Converts beam-source part of c_interp_source() from original code.
   * Plane-parallel case only. Fills beam_source_user_[lc].
   *
   * @param lc Layer index (0-based)
   * @param azimuth_mode Azimuthal mode number
   * @param kronecker_mode_0 Kronecker delta (1 if azimuth_mode=0, 0 otherwise)
   */
  void interpSource(int lc, int azimuth_mode, double kronecker_mode_0);

  /**
   * @brief Compute surface bidirectional reflectivity
   *
   * Converts c_surface_bidir() from original code (NMUG=50 Gauss-Legendre quadrature).
   * Fills surface_refl_quad_, surface_emis_quad_ (at computational angles) and surface_refl_user_, surface_emis_user_ (at user angles).
   * - Lambertian: surface_refl_quad_=surface_albedo, surface_emis_quad_=1-surface_albedo, surface_refl_user_=surface_albedo, surface_emis_user_=1-surface_albedo (azimuth_mode=0)
   * - BRDF: integrates bidirReflectivity() over azimuthal quadrature
   *
   * @param kronecker_mode_0 Kronecker delta (1 if azimuth_mode=0, 0 otherwise)
   * @param azimuth_mode Azimuthal mode number
   */
  void computeSurfaceBidir(double kronecker_mode_0, int azimuth_mode);

  /**
   * @brief Evaluate BRDF reflectivity model
   *
   * Dispatches to the appropriate BRDF model based on config_->flags.brdf_type.
   * Converts c_bidir_reflectivity() from original code.
   *
   * @param mu  Cosine of outgoing (reflected) polar angle
   * @param mup Cosine of incoming polar angle
   * @param dphi Azimuthal angle difference [radians]
   * @return BRDF value at this geometry
   */
  double bidirReflectivity(double mu, double mup, double dphi) const;

  /**
   * @brief Hapke BRDF model
   *
   * Converts c_bidir_reflectivity_hapke() from original code.
   * Parameters are hardcoded (w=0.6, hh=0.06, b0=1).
   *
   * @param mu  Cosine of outgoing polar angle
   * @param mup Cosine of incoming polar angle
   * @param dphi Azimuthal angle difference [radians]
   * @return Hapke BRDF value
   */
  double bidirReflectivityHapke(double mu, double mup, double dphi) const;

  /**
   * @brief RPV (Rahman-Pinty-Verstraete) BRDF model
   *
   * Converts c_bidir_reflectivity_rpv() from original code.
   *
   * @param rpv  RPV BRDF parameters
   * @param mu1  Cosine of outgoing polar angle
   * @param mu2  Cosine of incoming polar angle
   * @param phi  Azimuthal angle difference [radians]
   * @param badmu Minimum valid mu (0.0 if no limit)
   * @return RPV BRDF value
   */
  double bidirReflectivityRpv(const RpvBrdfSpec& rpv,
                double mu1, double mu2, double phi,
                double badmu) const;

  /**
   * @brief Ross-Li (Ambrals) BRDF model
   *
   * Combines Ross-Thick volumetric kernel and Li-Sparse-Reciprocal geometric kernel.
   * Port of BRDF_ROSSLI from Fortran DISORT V3 (BDREF.f).
   *
   * @param mu  Cosine of outgoing polar angle
   * @param mup Cosine of incoming polar angle
   * @param dphi Azimuthal angle difference [radians]
   * @return Ross-Li BRDF value
   */
  double bidirReflectivityRossLi(double mu, double mup, double dphi) const;

  /**
   * @brief Cox-Munk ocean BRDF model
   *
   * 1D Gaussian rough ocean surface with Fresnel reflectance and optional shadowing.
   * Port of OCEABRDF2 from Fortran DISORT V3 (BDREF.f).
   *
   * @param mu  Cosine of outgoing polar angle
   * @param mup Cosine of incoming polar angle
   * @param dphi Azimuthal angle difference [radians]
   * @return Cox-Munk BRDF value
   */
  double bidirReflectivityCoxMunk(double mu, double mup, double dphi) const;

  /// Tsang shadowing function for Cox-Munk BRDF
  static double shadowEta(double cos_theta, double sigma_sq);

  /**
   * @brief Compute intensities at user angles and accumulate azimuthal modes
   *
   * Converts c_user_intensities() from original code.
   * Integrates the characteristic equation for each (output level, user angle) pair.
   * Modifies eigenvec_user_[lc] in place (multiplies by integration constants).
   * Accumulates into result.intensity and result.intensity_azimuth_avg.
   *
   * @param azimuth_mode Azimuthal mode number
   * @param kronecker_mode_0 Kronecker delta (1 if azimuth_mode=0, 0 otherwise)
   * @param result Output result structure
   * @param bplanck Planck function for bottom boundary temperature
   * @param tplanck Planck function for top boundary temperature
   */
  double computeUserIntensities(int azimuth_mode, double kronecker_mode_0, DisortResult& result,
                  double bplanck, double tplanck);

  // ========================================================================
  // Working Arrays (Pre-allocated for efficiency)
  // ========================================================================

  // Configuration (pointer, not owned)
  DisortConfig* config_ = nullptr;

  // Dimensions
  int nlyr_;    // Number of layers
  int nstr_;    // Number of streams
  int nn_;      // num_streams/2 (number of upward streams)
  int ntau_;    // Number of user output levels
  int numu_;    // Number of user output angles
  int nphi_;    // Number of user azimuth angles
  int ncut_;    // Number of computational layers (may be < num_layers if truncated)

  // Flags
  bool deltam_;        // Delta-M transformation enabled
  bool deltam_plus_;   // Delta-M+ transformation enabled (Lin et al. 2018)
  bool lyrcut_;        // Bottom layers truncated
  bool scat_yes_;      // Scattering present

  // Delta-M scaled optical properties
  std::vector<double> scaled_dtau_;  // Scaled optical depth (num_layers)
  std::vector<double> tau_cumulative_;     // Cumulative unscaled optical depth (0..num_layers) - needed for IMS correction
  std::vector<double> scaled_tau_cumulative_;   // Cumulative scaled optical depth (0..num_layers) - needs index 0
  std::vector<double> scaled_ssa_;    // Scaled single-scatter surface_albedo (num_layers)
  Eigen::MatrixXd scaled_phase_coeffs_;           // Scaled phase function coefficients scaled_phase_coeffs_(l, lc) [num_streams × (num_layers+1)]
  std::vector<double> forward_scatter_frac_;     // Forward-scatter fraction f (num_layers)

  // Beam attenuation
  std::vector<double> chapman_;       // Chapman function at layer midpoints (num_layers)
  std::vector<double> chapman_tau_;    // Chapman optical depth (0..num_layers) - needs index 0
  std::vector<double> beam_transmission_;   // exp(-chtau) beam transmission (0..num_layers) - needs index 0

  // User output levels in delta-M coords
  std::vector<double> scaled_tau_user_;   // Scaled user optical depths (num_user_tau)
  std::vector<int> layer_of_user_level_;       // Layer index for each user level (num_user_tau)

  // Quadrature angles and weights
  Eigen::VectorXd quad_angle_;       // Cosines of computational angles (num_streams)
  Eigen::VectorXd quad_weight_;       // Quadrature weights (num_streams)

  // Legendre polynomials (RowMajor: row=degree l, col=angle index; matches legendrePolynomialsFlat layout)
  using RowMajMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  RowMajMat legendre_quad_;  // At computational angles, legendre_quad_(l, jq) [(num_streams+1) × num_streams]
  RowMajMat legendre_user_;  // At user angles,          legendre_user_(l, iu)  [(num_streams+1) × num_user_mu]
  Eigen::VectorXd legendre_beam_;  // At beam angle,     legendre_beam_(l)      [num_streams+1]

  // Eigenvalues and eigenvectors (per layer, contiguous storage)
  LayerVectors eigenvalues_;       // Eigenvalues k: nstr × (nlyr+1) contiguous
  LayerMatrices eigenvectors_;     // Eigenvectors G: nstr × nstr×(nlyr+1) contiguous

  // Coefficient matrices for eigenvalue problem
  LayerMatrices phase_matrix_;     // Phase function matrix: nstr × nstr×nlyr contiguous
  Eigen::MatrixXd alpha_minus_beta_;      // α - β matrix (nn x nn)
  Eigen::MatrixXd alpha_plus_beta_;      // α + β matrix (nn x nn)
  Eigen::MatrixXd eigen_product_;    // (α+β)(α-β) eigenvalue matrix (nn x nn)
  Eigen::MatrixXd reduced_eigenvectors_;    // Eigenvectors of reduced problem (nn x nn)
  Eigen::VectorXd reduced_eigenvalues_;     // Eigenvalues of reduced problem (nn)

  // Boundary condition coefficient matrix
  Eigen::MatrixXd band_matrix_;    // Banded matrix for boundary value problem
  std::vector<int> band_pivot_;  // Pivot indices from banded LU factorization

  // Particular solutions (contiguous storage)
  LayerVectors beam_particular_;         // Beam source ZZ / ZBEAM0: nstr × (nlyr+1) contiguous
  LayerVectors thermal_particular_z0_;   // Isotropic source Z0: nstr × (nlyr+1) contiguous
  LayerVectors thermal_particular_z1_;   // Isotropic source Z1: nstr × (nlyr+1) contiguous

  // Pseudo-spherical beam source (use_spherical_beam=true, port of KS Eq. 7-10)
  LayerVectors beam_particular_slope_;   // ZBEAM1 coefficient: nstr × (nlyr+1) contiguous
  std::vector<double> beam_particular_atten_; // ZBEAMA scalar per layer (=XBA=1/CH)

  // Solution vectors
  Eigen::VectorXd integration_constants_;       // Integration constants (num_streams*num_layers)

  // Output working arrays
  Eigen::MatrixXd intensity_quad_;      // Intensity at computational angles (num_streams x num_user_tau)

  // Work arrays
  Eigen::VectorXd work_vec_;       // Temporary workspace (num_streams)

  // Thermal emission working arrays (active when use_thermal_emission=true)
  double planck_bottom_ = 0.0;              // Planck function at bottom boundary
  double planck_top_ = 0.0;              // Planck function at top boundary (× emissivity_top)
  std::vector<double> planck_intercept_;           // Planck linear intercept per layer (num_layers)
  std::vector<double> planck_slope_;           // Planck linear slope per layer (num_layers)

  // User-angle intensity working arrays (contiguous storage, active when !comp_only_fluxes && use_user_mu)
  LayerMatrices eigenvec_user_;          // Eigenvectors at user angles: numu × nstr×nlyr contiguous
  LayerVectors beam_source_user_;        // Beam source at user angles: numu × nlyr contiguous
  LayerVectors thermal_source_z0_user_;  // Thermal source Z0 at user angles: numu × nlyr contiguous
  LayerVectors thermal_source_z1_user_;  // Thermal source Z1 at user angles: numu × nlyr contiguous
  LayerVectors beam_source_raw_;         // Beam source before reordering: nstr × nlyr contiguous

  Eigen::MatrixXd intensity_azim_mode_;                // Intensity for current azimuthal mode (num_user_mu x num_user_tau)
  Eigen::MatrixXd surface_refl_user_;                // Surface reflectivity at user angles (num_user_mu x (nn+1))
  Eigen::VectorXd surface_emis_user_;                // Surface emissivity at user angles (num_user_mu)

  // BRDF/surface bidirectional reflectivity at computational angles
  Eigen::MatrixXd surface_refl_quad_;  ///< Fourier coefficient of BRDF at computational angles (nn x (nn+1))
               ///<   row = outgoing stream index (0..nn-1)
               ///<   col 0 = direct beam, col 1..nn = diffuse quadrature streams
  Eigen::VectorXd surface_emis_quad_;  ///< Surface directional emissivity at computational angles (nn)

  // BRDF azimuthal integration quadrature (NMUG=200 Gauss-Legendre points on [-1,1])
  static constexpr int NMUG = 200;
  Eigen::VectorXd gmu_brdf_;  ///< Cosines of azimuthal quadrature angles (NMUG)
  Eigen::VectorXd gwt_brdf_;  ///< Azimuthal quadrature weights (NMUG)
  Eigen::MatrixXd cosmp_brdf_;  ///< Precomputed cos(m * PI * gmu_brdf_(k)) [nstr_ x NMUG/2]

  /// Fourier coefficients of beam-reflected BRDF at user angles: brdf_fourier_user_beam_(iu, m).
  /// Stores surface_refl_user_(iu, 0) for each azimuthal mode m=0..nstr_-1.
  /// Only allocated when BRDF surface + beam + user intensities are requested.
  Eigen::MatrixXd brdf_fourier_user_beam_;

  // ========================================================================
  // Allocation Cache (avoid redundant re-allocation on repeated solve() calls)
  // ========================================================================

  /// Dimensions from the last allocateWorkingArrays() call.
  /// Set to -1 so the first call always allocates.
  int alloc_nlyr_ = -1;
  int alloc_nstr_ = -1;
  int alloc_numu_ = -1;
  int alloc_ntau_ = -1;
  bool alloc_special_bc_        = false;
  bool alloc_compute_intensity_ = false;

  /// Pre-allocated result reused across calls with identical dimensions.
  DisortResult result_;

  // ========================================================================
  // Pre-allocated scratch buffers — avoid per-call heap allocation in hot path
  // ========================================================================

  // solveEigen() scratch (called num_layers*num_streams times per solve, e.g. 96×/solve for 6L+16S)
  std::vector<double> cmu_vec_;       ///< quad_angle_ as std::vector for legendrePolynomials (num_streams)
  Eigen::VectorXd     eigenvalues_tmp_;  ///< Eigenvalues scratch (nn)
  Eigen::MatrixXd     eigenvectors_tmp_; ///< Eigenvectors scratch (nn × nn)
  Eigen::EigenSolver<Eigen::MatrixXd> eig_solver_;  ///< Fallback for conservative scattering (omega=1)
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sym_eig_solver_;  ///< Fast path for SPD eigenvalue problem
  Eigen::LLT<Eigen::MatrixXd> cholesky_;            ///< Cholesky factor of -B_s for symmetrization
  Eigen::VectorXd similarity_d_;       ///< D = diag(sqrt(cwt * mu)) for eigenvalue symmetrization (nn)
  Eigen::VectorXd similarity_d_inv_;   ///< D^{-1} = diag(1/sqrt(cwt * mu)) (nn)
  Eigen::MatrixXd sym_A_s_;            ///< Scratch: symmetrized alpha_plus_beta_ (nn x nn)
  Eigen::MatrixXd sym_neg_B_s_;        ///< Scratch: symmetrized -alpha_minus_beta_ (nn x nn)
  Eigen::VectorXd v_tmp_;              ///< Scratch: eigenvector recovery from triangular solve (nn)
  std::vector<int>    sort_indices_tmp_;             ///< Eigenvalue sort permutation (nn)

  // Main azimuthal loop scratch (called num_streams times per solve, e.g. 16×)
  std::vector<double> umu_vec_;       ///< User angles as std::vector (num_user_mu) for legendrePolynomials

  // computeBeamSource() scratch (called num_streams times per solve, e.g. 16×)
  std::vector<double>                  umu0_vec_;  ///< Beam angle 1-element vector {-direct_beam_mu} for legendrePolynomials
  Eigen::MatrixXd                      beam_A_;    ///< Coefficient matrix (num_streams × num_streams)
  Eigen::VectorXd                      beam_b_;    ///< RHS vector (num_streams)
  Eigen::PartialPivLU<Eigen::MatrixXd> beam_lu_;   ///< Pre-allocated LU (avoids per-layer heap alloc)
  Eigen::VectorXd                      beam_x_;    ///< Solution vector x = lu.solve(b) (num_streams)

  // computeIsotropicSource() scratch (called num_layers times per solve when use_thermal_emission=true)
  Eigen::MatrixXd                      isot_A_;    ///< [I - C] matrix (num_streams × num_streams)
  Eigen::PartialPivLU<Eigen::MatrixXd> isot_lu_;   ///< Pre-allocated LU for isotropic source
  Eigen::VectorXd                      isot_b_;    ///< Shared RHS scratch (num_streams)
  Eigen::VectorXd                      isot_z1_;   ///< Z1 solution vector (num_streams)
  Eigen::VectorXd                      isot_z0_;   ///< Z0 solution vector (num_streams)

  // computeBeamSourceSpherical() scratch (same frequency as above when use_spherical_beam=true)
  Eigen::MatrixXd     spher_A_;       ///< Coefficient matrix (num_streams × num_streams)
  Eigen::VectorXd     spher_zj_;      ///< Particular solution vector (num_streams)
  Eigen::VectorXd     spher_wk_;      ///< Work vector (num_streams)

  // solve0() scratch (called num_streams times per solve, e.g. 16×)
  Eigen::VectorXd     solve0_b_;      ///< RHS vector (num_streams * num_layers)

  // solveEigen() phase_matrix_ DGEMM scratch (called num_layers*num_streams times per solve, e.g. 96×)
  RowMajMat           cc_gl_ylmc_;    ///< diag(gl_col) * ylmc_block[:, :nn] scratch [num_streams × nn]
  Eigen::MatrixXd     cc_top_half_;   ///< Top nn rows of phase_matrix_ before symmetry [nn × num_streams]

  // interpEigenvec() scratch (called num_layers*num_streams times per solve, e.g. 96×)
  Eigen::VectorXd     interp_wk_l_;   ///< Legendre projection scratch (num_streams)
  Eigen::MatrixXd     interp_W_;      ///< W = ylmc_cwt × evecc scratch [(num_streams) × num_streams]
  Eigen::MatrixXd     interp_GU_tmp_; ///< GU before permutation scratch [num_user_mu × num_streams]
  RowMajMat           interp_ylmc_cwt_; ///< ylmc × diag(cwt), updated per azimuth_mode [(num_streams+1) × num_streams]

  // interpSource() scratch (called num_layers*num_streams times per solve, e.g. 96×)
  Eigen::VectorXd     interp_psi0_;   ///< Beam source Legendre projection (num_streams)

  // computeUserIntensities() scratch (called ~num_layers*num_user_tau*num_user_mu times per solve)
  Eigen::VectorXd     ci_wk_neg_;     ///< exp(kk_neg * dtau) for full-layer integration (nn)
  Eigen::VectorXd     ci_wk_curr_neg_;///< exp(-kk_neg * dtau) for partial-layer integration (nn)
  Eigen::VectorXd     ci_wk_bnd_;     ///< exp(-kk_pos * dtau) for bottom boundary (nn)

  // applyIntensityCorrection() scratch (called once per solve when corint=true)
  std::vector<double> ic_phasa_;      ///< Exact phase function per layer (num_layers+1)
  std::vector<double> ic_phasm_;      ///< Delta-M phase function per layer (num_layers+1)
  std::vector<double> ic_phast_;      ///< TMS phase function per layer (num_layers+1)
  std::vector<double> ic_ssalb_;      ///< Single-scatter surface_albedo per layer (num_layers+1)
  std::vector<double> ic_oprim_;      ///< Scaled SSA per layer (num_layers+1)
  std::vector<double> ic_phirad_;     ///< Phi angles in radians (num_phi+1)

  // ========================================================================
  // Intensity Correction (Nakajima-Tanaka TMS/IMS)
  // ========================================================================

  /**
   * @brief Apply Nakajima-Tanaka intensity correction (TMS + IMS)
   *
   * Converts c_intensity_correction() from original code.
   * Corrects the delta-M-truncated single-scatter term (TMS) and
   * optionally applies the IMS near-aureole correction.
   * Only called when corint=true (deltam && direct_beam_flux>0 && scat_yes && !comp_only_fluxes).
   *
   * @param result Output result structure (intensity modified in place)
   */
  void applyIntensityCorrection(DisortResult& result);

  /**
   * @brief Single-scatter integral (exact or delta-M phase function)
   *
   * Converts c_single_scat() from original code.
   * Computes eq. STWL (65b,d,e).
   *
   * @param dither   Small value (100*DBL_EPSILON) to test near-singular geometry
   * @param layru    Layer index containing user level lu (1-based)
   * @param num_layers     Number of layers to integrate over
   * @param phase_function    Phase function per layer (1-based, size num_layers)
   * @param omega    Single-scatter surface_albedo per layer (1-based, size num_layers)
   * @param tau      Cumulative optical depths (0-based, size num_layers+1)
   * @param mu_user      Cosine of emergent angle
   * @param direct_beam_mu     Cosine of incident angle
   * @param tau_user     User optical depth for this output level
   * @param direct_beam_flux    Incident beam irradiance
   * @return Single-scattered intensity contribution
   */
  double singleScat(double dither, int layru, int num_layers,
            const std::vector<double>& phase_function,
            const std::vector<double>& omega,
            const std::vector<double>& tau,
            double mu_user, double direct_beam_mu, double tau_user, double direct_beam_flux) const;

  /**
   * @brief Secondary scatter integral for IMS correction
   *
   * Converts c_secondary_scat() from original code.
   * Computes eq. STWL (A7) using unscaled optical depths.
   *
   * @param iu     User angle index (1-based)
   * @param lu     User level index (1-based)
   * @param ctheta Cosine of scattering angle
   * @param layru  Layer index for user level lu (1-based)
   * @return IMS correction term duims
   */
  double secondaryScat(int iu, int lu, double ctheta, int layru) const;

  /**
   * @brief Xi function for IMS correction (eq. STWL 72)
   *
   * Converts c_xi_func() from original code.
   *
   * @param umu1  Cosine of angle 1
   * @param umu2  Cosine of angle 2 (= umu3 in original)
   * @param tau   Optical thickness
   * @return Xi function value
   */
  double xiFunc(double umu1, double umu2, double tau) const;

  // ========================================================================
  // New Intensity Correction (Buras-Emde TMS/IMS, intensity_corr_nakajima=false)
  // ========================================================================

  /**
   * @brief Apply Buras-Emde intensity correction (TMS + IMS, new algorithm)
   *
   * Converts c_new_intensity_correction() from original code.
   * Like the old correction, applies TMS + IMS, but uses a tabulated
   * phase function (config.phase_function/mu_phase_function) instead of Legendre expansion.
   * Only called when corint=true AND intensity_corr_nakajima=false.
   *
   * @param result Output result structure (intensity modified in place)
   */
  void applyNewIntensityCorrection(DisortResult& result);

  /**
   * @brief Apply analytic BRDF intensity correction
   *
   * Port of INTCOR_BEAM_REFLEC from Fortran DISORT V3 (DISORT.f:2461-2577).
   * Corrects reflected beam intensity at user angles by comparing the exact (analytic)
   * BRDF with the Fourier-reconstructed approximation.
   *
   * @param result Output result structure (intensity modified in place)
   */
  void applyBrdfIntensityCorrection(DisortResult& result);

  /**
   * @brief Prepare integration grids for new secondary-scatter integral
   *
   * Converts prep_double_scat_integr() from original code.
   * Computes equidistant-in-abs-phas2 mu grid (mu_eq), sign array (neg_phas),
   * and normalization (norm_phas) used by calcPhaseSquared().
   *
   * @param num_phase_func_angles    Number of phase function tabulation points
   * @param nf        Number of integration grid points (100)
   * @param mu_phase_function  Cosine angle grid [num_phase_func_angles] (1-based)
   * @param phas2     Residual phase function [num_phase_func_angles × num_user_tau] (1-based)
   * @param mu_eq     [out] Integration angle grid [nf × num_user_tau] (1-based)
   * @param neg_phas  [out] Sign indicator [nf × num_user_tau] (1-based)
   * @param norm_phas [out] Normalization factor [num_user_tau] (1-based)
   */
  void prepDoubleScatIntegr(int num_phase_func_angles, int nf,
                const std::vector<double>& mu_phase_function,
                const std::vector<double>& phas2,
                std::vector<double>& mu_eq,
                std::vector<int>& neg_phas,
                std::vector<double>& norm_phas) const;

  /**
   * @brief New secondary scatter integral (Buras-Emde IMS correction)
   *
   * Converts c_new_secondary_scat() from original code.
   * Computes pspike = 2*p" - p"*p" for the IMS correction.
   *
   * @param iu        User angle index (1-based)
   * @param lu        User level index (1-based)
   * @param it        Phase function index for ctheta (1-based, from locate+1)
   * @param ctheta    Cosine of scattering angle
   * @param layru     Layer index for user level lu (1-based)
   * @param nf        Number of integration grid points
   * @param num_phase_func_angles    Number of phase function tabulation points
   * @param phas2     Residual phase function [num_phase_func_angles × num_user_tau]
   * @param mu_eq     Integration angle grid [nf × num_user_tau]
   * @param neg_phas  Sign indicator [nf × num_user_tau]
   * @param norm_phas Normalization factor for level lu
   * @return New IMS correction term
   */
  double newSecondaryScat(int iu, int lu, int it, double ctheta, int layru,
              int nf, int num_phase_func_angles,
              const std::vector<double>& phas2,
              const std::vector<double>& mu_eq,
              const std::vector<int>& neg_phas,
              double norm_phas) const;

  /**
   * @brief Compute convolution integral p" * p" (Buras-Emde IMS)
   *
   * Converts calc_phase_squared() from original code.
   * Computes the azimuthal integral of p"(mu_1) * p"(theta_12) over mu_1.
   *
   * @param num_phase_func_angles    Number of phase function tabulation points
   * @param lu        User level index (1-based)
   * @param ctheta    Cosine of primary scattering angle
   * @param nf        Number of integration grid points
   * @param mu_phase_function  Cosine angle grid [num_phase_func_angles]
   * @param phas2     Residual phase function [num_phase_func_angles × num_user_tau]
   * @param mu_eq     Integration angle grid [nf × num_user_tau]
   * @param neg_phas  Sign indicator [nf × num_user_tau]
   * @param norm_phas Normalization factor
   * @return p" * p" convolution value
   */
  double calcPhaseSquared(int num_phase_func_angles, int lu, double ctheta, int nf,
              const std::vector<double>& mu_phase_function,
              const std::vector<double>& phas2,
              const std::vector<double>& mu_eq,
              const std::vector<int>& neg_phas,
              double norm_phas) const;

  /**
   * @brief Allocate all working arrays based on problem dimensions
   */
  void allocateWorkingArrays();

  // ========================================================================
  // SPECIAL_BC: Medium Albedo and Transmissivity
  // ========================================================================

  /**
   * @brief Compute medium surface_albedo and transmissivity (SPECIAL_BC mode)
   *
   * Converts c_albtrans() from original code.
   * Uses the reciprocity principle to compute surface_albedo(theta) and
   * transmissivity(theta) for all user incident beam angles at once.
   *
   * @param config Input configuration (modified: isotropic_flux_top=1, use_lambertian_surface=true)
   * @param result Output result (albedo_medium, transmissivity_medium filled)
   */
  void computeAlbTrans(DisortConfig& config, DisortResult& result);

  /**
   * @brief Solve for integration constants under isotropic illumination
   *
   * Converts c_solve1() from original code.
   * Simpler than solve0(): just sets isotropic_flux_top in top or bottom boundary,
   * then solves the already-LU-factored band_matrix_ matrix.
   *
   * @param ihom  0 = TOP_ILLUM (isotropic from top), 1 = BOT_ILLUM (from bottom)
   * @param ncol  Number of columns (= num_streams * ncut)
   *
   * Uses band_matrix_ (must be LU-factored by sgbfa) and band_pivot_ already set.
   */
  void solve1(int ihom, int ncol);

  /**
   * @brief Compute azimuthally-averaged intensity for SPECIAL_BC
   *
   * Converts c_albtrans_intensity() from original code.
   * Integrates homogeneous solution through layers using current integration_constants_.
   * Does NOT modify eigenvec_user_ in-place (so it can be called twice with different integration_constants_).
   *
   * @param u0u_out [out] Intensity matrix (numu_ rows × 2 cols)
   *                 col 0: upward intensity at TOP (lu=1 in C)
   *                 col 1: downward intensity at BOTTOM (lu=2 in C)
   */
  void albTransIntensity(Eigen::MatrixXd& u0u_out);

  /**
   * @brief Compute spherical surface_albedo and transmissivity
   *
   * Converts c_albtrans_spherical() from original code.
   * Uses current integration_constants_ and eigenvectors_ to integrate fluxes at top and bottom boundaries.
   *
   * @param sflup [out] Upward flux at top (factor 2 included)
   * @param sfldn [out] Downward flux at bottom (factor 2 included)
   */
  void albTransSpherical(double& sflup, double& sfldn);
};

} // namespace disortpp
