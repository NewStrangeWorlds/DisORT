#pragma once

#include "DisortFluxConfig.hpp"
#include "FluxResult.hpp"
#include "Legendre.hpp"
#include "Utils.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>

namespace disortpp {

/**
 * @brief Templated flux-only DISORT solver with compile-time stream count
 *
 * A performance-optimized variant of DisortSolver that:
 * - Only computes fluxes and mean intensities (no angular intensities)
 * - Uses compile-time NStr for Eigen fixed-size matrices (stack-allocated, SIMD)
 * - Outputs at all layer boundaries (no user-specified optical depths)
 * - Supports Lambertian surface only (no BRDF)
 * - Supports plane-parallel and spherical beam geometry
 * - Supports thermal emission
 *
 * For NStr=4-16, the fixed-size eigenvalue solver is 30-50% faster than dynamic.
 *
 * @tparam NStr Number of streams (must be even, >= 2)
 */
template<int NStr>
class DisortFluxSolver {
  static_assert(NStr >= 4, "NStr must be >= 4 (use a dedicated two-stream solver for NStr=2)");
  static_assert(NStr % 2 == 0, "NStr must be even");

public:
  static constexpr int NN = NStr / 2;

  /**
   * @brief Construct solver (lightweight, no allocation)
   */
  DisortFluxSolver() = default;

  /**
   * @brief Solve for fluxes and mean intensities
   *
   * @param config Input configuration. Must have num_streams == NStr.
   *               Only flux-relevant fields are used; intensity-related fields are ignored.
   * @return FluxResult with fluxes at all num_layers+1 layer boundaries
   * @throws std::invalid_argument if config.num_streams != NStr or other validation failures
   */
  FluxResult solve(const DisortFluxConfig& config);

private:
  // Fixed-size Eigen types
  using HalfMat = Eigen::Matrix<double, NN, NN>;
  using FullMat = Eigen::Matrix<double, NStr, NStr>;
  using FullVec = Eigen::Matrix<double, NStr, 1>;
  using HalfVec = Eigen::Matrix<double, NN, 1>;
  using LegMat  = Eigen::Matrix<double, NStr + 1, NStr, Eigen::RowMajor>;  // Legendre at quad angles (row-major for legendrePolynomialsFlat)
  using LegVec  = Eigen::Matrix<double, NStr + 1, 1>;     // Legendre at beam angle

  // ========================================================================
  // Internal Methods
  // ========================================================================

  void disortSet(const DisortFluxConfig& config, bool deltam);
  void gaussianQuadrature(int n);
  double chapmanFunction(int lc, double tau_frac);
  void solveEigen(int lc);
  void computeBeamSource();
  void computeBeamSourceSpherical();
  void computeIsotropicSource();
  void setMatrix();
  void solve0();
  void computeFluxes(FluxResult& result);
  void allocateWorkingArrays();

  // ========================================================================
  // Configuration (not owned)
  // ========================================================================
  const DisortFluxConfig* config_ = nullptr;

  // ========================================================================
  // Dimensions (runtime, depends on num_layers)
  // ========================================================================
  int nlyr_ = 0;
  int ncut_ = 0;
  bool deltam_ = false;
  bool deltam_plus_ = false;
  bool lyrcut_ = false;
  bool scat_yes_ = false;

  // ========================================================================
  // Fixed-Size Working Arrays (stack-allocated via Eigen fixed-size)
  // ========================================================================

  // Quadrature angles and weights
  FullVec quad_angle_;
  FullVec quad_weight_;

  // Legendre polynomials
  LegMat legendre_quad_;   // P_l(quad_angle_[jq]), l=0..NStr, jq=0..NStr-1
  LegVec legendre_beam_;   // P_l(-direct_beam_mu), l=0..NStr

  // Eigenvalue problem scratch
  HalfMat alpha_minus_beta_;
  HalfMat alpha_plus_beta_;
  HalfMat eigen_product_;
  Eigen::SelfAdjointEigenSolver<HalfMat> sym_eig_solver_;
  Eigen::LLT<HalfMat> cholesky_;
  HalfVec similarity_d_;       // D = diag(sqrt(cwt * mu)) for symmetrization
  HalfVec similarity_d_inv_;   // D^{-1}

  // Beam source scratch
  FullMat beam_A_;
  FullVec beam_b_;

  // Thermal source scratch
  FullMat isot_A_;
  FullVec isot_b_, isot_z1_;

  // Spherical beam scratch
  FullMat spher_A_;
  FullVec spher_zj_, spher_wk_;

  // ========================================================================
  // Dynamic Arrays (size depends on nlyr, allocated once per solve)
  // ========================================================================

  // Delta-M scaled optical properties
  std::vector<double> scaled_dtau_;           // [nlyr]
  std::vector<double> scaled_tau_cumulative_; // [nlyr+1]
  std::vector<double> scaled_ssa_;            // [nlyr]
  std::vector<double> forward_scatter_frac_;  // [nlyr]
  Eigen::Matrix<double, NStr, Eigen::Dynamic> scaled_phase_coeffs_;  // NStr x nlyr

  // Beam attenuation
  std::vector<double> chapman_;             // [nlyr]
  std::vector<double> chapman_tau_;         // [2*nlyr+2]
  std::vector<double> beam_transmission_;   // [nlyr+1]

  // Per-layer eigensystem (dynamic outer vector, fixed-size inner)
  std::vector<FullVec> eigenvalues_;       // [nlyr] each NStr
  std::vector<FullMat> eigenvectors_;      // [nlyr] each NStr x NStr
  std::vector<FullMat> phase_matrix_;      // [nlyr] each NStr x NStr

  // Particular solutions
  std::vector<FullVec> beam_particular_;   // [nlyr] each NStr
  std::vector<FullVec> thermal_z0_;        // [nlyr] each NStr
  std::vector<FullVec> thermal_z1_;        // [nlyr] each NStr

  // Spherical beam particular solutions
  std::vector<FullVec> beam_particular_slope_;  // [nlyr] each NStr
  std::vector<double>  beam_particular_atten_;  // [nlyr] scalar per layer

  // Thermal expansion coefficients
  std::vector<double> planck_intercept_;   // [nlyr]
  std::vector<double> planck_slope_;       // [nlyr]
  double planck_bottom_ = 0.0;
  double planck_top_ = 0.0;

  // Band matrix and solution (depends on nlyr)
  Eigen::MatrixXd band_matrix_;            // (9*NN-2) x (NStr*nlyr)
  std::vector<int> band_pivot_;            // NStr*nlyr
  Eigen::VectorXd integration_constants_;  // NStr*nlyr
  Eigen::VectorXd solve0_b_;              // NStr*nlyr

  // Surface reflectivity at computational angles (Lambertian)
  Eigen::Matrix<double, NN, Eigen::Dynamic> surface_refl_quad_;  // NN x (NN+1)
  HalfVec surface_emis_quad_;

  // Flux computation scratch
  Eigen::Matrix<double, NStr, Eigen::Dynamic> intensity_quad_;  // NStr x (nlyr+1)
  FullVec work_vec_;

  // Legendre computation scratch (raw arrays for legendrePolynomialsFlat)
  std::array<double, NStr> cmu_vec_;
  std::array<double, 1> umu0_vec_;

  // ========================================================================
  // Allocation Cache
  // ========================================================================
  int alloc_nlyr_ = -1;
  FluxResult result_;
};

} // namespace disortpp
