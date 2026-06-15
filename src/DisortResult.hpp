#pragma once

#include <vector>

namespace disortpp {

/**
 * @brief DISORT output/result container
 *
 * Corresponds to disort_output in original C code.
 * Contains all computed radiant quantities and intensities.
 *
 * Memory is managed automatically via RAII.
 * Arrays use 0-based indexing via std::vector.
 */
class DisortResult {
public:
  // Radiant quantities at each output level [num_user_tau] (0-based: 0..num_user_tau-1)
  std::vector<double> flux_direct_beam;            ///< Direct-beam flux (without delta-M scaling)
  std::vector<double> flux_down;                   ///< Diffuse downward flux (without delta-M scaling)
  std::vector<double> flux_up;                     ///< Diffuse upward flux
  std::vector<double> flux_tau_divergence;         ///< Flux divergence d(net flux)/d(optical depth)
  std::vector<double> mean_intensity;              ///< Mean intensity including direct beam
  std::vector<double> mean_intensity_down;         ///< Mean diffuse downward intensity
  std::vector<double> mean_intensity_up;           ///< Mean diffuse upward intensity
  std::vector<double> mean_intensity_direct_beam;  ///< Mean direct beam intensity

  // Albedo and transmissivity (only for SPECIAL_BC mode, 0-based)
  std::vector<double> albedo_medium;  ///< Albedo of medium [num_user_mu] (ibcnd = SPECIAL_BC only)
  std::vector<double> transmissivity_medium;  ///< Transmissivity of medium [num_user_mu] (ibcnd = SPECIAL_BC only)
  double spherical_albedo = 0.0; ///< Spherical albedo (ibcnd = SPECIAL_BC only)
  double spherical_transmissivity = 0.0; ///< Spherical transmissivity (ibcnd = SPECIAL_BC only)

  // Polar angle cosines at which intensities are returned [num_user_mu]
  // When use_user_mu=true, contains the user-specified mu values.
  // When use_user_mu=false, contains the computational quadrature angles.
  std::vector<double> mu_angles;

  // Intensities (if comp_only_fluxes = false, 0-based)
  std::vector<std::vector<std::vector<double>>> intensity;   ///< Intensity [num_user_tau][num_user_mu][num_phi]
  std::vector<std::vector<double>> intensity_azimuth_avg;               ///< Azimuthally-averaged intensity [num_user_tau][num_user_mu]

  // Intensity expansion coefficients (if output_fourier_expansion = true, 0-based)
  std::vector<std::vector<std::vector<double>>> intensity_fourier_expansion;  ///< Intensity [num_user_tau][num_user_mu][num_streams+1]

  // ------------------------------------------------------------------------
  // Temperature Jacobians (only filled when flags.compute_temperature_jacobian
  // and flags.use_thermal_emission). Each is [num_user_tau][num_layers+2]:
  //   column k = 0..num_layers : d(output)/d(temperature at level k)   [K^-1 * units]
  //   column num_layers+1      : d(output)/d(bc.temperature_bottom)    (surface)
  // Empty when the Jacobian is not requested. Ordering follows flags.index_from_bottom:
  // by default the level (row) and level-temperature (column) axes run top-of-atmosphere
  // -> bottom; when index_from_bottom is set both are reversed to match the user's
  // bottom-up indexing (the surface column always stays last).
  // ------------------------------------------------------------------------
  std::vector<std::vector<double>> flux_up_temperature_jac;          ///< d(flux_up)/dT
  std::vector<std::vector<double>> flux_down_temperature_jac;        ///< d(flux_down)/dT
  std::vector<std::vector<double>> mean_intensity_temperature_jac;       ///< d(mean_intensity)/dT
  std::vector<std::vector<double>> mean_intensity_down_temperature_jac;  ///< d(mean_intensity_down)/dT
  std::vector<std::vector<double>> mean_intensity_up_temperature_jac;    ///< d(mean_intensity_up)/dT
  std::vector<std::vector<double>> flux_divergence_temperature_jac;  ///< d(flux_tau_divergence)/dT (heating-rate Jacobian)

  /**
   * @brief Default constructor
   */
  DisortResult() = default;

  /// Number of output levels, user angles, and azimuthal angles (set by allocate())
  int num_user_tau() const { return ntau_; }
  int num_user_mu() const { return numu_; }
  int num_phi() const { return nphi_; }

  /**
   * @brief Allocate arrays based on problem dimensions
   * @param num_user_tau Number of output optical depth levels
   * @param num_user_mu Number of user polar angles
   * @param num_phi Number of azimuthal angles
   * @param num_streams Number of streams
   * @param special_bc True if using SPECIAL_BC mode
   * @param compute_intensity True if computing intensities (comp_only_fluxes=false)
   * @param output_fourier_expansion True if outputting intensity_fourier_expansion array
   *
   * This replaces c_disort_out_alloc() from the C code.
   */
  void allocate(int num_user_tau, int num_user_mu, int num_phi, int num_streams,
          bool special_bc, bool compute_intensity, bool output_fourier_expansion);

  /**
   * @brief Access 3D intensity array with 0-based indexing
   * @param iu User angle index (0-based: 0..num_user_mu-1)
   * @param lu Level index (0-based: 0..num_user_tau-1)
   * @param j Azimuthal index (0-based: 0..num_phi-1)
   */
  double& intensities(int iu, int lu, int j) {
    return intensity[lu][iu][j];
  }

  const double& intensities(int iu, int lu, int j) const {
    return intensity[lu][iu][j];
  }

  /**
   * @brief Access 2D intensity_azimuth_avg array with 0-based indexing
   * @param iu User angle index (0-based: 0..num_user_mu-1)
   * @param lu Level index (0-based: 0..num_user_tau-1)
   */
  double& intensitiesAzimuthAvg(int iu, int lu) {
    return intensity_azimuth_avg[lu][iu];
  }

  const double& intensitiesAzimuthAvg(int iu, int lu) const {
    return intensity_azimuth_avg[lu][iu];
  }

  /// Total downward flux at output level lu (direct beam + diffuse)
  double totalFluxDown(int lu) const { return flux_direct_beam[lu] + flux_down[lu]; }

  /// Net downward flux at output level lu (positive = net downward)
  double netFlux(int lu) const { return flux_direct_beam[lu] + flux_down[lu] - flux_up[lu]; }

private:
  int ntau_ = 0;
  int numu_ = 0;
  int nphi_ = 0;
};

} // namespace disortpp
