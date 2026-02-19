#pragma once

#include "DisortTypes.hpp"
#include <algorithm>
#include <vector>

namespace disortpp {

/**
 * @brief Lean configuration for DisortFluxSolver<NStr>
 *
 * Contains only the fields needed by the flux-only solver.
 * All fields are flat (no nested flags/bc structs).
 */
class DisortFluxConfig {
public:
  // ======== Dimensions ========
  int num_layers = 0;
  int num_streams = 0;
  int num_phase_func_moments = 0;

  /// Derived: max(num_phase_func_moments, num_streams)
  int nmomNstr() const { return std::max(num_phase_func_moments, num_streams); }

  // ======== Flags ========
  bool use_thermal_emission = false;
  bool use_spherical_beam = false;
  bool use_delta_m_plus = false;

  // ======== Boundary conditions ========
  double direct_beam_flux = 0.0;
  double direct_beam_mu = 0.5;
  double isotropic_flux_top = 0.0;
  double isotropic_flux_bottom = 0.0;
  double temperature_top = 0.0;
  double temperature_bottom = 0.0;
  double emissivity_top = 1.0;
  double surface_albedo = 0.0;

  // ======== Physical parameters ========
  double wavenumber_low = 0.0;
  double wavenumber_high = 0.0;
  double bottom_radius = 0.0;

  // ======== Layer/Level arrays ========
  std::vector<double> delta_tau;                           ///< Optical depth per layer [num_layers]
  std::vector<double> single_scat_albedo;                  ///< Single-scatter albedo per layer [num_layers]
  std::vector<std::vector<double>> phase_function_moments; ///< Phase function moments [num_layers][nmomNstr()+1]
  std::vector<double> temperature;                         ///< Temperature at levels [num_layers+1] (thermal only)
  std::vector<double> level_altitudes;                     ///< Altitude at levels [num_layers+1] (spherical only)

  // ======== Construction ========
  DisortFluxConfig() = default;

  /**
   * @brief Construct with basic dimensions
   * @param nlyr Number of layers
   * @param nstr Number of streams (must be even, >= 4)
   * @param nmom Number of phase function moments (not including zeroth)
   */
  DisortFluxConfig(int nlyr, int nstr, int nmom = 0);

  /**
   * @brief Allocate arrays based on current dimensions and flags
   */
  void allocate();

  /**
   * @brief Validate configuration
   * @throws std::invalid_argument if configuration is invalid
   */
  void validate() const;

  // ======== Phase function moments ========

  double& phaseFunctionMoments(int k, int lc) {
    return phase_function_moments[lc][k];
  }

  const double& phaseFunctionMoments(int k, int lc) const {
    return phase_function_moments[lc][k];
  }

  void setHenyeyGreenstein(double g, int lc = -1);
  void setIsotropic(int lc = -1);
  void setRayleigh(int lc = -1);
  void setHazeGarciaSiewert(int lc = -1);
  void setCloudGarciaSiewert(int lc = -1);
  void setPhaseFunction(PhaseFunction type, double g = 0.0, int lc = -1);
};

} // namespace disortpp
