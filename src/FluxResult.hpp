#pragma once

#include <vector>

namespace disortpp {

/**
 * @brief Lightweight flux-only result from DisortFluxSolver
 *
 * Contains fluxes and mean intensities at all layer boundaries.
 * Output levels: index 0 = TOA, index num_layers = BOA.
 * Size of all vectors is num_layers + 1.
 */
struct FluxResult {
  std::vector<double> flux_direct_beam;           ///< Direct-beam flux [num_layers+1]
  std::vector<double> flux_down;                   ///< Diffuse downward flux [num_layers+1]
  std::vector<double> flux_up;                     ///< Diffuse upward flux [num_layers+1]
  std::vector<double> flux_tau_divergence;         ///< Flux divergence d(net flux)/d(optical depth) [num_layers+1]
  std::vector<double> mean_intensity;              ///< Mean intensity including direct beam [num_layers+1]
  std::vector<double> mean_intensity_down;         ///< Mean diffuse downward intensity [num_layers+1]
  std::vector<double> mean_intensity_up;           ///< Mean diffuse upward intensity [num_layers+1]
  std::vector<double> mean_intensity_direct_beam;  ///< Mean direct beam intensity [num_layers+1]

  /// Number of output levels
  int num_levels() const { return static_cast<int>(flux_up.size()); }

  /**
   * @brief Allocate all vectors for num_layers layers
   * @param num_layers Number of atmospheric layers (output has num_layers+1 levels)
   */
  void allocate(int num_layers);

  /// Total downward flux at level lev (direct beam + diffuse)
  double totalFluxDown(int lev) const { return flux_direct_beam[lev] + flux_down[lev]; }

  /// Net downward flux at level lev (positive = net downward)
  double netFlux(int lev) const { return flux_direct_beam[lev] + flux_down[lev] - flux_up[lev]; }
};

} // namespace disortpp
