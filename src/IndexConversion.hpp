#pragma once

#include "DisortConfig.hpp"
#include "DisortFluxConfig.hpp"
#include "DisortResult.hpp"
#include "FluxResult.hpp"
#include <algorithm>
#include <numeric>

namespace disortpp {

// ============================================================================
// Input array reversal (bottom-up -> top-down for internal solver)
// ============================================================================

inline void reverseInputArrays(DisortConfig& config) {
  // Layer arrays (size num_layers)
  std::reverse(config.delta_tau.begin(), config.delta_tau.end());
  std::reverse(config.single_scat_albedo.begin(), config.single_scat_albedo.end());
  std::reverse(config.phase_function_moments.begin(), config.phase_function_moments.end());

  if (!config.phase_function.empty())
    std::reverse(config.phase_function.begin(), config.phase_function.end());

  // Level arrays (size num_layers+1)
  if (!config.temperature.empty())
    std::reverse(config.temperature.begin(), config.temperature.end());
  if (!config.level_altitudes.empty())
    std::reverse(config.level_altitudes.begin(), config.level_altitudes.end());

  // tau_user: transform from bottom-relative to top-relative optical depths
  if (config.flags.use_user_tau && !config.tau_user.empty()) {
    double total_tau = std::accumulate(config.delta_tau.begin(), config.delta_tau.end(), 0.0);
    for (auto& tau : config.tau_user)
      tau = total_tau - tau;
    std::reverse(config.tau_user.begin(), config.tau_user.end());
  }
}

inline void reverseInputArrays(DisortFluxConfig& config) {
  // Layer arrays (size num_layers)
  std::reverse(config.delta_tau.begin(), config.delta_tau.end());
  std::reverse(config.single_scat_albedo.begin(), config.single_scat_albedo.end());
  std::reverse(config.phase_function_moments.begin(), config.phase_function_moments.end());

  // Level arrays (size num_layers+1)
  if (!config.temperature.empty())
    std::reverse(config.temperature.begin(), config.temperature.end());
  if (!config.level_altitudes.empty())
    std::reverse(config.level_altitudes.begin(), config.level_altitudes.end());
}

// ============================================================================
// Output array reversal (top-down -> bottom-up for user)
// ============================================================================

inline void reverseOutputArrays(DisortResult& result) {
  // 1D flux/intensity arrays (size num_user_tau)
  std::reverse(result.flux_direct_beam.begin(), result.flux_direct_beam.end());
  std::reverse(result.flux_down.begin(), result.flux_down.end());
  std::reverse(result.flux_up.begin(), result.flux_up.end());
  std::reverse(result.flux_tau_divergence.begin(), result.flux_tau_divergence.end());
  std::reverse(result.mean_intensity.begin(), result.mean_intensity.end());
  std::reverse(result.mean_intensity_down.begin(), result.mean_intensity_down.end());
  std::reverse(result.mean_intensity_up.begin(), result.mean_intensity_up.end());
  std::reverse(result.mean_intensity_direct_beam.begin(), result.mean_intensity_direct_beam.end());

  // 3D intensity[lu][iu][j] — reverse outer (lu) dimension
  std::reverse(result.intensity.begin(), result.intensity.end());

  // 2D intensity_azimuth_avg[lu][iu] — reverse outer (lu) dimension
  std::reverse(result.intensity_azimuth_avg.begin(), result.intensity_azimuth_avg.end());

  // 3D intensity_fourier_expansion[lu][iu][k] — reverse outer (lu) dimension
  std::reverse(result.intensity_fourier_expansion.begin(), result.intensity_fourier_expansion.end());
}

inline void reverseOutputArrays(FluxResult& result) {
  std::reverse(result.flux_direct_beam.begin(), result.flux_direct_beam.end());
  std::reverse(result.flux_down.begin(), result.flux_down.end());
  std::reverse(result.flux_up.begin(), result.flux_up.end());
  std::reverse(result.flux_tau_divergence.begin(), result.flux_tau_divergence.end());
  std::reverse(result.mean_intensity.begin(), result.mean_intensity.end());
  std::reverse(result.mean_intensity_down.begin(), result.mean_intensity_down.end());
  std::reverse(result.mean_intensity_up.begin(), result.mean_intensity_up.end());
  std::reverse(result.mean_intensity_direct_beam.begin(), result.mean_intensity_direct_beam.end());
}

} // namespace disortpp
