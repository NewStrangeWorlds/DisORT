#include "DisortResult.hpp"
#include <stdexcept>

namespace disortpp {

void DisortResult::allocate(int num_user_tau, int num_user_mu, int num_phi, int num_streams,
              bool special_bc, bool compute_intensity, bool output_uum_flag) 
{
  ntau_ = num_user_tau;
  numu_ = num_user_mu;
  nphi_ = num_phi;

  // Radiant quantities at each output level (always allocated, 0-based)
  flux_direct_beam.assign(num_user_tau, 0.0);
  flux_down.assign(num_user_tau, 0.0);
  flux_up.assign(num_user_tau, 0.0);
  flux_tau_divergence.assign(num_user_tau, 0.0);
  mean_intensity.assign(num_user_tau, 0.0);
  mean_intensity_down.assign(num_user_tau, 0.0);
  mean_intensity_up.assign(num_user_tau, 0.0);
  mean_intensity_direct_beam.assign(num_user_tau, 0.0);

  // Polar angle cosines at which intensities are returned
  mu_angles.assign(num_user_mu, 0.0);

  // Albedo and transmissivity (only for SPECIAL_BC mode)
  if (special_bc) {
    albedo_medium.resize(num_user_mu);
    transmissivity_medium.resize(num_user_mu);
  }

  // Intensities (if not comp_only_fluxes)
  if (compute_intensity) {
    // intensity: 3D array [num_user_tau][num_user_mu][num_phi]
    intensity.assign(num_user_tau, std::vector<std::vector<double>>(
      num_user_mu, std::vector<double>(num_phi, 0.0)));

    // intensity_azimuth_avg: 2D array [num_user_tau][num_user_mu]
    intensity_azimuth_avg.assign(num_user_tau, std::vector<double>(num_user_mu, 0.0));

    // intensity_fourier_expansion: optional 3D array [num_user_tau][num_user_mu][num_streams+1]
    if (output_uum_flag) {
      intensity_fourier_expansion.assign(num_user_tau, std::vector<std::vector<double>>(
        num_user_mu, std::vector<double>(num_streams + 1, 0.0)));
    }
  }
}

} // namespace disortpp
