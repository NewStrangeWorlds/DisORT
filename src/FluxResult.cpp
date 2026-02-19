#include "FluxResult.hpp"

namespace disortpp {

void FluxResult::allocate(int num_layers) 
{
  const int n = num_layers + 1;
  flux_direct_beam.assign(n, 0.0);
  flux_down.assign(n, 0.0);
  flux_up.assign(n, 0.0);
  flux_tau_divergence.assign(n, 0.0);
  mean_intensity.assign(n, 0.0);
  mean_intensity_down.assign(n, 0.0);
  mean_intensity_up.assign(n, 0.0);
  mean_intensity_direct_beam.assign(n, 0.0);
}

} // namespace disortpp
