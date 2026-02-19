#include "DisortFluxConfig.hpp"
#include "PhaseFunction.hpp"
#include <stdexcept>
#include <string>
#include <cmath>

namespace disortpp {

DisortFluxConfig::DisortFluxConfig(int nlyr, int nstr, int nmom)
  : num_layers(nlyr), num_streams(nstr), num_phase_func_moments(nmom)
{
  if (num_layers <= 0) {
    throw std::invalid_argument("num_layers must be > 0");
  }
  if (num_streams < 4 || num_streams % 2 != 0) {
    throw std::invalid_argument("num_streams must be even and >= 4");
  }
  if (num_phase_func_moments < 0) {
    throw std::invalid_argument("num_phase_func_moments must be >= 0");
  }
}

void DisortFluxConfig::allocate()
{
  delta_tau.resize(num_layers, 0.0);
  single_scat_albedo.resize(num_layers, 0.0);

  const int nmom_size = nmomNstr() + 1;
  phase_function_moments.resize(num_layers);
  for (int lc = 0; lc < num_layers; ++lc) {
    phase_function_moments[lc].resize(nmom_size, 0.0);
  }

  if (use_thermal_emission) {
    temperature.resize(num_layers + 1, 0.0);
  }

  if (use_spherical_beam) {
    level_altitudes.resize(num_layers + 1, 0.0);
  }
}

void DisortFluxConfig::validate() const 
{
  // Dimensions
  if (num_layers <= 0) {
    throw std::invalid_argument("num_layers must be > 0");
  }
  if (num_streams < 4 || num_streams % 2 != 0) {
    throw std::invalid_argument("num_streams must be even and >= 4");
  }
  if (num_phase_func_moments < 0) {
    throw std::invalid_argument("num_phase_func_moments must be >= 0");
  }

  // Array sizes
  if (static_cast<int>(delta_tau.size()) < num_layers) {
    throw std::invalid_argument("delta_tau size (" + std::to_string(delta_tau.size())
      + ") < num_layers (" + std::to_string(num_layers) + ")");
  }
  if (static_cast<int>(single_scat_albedo.size()) < num_layers) {
    throw std::invalid_argument("single_scat_albedo size < num_layers");
  }
  if (static_cast<int>(phase_function_moments.size()) < num_layers) {
    throw std::invalid_argument("phase_function_moments size < num_layers");
  }

  const int nmom_needed = nmomNstr() + 1;
  for (int lc = 0; lc < num_layers; ++lc) {
    if (static_cast<int>(phase_function_moments[lc].size()) < nmom_needed) {
      throw std::invalid_argument("phase_function_moments[" + std::to_string(lc)
        + "] size < nmomNstr()+1");
    }
  }

  // Physical validity per layer
  for (int lc = 0; lc < num_layers; ++lc) {
    if (delta_tau[lc] < 0.0) {
      throw std::invalid_argument("delta_tau[" + std::to_string(lc) + "] must be >= 0");
    }
    if (single_scat_albedo[lc] < 0.0 || single_scat_albedo[lc] > 1.0) {
      throw std::invalid_argument("single_scat_albedo[" + std::to_string(lc) + "] must be in [0, 1]");
    }
  }

  // Boundary conditions
  if (direct_beam_flux < 0.0) {
    throw std::invalid_argument("direct_beam_flux must be >= 0");
  }
  if (direct_beam_flux > 0.0 && (direct_beam_mu <= 0.0 || direct_beam_mu > 1.0)) {
    throw std::invalid_argument("direct_beam_mu must be in (0, 1] when direct_beam_flux > 0");
  }

  if (surface_albedo < 0.0 || surface_albedo > 1.0) {
    throw std::invalid_argument("surface_albedo must be in [0, 1]");
  }

  // Thermal emission
  if (use_thermal_emission) {
    if (static_cast<int>(temperature.size()) < num_layers + 1) {
      throw std::invalid_argument("temperature size < num_layers + 1");
    }
    for (int lev = 0; lev <= num_layers; ++lev) {
      if (temperature[lev] < 0.0) {
        throw std::invalid_argument("temperature[" + std::to_string(lev) + "] must be >= 0");
      }
    }
    if (wavenumber_low < 0.0 || wavenumber_low >= wavenumber_high) {
      throw std::invalid_argument("wavenumber: need 0 <= wavenumber_low < wavenumber_high");
    }
    if (temperature_top < 0.0 || temperature_bottom < 0.0) {
      throw std::invalid_argument("temperature_top and temperature_bottom must be >= 0");
    }
    if (emissivity_top < 0.0 || emissivity_top > 1.0) {
      throw std::invalid_argument("emissivity_top must be in [0, 1]");
    }
  }

  // Spherical geometry
  if (use_spherical_beam) {
    if (bottom_radius <= 0.0) {
      throw std::invalid_argument("bottom_radius must be > 0 when use_spherical_beam is true");
    }
    if (static_cast<int>(level_altitudes.size()) < num_layers + 1) {
      throw std::invalid_argument("level_altitudes size < num_layers + 1");
    }
  }

  // Delta-M+
  if (use_delta_m_plus) {
    if (num_phase_func_moments < num_streams + 1) {
      throw std::invalid_argument(
        "use_delta_m_plus requires num_phase_func_moments >= num_streams + 1");
    }
  }
}

// Phase function convenience methods
void DisortFluxConfig::setHenyeyGreenstein(double g, int lc) 
{
  phase_function::fillHenyeyGreenstein(phase_function_moments, nmomNstr(), num_layers, g, lc);
}

void DisortFluxConfig::setIsotropic(int lc) 
{
  phase_function::fillIsotropic(phase_function_moments, nmomNstr(), num_layers, lc);
}

void DisortFluxConfig::setRayleigh(int lc) {
  phase_function::fillRayleigh(phase_function_moments, nmomNstr(), num_layers, lc);
}

void DisortFluxConfig::setHazeGarciaSiewert(int lc) 
{
  phase_function::fillHazeGarciaSiewert(phase_function_moments, nmomNstr(), num_layers, lc);
}

void DisortFluxConfig::setCloudGarciaSiewert(int lc) {
  phase_function::fillCloudGarciaSiewert(phase_function_moments, nmomNstr(), num_layers, lc);
}

void DisortFluxConfig::setPhaseFunction(PhaseFunction type, double g, int lc) 
{
  phase_function::fillPhaseFunction(phase_function_moments, nmomNstr(), num_layers, type, g, lc);
}

} // namespace disortpp
