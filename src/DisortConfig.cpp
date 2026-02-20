#include "DisortConfig.hpp"
#include "PhaseFunction.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>

namespace disortpp {

DisortConfig::DisortConfig(int nlyr_, int nstr_, int nmom_)
  : num_layers(nlyr_), num_phase_func_moments(nmom_), num_streams(nstr_)
{
  // Validate basic dimensions
  if (num_layers <= 0) {
    throw std::invalid_argument("num_layers must be positive");
  }
  if (num_streams < 4 || num_streams % 2 != 0) {
    throw std::invalid_argument("num_streams must be even and >= 4");
  }
  if (num_phase_func_moments < 0) {
    throw std::invalid_argument("num_phase_func_moments must be non-negative");
  }
}

void DisortConfig::allocate() 
{
  // Allocate layer properties (always needed)
  delta_tau.resize(num_layers);
  single_scat_albedo.resize(num_layers);

  // phase_function_moments: [num_layers][nmomNstr()+1]
  phase_function_moments.resize(num_layers, std::vector<double>(nmomNstr() + 1, 0.0));

  // Temperature (only if thermal emission is enabled)
  if (flags.use_thermal_emission) {
    temperature.resize(num_layers + 1);
  }

  // User output specification
  // Always allocate tau_user - it's needed internally even if use_user_tau=false
  if (num_user_tau > 0) {
    tau_user.resize(num_user_tau);
  }

  // Always allocate mu_user - it's needed internally even if use_user_mu=false
  if (num_user_mu > 0) {
    mu_user.resize(num_user_mu);
  }

  if (!flags.comp_only_fluxes && num_phi > 0) {
    phi_user.resize(num_phi);
  }

  // Spherical geometry
  if (flags.use_spherical_beam) {
    if (bottom_radius <= 0.0) {
      throw std::invalid_argument("bottom_radius must be positive when use_spherical_beam=true");
    }
    level_altitudes.resize(num_layers + 1);
  }

  // Phase function (for new intensity correction)
  if (flags.intensity_corr_buras && !flags.intensity_corr_nakajima) {
    if (num_phase_func_angles > 0) {
      mu_phase_function.resize(num_phase_func_angles);
      phase_function.resize(num_layers, std::vector<double>(num_phase_func_angles, 0.0));
    }
  }

}

void DisortConfig::validate() const 
{
  // Dimension checks
  if (num_layers <= 0) {
    throw std::invalid_argument("num_layers must be positive");
  }

  if (num_streams < 4) {
    throw std::invalid_argument("num_streams must be >= 4");
  }

  if (num_streams % 2 != 0) {
    throw std::invalid_argument("num_streams must be even");
  }

  if (num_phase_func_moments < 0) {
    throw std::invalid_argument("num_phase_func_moments must be non-negative");
  }

  // Array size checks
  if (delta_tau.size() < static_cast<size_t>(num_layers)) {
    throw std::invalid_argument("delta_tau array is too small");
  }

  if (single_scat_albedo.size() < static_cast<size_t>(num_layers)) {
    throw std::invalid_argument("single_scat_albedo array is too small");
  }

  if (static_cast<int>(phase_function_moments.size()) < num_layers) {
    throw std::invalid_argument("phase_function_moments must have at least num_layers layers");
  }
  for (int lc = 0; lc < num_layers; ++lc) {
    if (static_cast<int>(phase_function_moments[lc].size()) < nmomNstr() + 1) {
      throw std::invalid_argument("phase_function_moments[" + std::to_string(lc)
        + "] must have at least nmomNstr()+1 elements");
    }
  }

  // Physical validity checks
  for (int lc = 0; lc < num_layers; ++lc) {
    if (delta_tau[lc] < 0.0) {
      throw std::invalid_argument("delta_tau[" + std::to_string(lc) + "] is negative");
    }

    if (single_scat_albedo[lc] < 0.0 || single_scat_albedo[lc] > 1.0) {
      throw std::invalid_argument("single_scat_albedo[" + std::to_string(lc) + "] must be in [0, 1]");
    }
  }

  // Temperature checks
  if (flags.use_thermal_emission) {
    if (temperature.size() < static_cast<size_t>(num_layers + 1)) {
      throw std::invalid_argument("temperature array is too small (need num_layers+1 when use_thermal_emission=true)");
    }

    for (int lev = 0; lev <= num_layers; ++lev) {
      if (temperature[lev] < 0.0) {
        throw std::invalid_argument("temperature[" + std::to_string(lev) + "] is negative");
      }
    }

    // Mirrors C DISORT check: wavenumber_low >= 0 and wavenumber_high >= wavenumber_low
    if (wavenumber_low < 0.0 || wavenumber_high < wavenumber_low) {
      throw std::invalid_argument("Invalid wavenumber range for Planck function");
    }
  }

  // Boundary condition checks
  if (bc.direct_beam_flux < 0.0) {
    throw std::invalid_argument("direct_beam_flux must be non-negative");
  }

  if (bc.direct_beam_mu <= 0.0 || bc.direct_beam_mu > 1.0) {
    if (bc.direct_beam_flux > 0.0) {  // Only required if there's a beam
      throw std::invalid_argument("direct_beam_mu must be in (0, 1] when direct_beam_flux > 0");
    }
  }

  if (flags.use_lambertian_surface) {
    if (bc.surface_albedo < 0.0 || bc.surface_albedo > 1.0) {
      throw std::invalid_argument("surface_albedo must be in [0, 1] when use_lambertian_surface=true");
    }
  }

  // User angle checks
  if (flags.use_user_mu) {
    if (num_user_mu <= 0) {
      throw std::invalid_argument("num_user_mu must be positive when use_user_mu=true");
    }

    if (mu_user.size() < static_cast<size_t>(num_user_mu)) {
      throw std::invalid_argument("mu_user array is too small");
    }

    // Check that mu_user values are in valid range [-1, 1] and sorted
    for (int iu = 0; iu < num_user_mu; ++iu) {
      if (std::abs(mu_user[iu]) > 1.0) {
        throw std::invalid_argument("mu_user[" + std::to_string(iu) + "] out of range [-1, 1]");
      }

      if (iu > 0 && mu_user[iu] <= mu_user[iu - 1]) {
        throw std::invalid_argument("mu_user must be in increasing order");
      }
    }
  }

  // Spherical geometry checks
  if (flags.use_spherical_beam) {
    if (bottom_radius <= 0.0) {
      throw std::invalid_argument("bottom_radius must be positive when use_spherical_beam=true");
    }

    if (level_altitudes.size() < static_cast<size_t>(num_layers + 1)) {
      throw std::invalid_argument("level_altitudes array is too small (need num_layers+1 when use_spherical_beam=true)");
    }
  }

  // Azimuthal angle checks (mirrors c_check_inputs lines 6153-6160)
  if (!flags.comp_only_fluxes && flags.ibcnd != BoundaryConditionType::Special) {
    if (num_phi <= 0) {
      throw std::invalid_argument("num_phi must be positive when comp_only_fluxes=false");
    }

    if (phi_user.size() < static_cast<size_t>(num_phi)) {
      throw std::invalid_argument("phi_user array is too small");
    }

    for (int j = 0; j < num_phi; ++j) {
      if (phi_user[j] < 0.0 || phi_user[j] > 360.0) {
        throw std::invalid_argument("phi_user[" + std::to_string(j) + "] must be in [0, 360]");
      }
    }
  }

  // Delta-M+ requires moments up to index num_streams + 1
  if (flags.use_delta_m_plus) {
    if (num_phase_func_moments < num_streams + 1) {
      throw std::invalid_argument(
        "use_delta_m_plus=true requires num_phase_func_moments >= num_streams + 1 (need moments M and M+1)");
    }
  }

  // Convergence criterion: 0.0 means disabled (run all azimuthal modes),
  // matching the C DISORT convention where accuracy_fourier_series=0 disables early exit.
  if (accuracy_fourier_series < 0.0) {
    throw std::invalid_argument("accuracy_fourier_series must be non-negative (0 = disabled)");
  }

  // Planck (thermal emission) mode checks
  if (flags.use_thermal_emission) {
    if (wavenumber_low < 0.0 || wavenumber_high <= wavenumber_low) {
      throw std::invalid_argument(
        "use_thermal_emission=true requires 0 <= wavenumber_low < wavenumber_high");
    }
    if (bc.temperature_top < 0.0 || bc.temperature_bottom < 0.0) {
      throw std::invalid_argument(
        "use_thermal_emission=true requires non-negative temperature_top and temperature_bottom");
    }
    if (bc.emissivity_top < 0.0 || bc.emissivity_top > 1.0) {
      throw std::invalid_argument(
        "use_thermal_emission=true requires 0 <= emissivity_top <= 1");
    }
    if (static_cast<int>(temperature.size()) < num_layers + 1) {
      throw std::invalid_argument(
        "use_thermal_emission=true requires temperature array of size num_layers+1");
    }
  }
}

void DisortConfig::setHenyeyGreenstein(double g, int lc) 
{
  phase_function::fillHenyeyGreenstein(phase_function_moments, nmomNstr(), num_layers, g, lc);
}

void DisortConfig::setIsotropic(int lc) 
{
  phase_function::fillIsotropic(phase_function_moments, nmomNstr(), num_layers, lc);
}

void DisortConfig::setRayleigh(int lc) 
{
  phase_function::fillRayleigh(phase_function_moments, nmomNstr(), num_layers, lc);
}

void DisortConfig::setHazeGarciaSiewert(int lc) 
{
  phase_function::fillHazeGarciaSiewert(phase_function_moments, nmomNstr(), num_layers, lc);
}

void DisortConfig::setCloudGarciaSiewert(int lc) 
{
  phase_function::fillCloudGarciaSiewert(phase_function_moments, nmomNstr(), num_layers, lc);
}

void DisortConfig::setPhaseFunction(PhaseFunction type, double g, int lc) 
{
  phase_function::fillPhaseFunction(phase_function_moments, nmomNstr(), num_layers, type, g, lc);
}

} // namespace disortpp
