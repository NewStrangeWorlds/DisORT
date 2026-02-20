#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "DisortTypes.hpp"
#include "DisortConfig.hpp"
#include "DisortFluxConfig.hpp"
#include "DisortResult.hpp"
#include "FluxResult.hpp"
#include "DisortSolver.hpp"
#include "FluxSolver.hpp"

namespace py = pybind11;
using namespace disortpp;

// ============================================================================
// Per-wavelength overrides for spectral loop functions
// ============================================================================

struct SpectralOverrides {
    std::vector<std::vector<double>> delta_tau;                        // [n_wl][num_layers]
    std::vector<std::vector<double>> single_scat_albedo;               // [n_wl][num_layers]
    std::vector<std::vector<std::vector<double>>> phase_function_moments; // [n_wl][num_layers][nmom+1]
    std::vector<double> surface_albedo;        // [n_wl]
    std::vector<double> emissivity_top;        // [n_wl]
    std::vector<double> direct_beam_flux;      // [n_wl]
    std::vector<double> isotropic_flux_top;    // [n_wl]
    std::vector<double> isotropic_flux_bottom; // [n_wl]
};

inline void applyOverrides(DisortConfig& cfg, const SpectralOverrides& ov, int i) {
    if (!ov.delta_tau.empty()) cfg.delta_tau = ov.delta_tau[i];
    if (!ov.single_scat_albedo.empty()) cfg.single_scat_albedo = ov.single_scat_albedo[i];
    if (!ov.phase_function_moments.empty()) cfg.phase_function_moments = ov.phase_function_moments[i];
    if (!ov.surface_albedo.empty()) cfg.bc.surface_albedo = ov.surface_albedo[i];
    if (!ov.emissivity_top.empty()) cfg.bc.emissivity_top = ov.emissivity_top[i];
    if (!ov.direct_beam_flux.empty()) cfg.bc.direct_beam_flux = ov.direct_beam_flux[i];
    if (!ov.isotropic_flux_top.empty()) cfg.bc.isotropic_flux_top = ov.isotropic_flux_top[i];
    if (!ov.isotropic_flux_bottom.empty()) cfg.bc.isotropic_flux_bottom = ov.isotropic_flux_bottom[i];
}

inline void applyOverrides(DisortFluxConfig& cfg, const SpectralOverrides& ov, int i) {
    if (!ov.delta_tau.empty()) cfg.delta_tau = ov.delta_tau[i];
    if (!ov.single_scat_albedo.empty()) cfg.single_scat_albedo = ov.single_scat_albedo[i];
    if (!ov.phase_function_moments.empty()) cfg.phase_function_moments = ov.phase_function_moments[i];
    if (!ov.surface_albedo.empty()) cfg.surface_albedo = ov.surface_albedo[i];
    if (!ov.emissivity_top.empty()) cfg.emissivity_top = ov.emissivity_top[i];
    if (!ov.direct_beam_flux.empty()) cfg.direct_beam_flux = ov.direct_beam_flux[i];
    if (!ov.isotropic_flux_top.empty()) cfg.isotropic_flux_top = ov.isotropic_flux_top[i];
    if (!ov.isotropic_flux_bottom.empty()) cfg.isotropic_flux_bottom = ov.isotropic_flux_bottom[i];
}

// ============================================================================
// Spectral loop helpers — run the wavenumber loop in C++ with GIL released
// ============================================================================

static std::vector<DisortResult> solveSpectral(
    DisortConfig config, const std::vector<double>& wavenumbers,
    const SpectralOverrides& ov) {
  const int n = static_cast<int>(wavenumbers.size());
  std::vector<DisortResult> results(n);
  #pragma omp parallel
  {
    DisortSolver solver;
    DisortConfig local_config = config;
    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) {
      local_config.wavenumber_low = wavenumbers[i];
      local_config.wavenumber_high = wavenumbers[i];
      applyOverrides(local_config, ov, i);
      results[i] = solver.solve(local_config);
    }
  }
  return results;
}

static std::vector<DisortResult> solveSpectralBands(
    DisortConfig config,
    const std::vector<std::pair<double, double>>& wavenumber_bands,
    const SpectralOverrides& ov) {
  const int n = static_cast<int>(wavenumber_bands.size());
  std::vector<DisortResult> results(n);
  #pragma omp parallel
  {
    DisortSolver solver;
    DisortConfig local_config = config;
    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) {
      local_config.wavenumber_low = wavenumber_bands[i].first;
      local_config.wavenumber_high = wavenumber_bands[i].second;
      applyOverrides(local_config, ov, i);
      results[i] = solver.solve(local_config);
    }
  }
  return results;
}

template<int NStr>
static std::vector<FluxResult> solveFluxSpectralImpl(
    DisortFluxConfig& config, const std::vector<double>& wavenumbers,
    const SpectralOverrides& ov) {
  const int n = static_cast<int>(wavenumbers.size());
  std::vector<FluxResult> results(n);
  #pragma omp parallel
  {
    DisortFluxSolver<NStr> solver;
    DisortFluxConfig local_config = config;
    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) {
      local_config.wavenumber_low = wavenumbers[i];
      local_config.wavenumber_high = wavenumbers[i];
      applyOverrides(local_config, ov, i);
      results[i] = solver.solve(local_config);
    }
  }
  return results;
}

static std::vector<FluxResult> solveFluxSpectral(
    DisortFluxConfig& config, const std::vector<double>& wavenumbers,
    const SpectralOverrides& ov) {
  switch (config.num_streams) {
    case 4:  return solveFluxSpectralImpl<4>(config, wavenumbers, ov);
    case 6:  return solveFluxSpectralImpl<6>(config, wavenumbers, ov);
    case 8:  return solveFluxSpectralImpl<8>(config, wavenumbers, ov);
    case 10: return solveFluxSpectralImpl<10>(config, wavenumbers, ov);
    case 12: return solveFluxSpectralImpl<12>(config, wavenumbers, ov);
    case 14: return solveFluxSpectralImpl<14>(config, wavenumbers, ov);
    case 16: return solveFluxSpectralImpl<16>(config, wavenumbers, ov);
    case 32: return solveFluxSpectralImpl<32>(config, wavenumbers, ov);
    case 64: return solveFluxSpectralImpl<64>(config, wavenumbers, ov);
    default:
      throw std::invalid_argument(
          "Unsupported num_streams=" + std::to_string(config.num_streams) +
          ". Supported values: 4, 6, 8, 10, 12, 14, 16, 32, 64");
  }
}

template<int NStr>
static std::vector<FluxResult> solveFluxSpectralBandsImpl(
    DisortFluxConfig& config,
    const std::vector<std::pair<double, double>>& wavenumber_bands,
    const SpectralOverrides& ov) {
  const int n = static_cast<int>(wavenumber_bands.size());
  std::vector<FluxResult> results(n);
  #pragma omp parallel
  {
    DisortFluxSolver<NStr> solver;
    DisortFluxConfig local_config = config;
    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) {
      local_config.wavenumber_low = wavenumber_bands[i].first;
      local_config.wavenumber_high = wavenumber_bands[i].second;
      applyOverrides(local_config, ov, i);
      results[i] = solver.solve(local_config);
    }
  }
  return results;
}

static std::vector<FluxResult> solveFluxSpectralBands(
    DisortFluxConfig& config,
    const std::vector<std::pair<double, double>>& wavenumber_bands,
    const SpectralOverrides& ov) {
  switch (config.num_streams) {
    case 4:  return solveFluxSpectralBandsImpl<4>(config, wavenumber_bands, ov);
    case 6:  return solveFluxSpectralBandsImpl<6>(config, wavenumber_bands, ov);
    case 8:  return solveFluxSpectralBandsImpl<8>(config, wavenumber_bands, ov);
    case 10: return solveFluxSpectralBandsImpl<10>(config, wavenumber_bands, ov);
    case 12: return solveFluxSpectralBandsImpl<12>(config, wavenumber_bands, ov);
    case 14: return solveFluxSpectralBandsImpl<14>(config, wavenumber_bands, ov);
    case 16: return solveFluxSpectralBandsImpl<16>(config, wavenumber_bands, ov);
    case 32: return solveFluxSpectralBandsImpl<32>(config, wavenumber_bands, ov);
    case 64: return solveFluxSpectralBandsImpl<64>(config, wavenumber_bands, ov);
    default:
      throw std::invalid_argument(
          "Unsupported num_streams=" + std::to_string(config.num_streams) +
          ". Supported values: 4, 6, 8, 10, 12, 14, 16, 32, 64");
  }
}

PYBIND11_MODULE(disortpp, m) {
  m.doc() = "DisORT++ - Discrete Ordinates Radiative Transfer solver";
  m.attr("__version__") = "1.0";

  // ========================================================================
  // Enums
  // ========================================================================

  py::enum_<PhaseFunction>(m, "PhaseFunction")
      .value("Isotropic", PhaseFunction::Isotropic)
      .value("Rayleigh", PhaseFunction::Rayleigh)
      .value("HenyeyGreenstein", PhaseFunction::HenyeyGreenstein)
      .value("HazeGarciaSiewert", PhaseFunction::HazeGarciaSiewert)
      .value("CloudGarciaSiewert", PhaseFunction::CloudGarciaSiewert);

  py::enum_<BoundaryConditionType>(m, "BoundaryConditionType")
      .value("General", BoundaryConditionType::General)
      .value("Special", BoundaryConditionType::Special);

  py::enum_<BrdfType>(m, "BrdfType")
      .value("NONE", BrdfType::None)
      .value("RPV", BrdfType::RPV)
      .value("CoxMunk", BrdfType::CoxMunk)
      .value("Ambrals", BrdfType::Ambrals)
      .value("Hapke", BrdfType::Hapke);

  // ========================================================================
  // POD structs
  // ========================================================================

  py::class_<DisortFlags>(m, "DisortFlags")
      .def(py::init<>())
      .def_readwrite("use_user_tau", &DisortFlags::use_user_tau)
      .def_readwrite("use_user_mu", &DisortFlags::use_user_mu)
      .def_readwrite("ibcnd", &DisortFlags::ibcnd)
      .def_readwrite("use_lambertian_surface", &DisortFlags::use_lambertian_surface)
      .def_readwrite("use_thermal_emission", &DisortFlags::use_thermal_emission)
      .def_readwrite("use_spherical_beam", &DisortFlags::use_spherical_beam)
      .def_readwrite("comp_only_fluxes", &DisortFlags::comp_only_fluxes)
      .def_readwrite("brdf_type", &DisortFlags::brdf_type)
      .def_readwrite("intensity_corr_buras", &DisortFlags::intensity_corr_buras)
      .def_readwrite("intensity_corr_nakajima", &DisortFlags::intensity_corr_nakajima)
      .def_readwrite("use_delta_m_plus", &DisortFlags::use_delta_m_plus)
      .def_readwrite("output_fourier_expansion", &DisortFlags::output_fourier_expansion);

  py::class_<BoundaryConditions>(m, "BoundaryConditions")
      .def(py::init<>())
      .def_readwrite("direct_beam_flux", &BoundaryConditions::direct_beam_flux)
      .def_readwrite("direct_beam_mu", &BoundaryConditions::direct_beam_mu)
      .def_readwrite("direct_beam_phi", &BoundaryConditions::direct_beam_phi)
      .def_readwrite("isotropic_flux_top", &BoundaryConditions::isotropic_flux_top)
      .def_readwrite("isotropic_flux_bottom", &BoundaryConditions::isotropic_flux_bottom)
      .def_readwrite("temperature_top", &BoundaryConditions::temperature_top)
      .def_readwrite("temperature_bottom", &BoundaryConditions::temperature_bottom)
      .def_readwrite("emissivity_top", &BoundaryConditions::emissivity_top)
      .def_readwrite("surface_albedo", &BoundaryConditions::surface_albedo);

  py::class_<RpvBrdfSpec>(m, "RpvBrdfSpec")
      .def(py::init<>())
      .def_readwrite("rho0", &RpvBrdfSpec::rho0)
      .def_readwrite("k", &RpvBrdfSpec::k)
      .def_readwrite("theta", &RpvBrdfSpec::theta)
      .def_readwrite("sigma", &RpvBrdfSpec::sigma)
      .def_readwrite("t1", &RpvBrdfSpec::t1)
      .def_readwrite("t2", &RpvBrdfSpec::t2)
      .def_readwrite("scale", &RpvBrdfSpec::scale);

  py::class_<AmbralsBrdfSpec>(m, "AmbralsBrdfSpec")
      .def(py::init<>())
      .def_readwrite("iso", &AmbralsBrdfSpec::iso)
      .def_readwrite("vol", &AmbralsBrdfSpec::vol)
      .def_readwrite("geo", &AmbralsBrdfSpec::geo);

  py::class_<CoxMunkBrdfSpec>(m, "CoxMunkBrdfSpec")
      .def(py::init<>())
      .def_readwrite("u10", &CoxMunkBrdfSpec::u10)
      .def_readwrite("pcl", &CoxMunkBrdfSpec::pcl)
      .def_readwrite("xsal", &CoxMunkBrdfSpec::xsal)
      .def_readwrite("refractive_index", &CoxMunkBrdfSpec::refractive_index)
      .def_readwrite("do_shadow", &CoxMunkBrdfSpec::do_shadow);

  py::class_<HapkeBrdfSpec>(m, "HapkeBrdfSpec")
      .def(py::init<>())
      .def_readwrite("b0", &HapkeBrdfSpec::b0)
      .def_readwrite("hh", &HapkeBrdfSpec::hh)
      .def_readwrite("w", &HapkeBrdfSpec::w);

  py::class_<BrdfSpecification>(m, "BrdfSpecification")
      .def(py::init<>())
      .def_readwrite("rpv", &BrdfSpecification::rpv)
      .def_readwrite("ambrals", &BrdfSpecification::ambrals)
      .def_readwrite("cox_munk", &BrdfSpecification::cox_munk)
      .def_readwrite("hapke", &BrdfSpecification::hapke)
      .def("set_rpv", &BrdfSpecification::setRpv, py::arg("spec"))
      .def("set_ambrals", &BrdfSpecification::setAmbrals, py::arg("spec"))
      .def("set_cox_munk", &BrdfSpecification::setCoxMunk, py::arg("spec"))
      .def("set_hapke", &BrdfSpecification::setHapke, py::arg("spec"));

  // ========================================================================
  // DisortConfig
  // ========================================================================

  py::class_<DisortConfig>(m, "DisortConfig")
      .def(py::init<>())
      .def(py::init<int, int, int>(),
           py::arg("num_layers"), py::arg("num_streams"),
           py::arg("num_phase_func_moments") = 0)

      // Nested structs
      .def_readwrite("flags", &DisortConfig::flags)
      .def_readwrite("bc", &DisortConfig::bc)
      .def_readwrite("brdf", &DisortConfig::brdf)

      // Dimensions
      .def_readwrite("num_layers", &DisortConfig::num_layers)
      .def_readwrite("num_phase_func_moments", &DisortConfig::num_phase_func_moments)
      .def_readwrite("num_streams", &DisortConfig::num_streams)
      .def_readwrite("num_user_tau", &DisortConfig::num_user_tau)
      .def_readwrite("num_user_mu", &DisortConfig::num_user_mu)
      .def_readwrite("num_phi", &DisortConfig::num_phi)
      .def_readwrite("num_phase_func_angles", &DisortConfig::num_phase_func_angles)

      // Physical parameters
      .def_readwrite("wavenumber_low", &DisortConfig::wavenumber_low)
      .def_readwrite("wavenumber_high", &DisortConfig::wavenumber_high)
      .def_readwrite("accuracy_fourier_series", &DisortConfig::accuracy_fourier_series)
      .def_readwrite("bottom_radius", &DisortConfig::bottom_radius)

      // Layer/level arrays
      .def_readwrite("delta_tau", &DisortConfig::delta_tau)
      .def_readwrite("single_scat_albedo", &DisortConfig::single_scat_albedo)
      .def_readwrite("phase_function_moments", &DisortConfig::phase_function_moments)
      .def_readwrite("temperature", &DisortConfig::temperature)
      .def_readwrite("level_altitudes", &DisortConfig::level_altitudes)

      // User output specification
      .def_readwrite("tau_user", &DisortConfig::tau_user)
      .def_readwrite("mu_user", &DisortConfig::mu_user)
      .def_readwrite("phi_user", &DisortConfig::phi_user)

      // Phase function specification
      .def_readwrite("mu_phase_function", &DisortConfig::mu_phase_function)
      .def_readwrite("phase_function", &DisortConfig::phase_function)

      // Methods
      .def("nmom_nstr", &DisortConfig::nmomNstr)
      .def("allocate", &DisortConfig::allocate)
      .def("validate", &DisortConfig::validate)
      .def("set_henyey_greenstein", &DisortConfig::setHenyeyGreenstein,
           py::arg("g"), py::arg("lc") = -1)
      .def("set_isotropic", &DisortConfig::setIsotropic,
           py::arg("lc") = -1)
      .def("set_rayleigh", &DisortConfig::setRayleigh,
           py::arg("lc") = -1)
      .def("set_haze_garcia_siewert", &DisortConfig::setHazeGarciaSiewert,
           py::arg("lc") = -1)
      .def("set_cloud_garcia_siewert", &DisortConfig::setCloudGarciaSiewert,
           py::arg("lc") = -1)
      .def("set_phase_function", &DisortConfig::setPhaseFunction,
           py::arg("type"), py::arg("g") = 0.0, py::arg("lc") = -1);

  // ========================================================================
  // DisortFluxConfig
  // ========================================================================

  py::class_<DisortFluxConfig>(m, "DisortFluxConfig")
      .def(py::init<>())
      .def(py::init<int, int, int>(),
           py::arg("num_layers"), py::arg("num_streams"),
           py::arg("num_phase_func_moments") = 0)

      // Dimensions
      .def_readwrite("num_layers", &DisortFluxConfig::num_layers)
      .def_readwrite("num_streams", &DisortFluxConfig::num_streams)
      .def_readwrite("num_phase_func_moments", &DisortFluxConfig::num_phase_func_moments)

      // Flags
      .def_readwrite("use_thermal_emission", &DisortFluxConfig::use_thermal_emission)
      .def_readwrite("use_spherical_beam", &DisortFluxConfig::use_spherical_beam)
      .def_readwrite("use_delta_m_plus", &DisortFluxConfig::use_delta_m_plus)

      // Boundary conditions
      .def_readwrite("direct_beam_flux", &DisortFluxConfig::direct_beam_flux)
      .def_readwrite("direct_beam_mu", &DisortFluxConfig::direct_beam_mu)
      .def_readwrite("isotropic_flux_top", &DisortFluxConfig::isotropic_flux_top)
      .def_readwrite("isotropic_flux_bottom", &DisortFluxConfig::isotropic_flux_bottom)
      .def_readwrite("temperature_top", &DisortFluxConfig::temperature_top)
      .def_readwrite("temperature_bottom", &DisortFluxConfig::temperature_bottom)
      .def_readwrite("emissivity_top", &DisortFluxConfig::emissivity_top)
      .def_readwrite("surface_albedo", &DisortFluxConfig::surface_albedo)

      // Physical parameters
      .def_readwrite("wavenumber_low", &DisortFluxConfig::wavenumber_low)
      .def_readwrite("wavenumber_high", &DisortFluxConfig::wavenumber_high)
      .def_readwrite("bottom_radius", &DisortFluxConfig::bottom_radius)

      // Layer/level arrays
      .def_readwrite("delta_tau", &DisortFluxConfig::delta_tau)
      .def_readwrite("single_scat_albedo", &DisortFluxConfig::single_scat_albedo)
      .def_readwrite("phase_function_moments", &DisortFluxConfig::phase_function_moments)
      .def_readwrite("temperature", &DisortFluxConfig::temperature)
      .def_readwrite("level_altitudes", &DisortFluxConfig::level_altitudes)

      // Methods
      .def("nmom_nstr", &DisortFluxConfig::nmomNstr)
      .def("allocate", &DisortFluxConfig::allocate)
      .def("validate", &DisortFluxConfig::validate)
      .def("set_henyey_greenstein", &DisortFluxConfig::setHenyeyGreenstein,
           py::arg("g"), py::arg("lc") = -1)
      .def("set_isotropic", &DisortFluxConfig::setIsotropic,
           py::arg("lc") = -1)
      .def("set_rayleigh", &DisortFluxConfig::setRayleigh,
           py::arg("lc") = -1)
      .def("set_haze_garcia_siewert", &DisortFluxConfig::setHazeGarciaSiewert,
           py::arg("lc") = -1)
      .def("set_cloud_garcia_siewert", &DisortFluxConfig::setCloudGarciaSiewert,
           py::arg("lc") = -1)
      .def("set_phase_function", &DisortFluxConfig::setPhaseFunction,
           py::arg("type"), py::arg("g") = 0.0, py::arg("lc") = -1);

  // ========================================================================
  // DisortResult
  // ========================================================================

  py::class_<DisortResult>(m, "DisortResult")
      .def(py::init<>())

      // 1D flux vectors
      .def_readwrite("flux_direct_beam", &DisortResult::flux_direct_beam)
      .def_readwrite("flux_down", &DisortResult::flux_down)
      .def_readwrite("flux_up", &DisortResult::flux_up)
      .def_readwrite("flux_tau_divergence", &DisortResult::flux_tau_divergence)
      .def_readwrite("mean_intensity", &DisortResult::mean_intensity)
      .def_readwrite("mean_intensity_down", &DisortResult::mean_intensity_down)
      .def_readwrite("mean_intensity_up", &DisortResult::mean_intensity_up)
      .def_readwrite("mean_intensity_direct_beam", &DisortResult::mean_intensity_direct_beam)

      // Special BC mode
      .def_readwrite("albedo_medium", &DisortResult::albedo_medium)
      .def_readwrite("transmissivity_medium", &DisortResult::transmissivity_medium)
      .def_readwrite("spherical_albedo", &DisortResult::spherical_albedo)
      .def_readwrite("spherical_transmissivity", &DisortResult::spherical_transmissivity)

      // Angle information
      .def_readwrite("mu_angles", &DisortResult::mu_angles)

      // 2D/3D intensity arrays
      .def_readwrite("intensity", &DisortResult::intensity)
      .def_readwrite("intensity_azimuth_avg", &DisortResult::intensity_azimuth_avg)
      .def_readwrite("intensity_fourier_expansion", &DisortResult::intensity_fourier_expansion)

      // Methods
      .def("num_user_tau", &DisortResult::num_user_tau)
      .def("num_user_mu", &DisortResult::num_user_mu)
      .def("num_phi", &DisortResult::num_phi)
      .def("total_flux_down", &DisortResult::totalFluxDown, py::arg("lu"))
      .def("net_flux", &DisortResult::netFlux, py::arg("lu"))
      .def("intensities",
           py::overload_cast<int, int, int>(&DisortResult::intensities, py::const_),
           py::arg("iu"), py::arg("lu"), py::arg("j"))
      .def("intensities_azimuth_avg",
           py::overload_cast<int, int>(&DisortResult::intensitiesAzimuthAvg, py::const_),
           py::arg("iu"), py::arg("lu"));

  // ========================================================================
  // FluxResult
  // ========================================================================

  py::class_<FluxResult>(m, "FluxResult")
      .def(py::init<>())
      .def_readwrite("flux_direct_beam", &FluxResult::flux_direct_beam)
      .def_readwrite("flux_down", &FluxResult::flux_down)
      .def_readwrite("flux_up", &FluxResult::flux_up)
      .def_readwrite("flux_tau_divergence", &FluxResult::flux_tau_divergence)
      .def_readwrite("mean_intensity", &FluxResult::mean_intensity)
      .def_readwrite("mean_intensity_down", &FluxResult::mean_intensity_down)
      .def_readwrite("mean_intensity_up", &FluxResult::mean_intensity_up)
      .def_readwrite("mean_intensity_direct_beam", &FluxResult::mean_intensity_direct_beam)
      .def("num_levels", &FluxResult::num_levels)
      .def("total_flux_down", &FluxResult::totalFluxDown, py::arg("lev"))
      .def("net_flux", &FluxResult::netFlux, py::arg("lev"));

  // ========================================================================
  // DisortSolver
  // ========================================================================

  py::class_<DisortSolver>(m, "DisortSolver",
      "General discrete ordinates radiative transfer solver")
      .def(py::init<>())
      .def("solve", [](DisortSolver& self, DisortConfig& config) {
        py::gil_scoped_release release;
        return self.solve(config);
      }, py::arg("config"),
         "Solve the radiative transfer problem. Note: config may be modified "
         "by the solver to set automatic dimensions.");

  // ========================================================================
  // DisortFluxSolver (template instantiations)
  // ========================================================================

  py::class_<DisortFluxSolver<4>>(m, "DisortFluxSolver4",
      "Flux-only solver with 4 streams")
      .def(py::init<>())
      .def("solve", [](DisortFluxSolver<4>& self, const DisortFluxConfig& config) {
        py::gil_scoped_release release;
        return self.solve(config);
      }, py::arg("config"));

  py::class_<DisortFluxSolver<6>>(m, "DisortFluxSolver6",
      "Flux-only solver with 6 streams")
      .def(py::init<>())
      .def("solve", [](DisortFluxSolver<6>& self, const DisortFluxConfig& config) {
        py::gil_scoped_release release;
        return self.solve(config);
      }, py::arg("config"));

  py::class_<DisortFluxSolver<8>>(m, "DisortFluxSolver8",
      "Flux-only solver with 8 streams")
      .def(py::init<>())
      .def("solve", [](DisortFluxSolver<8>& self, const DisortFluxConfig& config) {
        py::gil_scoped_release release;
        return self.solve(config);
      }, py::arg("config"));

  py::class_<DisortFluxSolver<10>>(m, "DisortFluxSolver10",
      "Flux-only solver with 10 streams")
      .def(py::init<>())
      .def("solve", [](DisortFluxSolver<10>& self, const DisortFluxConfig& config) {
        py::gil_scoped_release release;
        return self.solve(config);
      }, py::arg("config"));

  py::class_<DisortFluxSolver<12>>(m, "DisortFluxSolver12",
      "Flux-only solver with 12 streams")
      .def(py::init<>())
      .def("solve", [](DisortFluxSolver<12>& self, const DisortFluxConfig& config) {
        py::gil_scoped_release release;
        return self.solve(config);
      }, py::arg("config"));

  py::class_<DisortFluxSolver<14>>(m, "DisortFluxSolver14",
      "Flux-only solver with 14 streams")
      .def(py::init<>())
      .def("solve", [](DisortFluxSolver<14>& self, const DisortFluxConfig& config) {
        py::gil_scoped_release release;
        return self.solve(config);
      }, py::arg("config"));

  py::class_<DisortFluxSolver<16>>(m, "DisortFluxSolver16",
      "Flux-only solver with 16 streams")
      .def(py::init<>())
      .def("solve", [](DisortFluxSolver<16>& self, const DisortFluxConfig& config) {
        py::gil_scoped_release release;
        return self.solve(config);
      }, py::arg("config"));

  py::class_<DisortFluxSolver<32>>(m, "DisortFluxSolver32",
      "Flux-only solver with 32 streams")
      .def(py::init<>())
      .def("solve", [](DisortFluxSolver<32>& self, const DisortFluxConfig& config) {
        py::gil_scoped_release release;
        return self.solve(config);
      }, py::arg("config"));

  py::class_<DisortFluxSolver<64>>(m, "DisortFluxSolver64",
      "Flux-only solver with 64 streams")
      .def(py::init<>())
      .def("solve", [](DisortFluxSolver<64>& self, const DisortFluxConfig& config) {
        py::gil_scoped_release release;
        return self.solve(config);
      }, py::arg("config"));

  // Factory function
  m.def("create_flux_solver", [](int nstr) -> py::object {
    switch (nstr) {
      case 4:  return py::cast(DisortFluxSolver<4>());
      case 6:  return py::cast(DisortFluxSolver<6>());
      case 8:  return py::cast(DisortFluxSolver<8>());
      case 10: return py::cast(DisortFluxSolver<10>());
      case 12: return py::cast(DisortFluxSolver<12>());
      case 14: return py::cast(DisortFluxSolver<14>());
      case 16: return py::cast(DisortFluxSolver<16>());
      case 32: return py::cast(DisortFluxSolver<32>());
      case 64: return py::cast(DisortFluxSolver<64>());
      default:
        throw std::invalid_argument(
            "Unsupported nstr=" + std::to_string(nstr) +
            ". Supported values: 4, 6, 8, 10, 12, 14, 16, 32, 64");
    }
  }, py::arg("nstr"),
     "Create a flux-only solver for the given number of streams "
     "(4, 6, 8, 10, 12, 14, 16, 32, or 64)");

  // ========================================================================
  // Spectral loop functions
  // ========================================================================

  m.def("solve_spectral",
      [](DisortConfig config, const std::vector<double>& wavenumbers,
         std::vector<std::vector<double>> delta_tau,
         std::vector<std::vector<double>> single_scat_albedo,
         std::vector<std::vector<std::vector<double>>> phase_function_moments,
         std::vector<double> surface_albedo,
         std::vector<double> emissivity_top,
         std::vector<double> direct_beam_flux,
         std::vector<double> isotropic_flux_top,
         std::vector<double> isotropic_flux_bottom) {
        SpectralOverrides ov{
            std::move(delta_tau), std::move(single_scat_albedo),
            std::move(phase_function_moments),
            std::move(surface_albedo), std::move(emissivity_top),
            std::move(direct_beam_flux),
            std::move(isotropic_flux_top), std::move(isotropic_flux_bottom)};
        py::gil_scoped_release release;
        return solveSpectral(std::move(config), wavenumbers, ov);
      },
      py::arg("config"), py::arg("wavenumbers"),
      py::arg("delta_tau") = std::vector<std::vector<double>>{},
      py::arg("single_scat_albedo") = std::vector<std::vector<double>>{},
      py::arg("phase_function_moments") = std::vector<std::vector<std::vector<double>>>{},
      py::arg("surface_albedo") = std::vector<double>{},
      py::arg("emissivity_top") = std::vector<double>{},
      py::arg("direct_beam_flux") = std::vector<double>{},
      py::arg("isotropic_flux_top") = std::vector<double>{},
      py::arg("isotropic_flux_bottom") = std::vector<double>{},
      "Solve for a list of single wavenumbers (sets wavenumber_low = wavenumber_high = wn). "
      "Optional per-wavelength arrays override the corresponding config values. "
      "The wavenumber loop runs entirely in C++. Returns a list of DisortResult.");

  m.def("solve_spectral_bands",
      [](DisortConfig config,
         const std::vector<std::pair<double, double>>& wavenumber_bands,
         std::vector<std::vector<double>> delta_tau,
         std::vector<std::vector<double>> single_scat_albedo,
         std::vector<std::vector<std::vector<double>>> phase_function_moments,
         std::vector<double> surface_albedo,
         std::vector<double> emissivity_top,
         std::vector<double> direct_beam_flux,
         std::vector<double> isotropic_flux_top,
         std::vector<double> isotropic_flux_bottom) {
        SpectralOverrides ov{
            std::move(delta_tau), std::move(single_scat_albedo),
            std::move(phase_function_moments),
            std::move(surface_albedo), std::move(emissivity_top),
            std::move(direct_beam_flux),
            std::move(isotropic_flux_top), std::move(isotropic_flux_bottom)};
        py::gil_scoped_release release;
        return solveSpectralBands(std::move(config), wavenumber_bands, ov);
      },
      py::arg("config"), py::arg("wavenumber_bands"),
      py::arg("delta_tau") = std::vector<std::vector<double>>{},
      py::arg("single_scat_albedo") = std::vector<std::vector<double>>{},
      py::arg("phase_function_moments") = std::vector<std::vector<std::vector<double>>>{},
      py::arg("surface_albedo") = std::vector<double>{},
      py::arg("emissivity_top") = std::vector<double>{},
      py::arg("direct_beam_flux") = std::vector<double>{},
      py::arg("isotropic_flux_top") = std::vector<double>{},
      py::arg("isotropic_flux_bottom") = std::vector<double>{},
      "Solve for a list of wavenumber bands (pairs of (low, high)). "
      "Optional per-wavelength arrays override the corresponding config values. "
      "The wavenumber loop runs entirely in C++. Returns a list of DisortResult.");

  m.def("solve_flux_spectral",
      [](DisortFluxConfig& config, const std::vector<double>& wavenumbers,
         std::vector<std::vector<double>> delta_tau,
         std::vector<std::vector<double>> single_scat_albedo,
         std::vector<std::vector<std::vector<double>>> phase_function_moments,
         std::vector<double> surface_albedo,
         std::vector<double> emissivity_top,
         std::vector<double> direct_beam_flux,
         std::vector<double> isotropic_flux_top,
         std::vector<double> isotropic_flux_bottom) {
        SpectralOverrides ov{
            std::move(delta_tau), std::move(single_scat_albedo),
            std::move(phase_function_moments),
            std::move(surface_albedo), std::move(emissivity_top),
            std::move(direct_beam_flux),
            std::move(isotropic_flux_top), std::move(isotropic_flux_bottom)};
        py::gil_scoped_release release;
        return solveFluxSpectral(config, wavenumbers, ov);
      },
      py::arg("config"), py::arg("wavenumbers"),
      py::arg("delta_tau") = std::vector<std::vector<double>>{},
      py::arg("single_scat_albedo") = std::vector<std::vector<double>>{},
      py::arg("phase_function_moments") = std::vector<std::vector<std::vector<double>>>{},
      py::arg("surface_albedo") = std::vector<double>{},
      py::arg("emissivity_top") = std::vector<double>{},
      py::arg("direct_beam_flux") = std::vector<double>{},
      py::arg("isotropic_flux_top") = std::vector<double>{},
      py::arg("isotropic_flux_bottom") = std::vector<double>{},
      "Solve for fluxes at a list of single wavenumbers. "
      "Optional per-wavelength arrays override the corresponding config values. "
      "Dispatches to DisortFluxSolver based on config.num_streams. "
      "Returns a list of FluxResult.");

  m.def("solve_flux_spectral_bands",
      [](DisortFluxConfig& config,
         const std::vector<std::pair<double, double>>& wavenumber_bands,
         std::vector<std::vector<double>> delta_tau,
         std::vector<std::vector<double>> single_scat_albedo,
         std::vector<std::vector<std::vector<double>>> phase_function_moments,
         std::vector<double> surface_albedo,
         std::vector<double> emissivity_top,
         std::vector<double> direct_beam_flux,
         std::vector<double> isotropic_flux_top,
         std::vector<double> isotropic_flux_bottom) {
        SpectralOverrides ov{
            std::move(delta_tau), std::move(single_scat_albedo),
            std::move(phase_function_moments),
            std::move(surface_albedo), std::move(emissivity_top),
            std::move(direct_beam_flux),
            std::move(isotropic_flux_top), std::move(isotropic_flux_bottom)};
        py::gil_scoped_release release;
        return solveFluxSpectralBands(config, wavenumber_bands, ov);
      },
      py::arg("config"), py::arg("wavenumber_bands"),
      py::arg("delta_tau") = std::vector<std::vector<double>>{},
      py::arg("single_scat_albedo") = std::vector<std::vector<double>>{},
      py::arg("phase_function_moments") = std::vector<std::vector<std::vector<double>>>{},
      py::arg("surface_albedo") = std::vector<double>{},
      py::arg("emissivity_top") = std::vector<double>{},
      py::arg("direct_beam_flux") = std::vector<double>{},
      py::arg("isotropic_flux_top") = std::vector<double>{},
      py::arg("isotropic_flux_bottom") = std::vector<double>{},
      "Solve for fluxes at a list of wavenumber bands (pairs of (low, high)). "
      "Optional per-wavelength arrays override the corresponding config values. "
      "Dispatches to DisortFluxSolver based on config.num_streams. "
      "Returns a list of FluxResult.");
}
