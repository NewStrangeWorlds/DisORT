#include "testing.hpp"
#include "FluxSolver.hpp"
#include "DisortSolver.hpp"
#include "DisortConfig.hpp"
#include "DisortFluxConfig.hpp"
#include <cmath>

using namespace disortpp;

// ============================================================================
// Helper: Convert DisortConfig to DisortFluxConfig
// ============================================================================

static DisortFluxConfig toFluxConfig(const DisortConfig& dc) {
  DisortFluxConfig fc(dc.num_layers, dc.num_streams, dc.num_phase_func_moments);
  fc.use_thermal_emission = dc.flags.use_thermal_emission;
  fc.use_spherical_beam = dc.flags.use_spherical_beam;
  fc.use_delta_m_plus = dc.flags.use_delta_m_plus;
  fc.direct_beam_flux = dc.bc.direct_beam_flux;
  fc.direct_beam_mu = dc.bc.direct_beam_mu;
  fc.isotropic_flux_top = dc.bc.isotropic_flux_top;
  fc.isotropic_flux_bottom = dc.bc.isotropic_flux_bottom;
  fc.temperature_top = dc.bc.temperature_top;
  fc.temperature_bottom = dc.bc.temperature_bottom;
  fc.emissivity_top = dc.bc.emissivity_top;
  fc.surface_albedo = dc.bc.surface_albedo;
  fc.wavenumber_low = dc.wavenumber_low;
  fc.wavenumber_high = dc.wavenumber_high;
  fc.bottom_radius = dc.bottom_radius;
  fc.delta_tau = dc.delta_tau;
  fc.single_scat_albedo = dc.single_scat_albedo;
  fc.phase_function_moments = dc.phase_function_moments;
  fc.temperature = dc.temperature;
  fc.level_altitudes = dc.level_altitudes;
  return fc;
}

// ============================================================================
// Helper: Compare DisortFluxSolver<NStr> vs DisortSolver (comp_only_fluxes=true)
// ============================================================================

/**
 * @brief Compare all 8 flux/mean-intensity vectors between FluxResult and DisortResult
 * @return true if all values match within tolerance
 */
static bool compareFluxes(const FluxResult& flux, const DisortResult& disort,
              int nlev, double rtol, const char* label) {
  bool ok = true;
  for (int lu = 0; lu < nlev; ++lu) {
    auto check = [&](double a, double b, const char* name) {
      // Combined relative + absolute tolerance (numpy-style)
      // Use 2e-9 absolute tolerance because fixed-size vs dynamic Eigen solvers
      // and LAPACK vs Eigen dense solvers produce ~1e-9 noise in theoretically-zero quantities
      double atol = 2e-9;
      double threshold = rtol * std::max(std::abs(a), std::abs(b)) + atol;
      if (std::abs(a - b) > threshold) {
        std::cerr << label << " level " << lu << " " << name
              << ": DisortFluxSolver=" << a << " DisortSolver=" << b
              << " diff=" << std::abs(a - b) << " tol=" << threshold << "\n";
        ok = false;
      }
    };
    check(flux.flux_direct_beam[lu], disort.flux_direct_beam[lu], "flux_direct_beam");
    check(flux.flux_down[lu], disort.flux_down[lu], "flux_down");
    check(flux.flux_up[lu], disort.flux_up[lu], "flux_up");
    check(flux.flux_tau_divergence[lu], disort.flux_tau_divergence[lu], "flux_tau_divergence");
    check(flux.mean_intensity[lu], disort.mean_intensity[lu], "mean_intensity");
    check(flux.mean_intensity_down[lu], disort.mean_intensity_down[lu], "mean_intensity_down");
    check(flux.mean_intensity_up[lu], disort.mean_intensity_up[lu], "mean_intensity_up");
    check(flux.mean_intensity_direct_beam[lu], disort.mean_intensity_direct_beam[lu], "mean_intensity_direct_beam");
  }
  return ok;
}

/**
 * @brief Run DisortFluxSolver<NStr> and DisortSolver on the same config and compare
 */
template<int NStr>
static bool runComparison(DisortConfig config, double rtol, const char* label) {
  // Run DisortSolver with comp_only_fluxes=true, use_user_tau=false
  DisortConfig cfg_disort = config;
  cfg_disort.flags.comp_only_fluxes = true;
  cfg_disort.flags.use_user_tau = false;
  cfg_disort.num_phi = 0;

  DisortSolver ds;
  DisortResult dr = ds.solve(cfg_disort);

  // Run DisortFluxSolver with converted DisortFluxConfig
  DisortFluxConfig cfg_flux = toFluxConfig(config);
  DisortFluxSolver<NStr> fs;
  FluxResult fr = fs.solve(cfg_flux);

  // Compare
  const int nlev = config.num_layers + 1;
  EXPECT_EQ(fr.num_levels(), nlev);
  return compareFluxes(fr, dr, nlev, rtol, label);
}

// ============================================================================
// Test: Pure absorption (ssalb = 0)
// ============================================================================

TEST(DisortFluxSolver, PureAbsorption4) {
  DisortConfig cfg(3, 4);
  cfg.bc.direct_beam_flux = 1.0;
  cfg.bc.direct_beam_mu = 0.5;
  cfg.bc.surface_albedo = 0.0;
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < 3; ++lc) {
    cfg.delta_tau[lc] = 0.5;
    cfg.single_scat_albedo[lc] = 0.0;
    cfg.setIsotropic(lc);
  }
  EXPECT_TRUE(runComparison<4>(cfg, 1e-12, "PureAbsorption4"));
}

// ============================================================================
// Test: Conservative scattering (ssalb = 1)
// ============================================================================

TEST(DisortFluxSolver, ConservativeScattering4) {
  DisortConfig cfg(2, 4);
  cfg.bc.direct_beam_flux = 1.0;
  cfg.bc.direct_beam_mu = 0.5;
  cfg.bc.surface_albedo = 0.0;
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < 2; ++lc) {
    cfg.delta_tau[lc] = 1.0;
    cfg.single_scat_albedo[lc] = 1.0;
    cfg.setIsotropic(lc);
  }
  // Relaxed tolerance: conservative scattering (omega=1) uses fallback eigensolver
  EXPECT_TRUE(runComparison<4>(cfg, 1e-8, "ConservativeScattering4"));
}

// ============================================================================
// Test: Isotropic scattering with beam source (NStr=4, 8, 16)
// ============================================================================

TEST(DisortFluxSolver, IsotropicBeam4) {
  DisortConfig cfg(2, 4);
  cfg.bc.direct_beam_flux = M_PI;
  cfg.bc.direct_beam_mu = 0.6;
  cfg.bc.surface_albedo = 0.1;
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < 2; ++lc) {
    cfg.delta_tau[lc] = 0.25;
    cfg.single_scat_albedo[lc] = 0.9;
    cfg.setIsotropic(lc);
  }
  EXPECT_TRUE(runComparison<4>(cfg, 1e-12, "IsotropicBeam4"));
}

TEST(DisortFluxSolver, IsotropicBeam8) {
  DisortConfig cfg(2, 8);
  cfg.bc.direct_beam_flux = M_PI;
  cfg.bc.direct_beam_mu = 0.6;
  cfg.bc.surface_albedo = 0.1;
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < 2; ++lc) {
    cfg.delta_tau[lc] = 0.25;
    cfg.single_scat_albedo[lc] = 0.9;
    cfg.setIsotropic(lc);
  }
  EXPECT_TRUE(runComparison<8>(cfg, 1e-12, "IsotropicBeam8"));
}

TEST(DisortFluxSolver, IsotropicBeam16) {
  DisortConfig cfg(2, 16);
  cfg.bc.direct_beam_flux = M_PI;
  cfg.bc.direct_beam_mu = 0.6;
  cfg.bc.surface_albedo = 0.1;
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < 2; ++lc) {
    cfg.delta_tau[lc] = 0.25;
    cfg.single_scat_albedo[lc] = 0.9;
    cfg.setIsotropic(lc);
  }
  EXPECT_TRUE(runComparison<16>(cfg, 1e-12, "IsotropicBeam16"));
}

// ============================================================================
// Test: Henyey-Greenstein scattering (exercises delta-M)
// ============================================================================

TEST(DisortFluxSolver, HenyeyGreenstein4) {
  DisortConfig cfg(3, 4);
  cfg.bc.direct_beam_flux = 1.0;
  cfg.bc.direct_beam_mu = 0.5;
  cfg.bc.surface_albedo = 0.1;
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < 3; ++lc) {
    cfg.delta_tau[lc] = 0.2;
    cfg.single_scat_albedo[lc] = 0.9;
    cfg.setHenyeyGreenstein(0.75, lc);
  }
  EXPECT_TRUE(runComparison<4>(cfg, 1e-12, "HenyeyGreenstein4"));
}

TEST(DisortFluxSolver, HenyeyGreenstein16) {
  DisortConfig cfg(3, 16);
  cfg.bc.direct_beam_flux = 1.0;
  cfg.bc.direct_beam_mu = 0.5;
  cfg.bc.surface_albedo = 0.1;
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < 3; ++lc) {
    cfg.delta_tau[lc] = 0.2;
    cfg.single_scat_albedo[lc] = 0.9;
    cfg.setHenyeyGreenstein(0.75, lc);
  }
  EXPECT_TRUE(runComparison<16>(cfg, 1e-12, "HenyeyGreenstein16"));
}

// ============================================================================
// Test: Thermal emission only (no beam)
// ============================================================================

TEST(DisortFluxSolver, ThermalEmission4) {
  DisortConfig cfg(3, 4);
  cfg.bc.direct_beam_flux = 0.0;
  cfg.bc.surface_albedo = 0.0;
  cfg.flags.use_lambertian_surface = true;
  cfg.flags.use_thermal_emission = true;
  cfg.wavenumber_low = 500.0;
  cfg.wavenumber_high = 1000.0;
  cfg.bc.temperature_top = 200.0;
  cfg.bc.temperature_bottom = 300.0;
  cfg.bc.emissivity_top = 1.0;
  cfg.allocate();
  for (int lc = 0; lc < 3; ++lc) {
    cfg.delta_tau[lc] = 0.5;
    cfg.single_scat_albedo[lc] = 0.5;
    cfg.setIsotropic(lc);
  }
  // Temperature at each level (linearly interpolated)
  cfg.temperature[0] = 200.0;
  cfg.temperature[1] = 225.0;
  cfg.temperature[2] = 275.0;
  cfg.temperature[3] = 300.0;
  EXPECT_TRUE(runComparison<4>(cfg, 1e-12, "ThermalEmission4"));
}

TEST(DisortFluxSolver, ThermalEmission16) {
  DisortConfig cfg(3, 16);
  cfg.bc.direct_beam_flux = 0.0;
  cfg.bc.surface_albedo = 0.0;
  cfg.flags.use_lambertian_surface = true;
  cfg.flags.use_thermal_emission = true;
  cfg.wavenumber_low = 500.0;
  cfg.wavenumber_high = 1000.0;
  cfg.bc.temperature_top = 200.0;
  cfg.bc.temperature_bottom = 300.0;
  cfg.bc.emissivity_top = 1.0;
  cfg.allocate();
  for (int lc = 0; lc < 3; ++lc) {
    cfg.delta_tau[lc] = 0.5;
    cfg.single_scat_albedo[lc] = 0.5;
    cfg.setIsotropic(lc);
  }
  cfg.temperature[0] = 200.0;
  cfg.temperature[1] = 225.0;
  cfg.temperature[2] = 275.0;
  cfg.temperature[3] = 300.0;
  EXPECT_TRUE(runComparison<16>(cfg, 1e-12, "ThermalEmission16"));
}

// ============================================================================
// Test: Combined beam + thermal
// ============================================================================

TEST(DisortFluxSolver, BeamPlusThermal8) {
  DisortConfig cfg(4, 8);
  cfg.bc.direct_beam_flux = 1.0;
  cfg.bc.direct_beam_mu = 0.6;
  cfg.bc.surface_albedo = 0.2;
  cfg.flags.use_lambertian_surface = true;
  cfg.flags.use_thermal_emission = true;
  cfg.wavenumber_low = 800.0;
  cfg.wavenumber_high = 1200.0;
  cfg.bc.temperature_top = 250.0;
  cfg.bc.temperature_bottom = 310.0;
  cfg.bc.emissivity_top = 1.0;
  cfg.allocate();
  for (int lc = 0; lc < 4; ++lc) {
    cfg.delta_tau[lc] = 0.3;
    cfg.single_scat_albedo[lc] = 0.8;
    cfg.setHenyeyGreenstein(0.5, lc);
  }
  cfg.temperature[0] = 250.0;
  cfg.temperature[1] = 265.0;
  cfg.temperature[2] = 280.0;
  cfg.temperature[3] = 295.0;
  cfg.temperature[4] = 310.0;
  EXPECT_TRUE(runComparison<8>(cfg, 1e-12, "BeamPlusThermal8"));
}

// ============================================================================
// Test: Spherical beam geometry
// ============================================================================

TEST(DisortFluxSolver, SphericalBeam4) {
  DisortConfig cfg(5, 4);
  cfg.bc.direct_beam_flux = 1.0;
  cfg.bc.direct_beam_mu = 0.3;  // Grazing incidence
  cfg.bc.surface_albedo = 0.1;
  cfg.flags.use_lambertian_surface = true;
  cfg.flags.use_spherical_beam = true;
  cfg.bottom_radius = 6371.0;  // Earth radius in km
  cfg.allocate();

  // Setup altitudes (decreasing from TOA to BOA)
  cfg.level_altitudes[0] = 100.0;
  cfg.level_altitudes[1] = 80.0;
  cfg.level_altitudes[2] = 60.0;
  cfg.level_altitudes[3] = 40.0;
  cfg.level_altitudes[4] = 20.0;
  cfg.level_altitudes[5] = 0.0;

  for (int lc = 0; lc < 5; ++lc) {
    cfg.delta_tau[lc] = 0.1;
    cfg.single_scat_albedo[lc] = 0.9;
    cfg.setIsotropic(lc);
  }
  EXPECT_TRUE(runComparison<4>(cfg, 1e-12, "SphericalBeam4"));
}

TEST(DisortFluxSolver, SphericalBeam16) {
  DisortConfig cfg(5, 16);
  cfg.bc.direct_beam_flux = 1.0;
  cfg.bc.direct_beam_mu = 0.3;
  cfg.bc.surface_albedo = 0.1;
  cfg.flags.use_lambertian_surface = true;
  cfg.flags.use_spherical_beam = true;
  cfg.bottom_radius = 6371.0;
  cfg.allocate();

  cfg.level_altitudes[0] = 100.0;
  cfg.level_altitudes[1] = 80.0;
  cfg.level_altitudes[2] = 60.0;
  cfg.level_altitudes[3] = 40.0;
  cfg.level_altitudes[4] = 20.0;
  cfg.level_altitudes[5] = 0.0;

  for (int lc = 0; lc < 5; ++lc) {
    cfg.delta_tau[lc] = 0.1;
    cfg.single_scat_albedo[lc] = 0.9;
    cfg.setIsotropic(lc);
  }
  EXPECT_TRUE(runComparison<16>(cfg, 1e-12, "SphericalBeam16"));
}

// ============================================================================
// Test: Multi-layer problem (10 layers)
// ============================================================================

TEST(DisortFluxSolver, MultiLayer8) {
  const int nlyr = 10;
  DisortConfig cfg(nlyr, 8);
  cfg.bc.direct_beam_flux = M_PI;
  cfg.bc.direct_beam_mu = 0.7;
  cfg.bc.surface_albedo = 0.3;
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < nlyr; ++lc) {
    cfg.delta_tau[lc] = 0.05 * (lc + 1);  // Increasing optical depth
    cfg.single_scat_albedo[lc] = 0.9 - 0.05 * lc;  // Decreasing SSA
    cfg.setHenyeyGreenstein(0.7, lc);
  }
  EXPECT_TRUE(runComparison<8>(cfg, 1e-12, "MultiLayer8"));
}

// ============================================================================
// Test: Reflective surface (surface_albedo > 0)
// ============================================================================

TEST(DisortFluxSolver, ReflectiveSurface4) {
  DisortConfig cfg(2, 4);
  cfg.bc.direct_beam_flux = 1.0;
  cfg.bc.direct_beam_mu = 0.5;
  cfg.bc.surface_albedo = 0.8;  // Highly reflective
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < 2; ++lc) {
    cfg.delta_tau[lc] = 0.1;
    cfg.single_scat_albedo[lc] = 0.9;
    cfg.setIsotropic(lc);
  }
  EXPECT_TRUE(runComparison<4>(cfg, 1e-12, "ReflectiveSurface4"));
}

// ============================================================================
// Test: Repeated calls (allocation cache)
// ============================================================================

TEST(DisortFluxSolver, RepeatedCalls) {
  DisortFluxConfig cfg(2, 4);
  cfg.direct_beam_flux = 1.0;
  cfg.direct_beam_mu = 0.5;
  cfg.surface_albedo = 0.0;
  cfg.allocate();
  for (int lc = 0; lc < 2; ++lc) {
    cfg.delta_tau[lc] = 0.3;
    cfg.single_scat_albedo[lc] = 0.9;
    cfg.setIsotropic(lc);
  }

  DisortFluxSolver<4> fs;

  // First call
  FluxResult r1 = fs.solve(cfg);
  EXPECT_EQ(r1.num_levels(), 3);

  // Second call (should reuse allocation)
  FluxResult r2 = fs.solve(cfg);
  EXPECT_EQ(r2.num_levels(), 3);

  // Results should be identical
  for (int lu = 0; lu < 3; ++lu) {
    EXPECT_NEAR(r1.flux_up[lu], r2.flux_up[lu], 1e-15);
    EXPECT_NEAR(r1.flux_down[lu], r2.flux_down[lu], 1e-15);
    EXPECT_NEAR(r1.flux_direct_beam[lu], r2.flux_direct_beam[lu], 1e-15);
    EXPECT_NEAR(r1.mean_intensity[lu], r2.mean_intensity[lu], 1e-15);
  }
}

// ============================================================================
// Test: NStr=2 is rejected (must use dedicated two-stream solver)
// ============================================================================

TEST(DisortFluxSolver, RejectNStr2) {
  EXPECT_THROW(DisortFluxConfig(2, 2), std::invalid_argument);
  EXPECT_THROW(DisortConfig(2, 2), std::invalid_argument);
}

// ============================================================================
// Test: NStr=32 (large stream count)
// ============================================================================

TEST(DisortFluxSolver, LargeStreams32) {
  DisortConfig cfg(2, 32);
  cfg.bc.direct_beam_flux = 1.0;
  cfg.bc.direct_beam_mu = 0.5;
  cfg.bc.surface_albedo = 0.1;
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < 2; ++lc) {
    cfg.delta_tau[lc] = 0.3;
    cfg.single_scat_albedo[lc] = 0.8;
    cfg.setHenyeyGreenstein(0.6, lc);
  }
  EXPECT_TRUE(runComparison<32>(cfg, 1e-11, "LargeStreams32"));
}

// ============================================================================
// Test: Validation errors
// ============================================================================

TEST(DisortFluxSolver, WrongNStr) {
  DisortFluxConfig cfg(2, 8);  // 8 streams
  cfg.direct_beam_flux = 1.0;
  cfg.direct_beam_mu = 0.5;
  cfg.allocate();
  for (int lc = 0; lc < 2; ++lc) {
    cfg.delta_tau[lc] = 0.1;
    cfg.single_scat_albedo[lc] = 0.5;
    cfg.setIsotropic(lc);
  }

  DisortFluxSolver<4> fs;  // Template NStr=4, but config has num_streams=8
  EXPECT_THROW(fs.solve(cfg), std::invalid_argument);
}

// ============================================================================
// Test: Isotropic flux top boundary
// ============================================================================

TEST(DisortFluxSolver, IsotropicFluxTop4) {
  DisortConfig cfg(2, 4);
  cfg.bc.direct_beam_flux = 0.0;
  cfg.bc.isotropic_flux_top = 1.0;
  cfg.bc.surface_albedo = 0.0;
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < 2; ++lc) {
    cfg.delta_tau[lc] = 0.5;
    cfg.single_scat_albedo[lc] = 0.9;
    cfg.setIsotropic(lc);
  }
  EXPECT_TRUE(runComparison<4>(cfg, 1e-12, "IsotropicFluxTop4"));
}

// ============================================================================
// Test: Optically thick atmosphere (layer cutoff)
// ============================================================================

TEST(DisortFluxSolver, OpticallyThick4) {
  DisortConfig cfg(5, 4);
  cfg.bc.direct_beam_flux = 1.0;
  cfg.bc.direct_beam_mu = 0.5;
  cfg.bc.surface_albedo = 0.0;
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < 5; ++lc) {
    cfg.delta_tau[lc] = 10.0;  // Very thick layers → will trigger lyrcut
    cfg.single_scat_albedo[lc] = 0.1;  // Mostly absorbing
    cfg.setIsotropic(lc);
  }
  EXPECT_TRUE(runComparison<4>(cfg, 1e-12, "OpticallyThick4"));
}

// ============================================================================
// Test: Rayleigh scattering
// ============================================================================

TEST(DisortFluxSolver, Rayleigh8) {
  DisortConfig cfg(3, 8);
  cfg.bc.direct_beam_flux = M_PI;
  cfg.bc.direct_beam_mu = 0.5;
  cfg.bc.surface_albedo = 0.0;
  cfg.flags.use_lambertian_surface = true;
  cfg.allocate();
  for (int lc = 0; lc < 3; ++lc) {
    cfg.delta_tau[lc] = 0.1;
    cfg.single_scat_albedo[lc] = 1.0;
    cfg.setRayleigh(lc);
  }
  // Rayleigh (conservative) scattering can amplify small eigensolver differences
  // between fixed-size and dynamic Eigen paths, so use slightly relaxed tolerance.
  // Agreement is still ~9 significant digits.
  EXPECT_TRUE(runComparison<8>(cfg, 1e-8, "Rayleigh8"));
}
