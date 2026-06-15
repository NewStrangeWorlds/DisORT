#include "testing.hpp"
#include "DisortSolver.hpp"
#include "DisortConfig.hpp"
#include "FluxSolver.hpp"
#include "DisortFluxConfig.hpp"
#include "Planck.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

using namespace disortpp;

// ============================================================================
// Helpers
// ============================================================================

// Build a thermal-emission test problem. Level temperatures Tlev has size
// nlyr+1 (levels 0..nlyr); Tsurf is the surface (bottom-boundary) temperature.
static DisortConfig buildThermalConfig(const std::vector<double>& Tlev, double Tsurf,
                                       int nstr, double ssa, double beam_flux = 0.0,
                                       bool spherical = false) {
  const int nlyr = static_cast<int>(Tlev.size()) - 1;
  DisortConfig config(nlyr, nstr);

  config.flags.use_thermal_emission = true;
  config.flags.use_user_tau         = false;   // output at layer boundaries (ntau = nlyr+1)
  config.flags.use_user_mu          = false;
  config.flags.comp_only_fluxes     = false;
  config.flags.use_lambertian_surface = true;
  config.flags.use_spherical_beam   = spherical;
  config.flags.ibcnd                = BoundaryConditionType::General;

  config.bc.direct_beam_flux  = beam_flux;
  config.bc.direct_beam_mu    = 0.5;
  config.bc.direct_beam_phi   = 0.0;
  config.bc.surface_albedo    = 0.1;
  config.bc.temperature_bottom = Tsurf;
  config.bc.temperature_top    = 0.0;
  config.bc.emissivity_top     = 1.0;

  config.wavenumber_low  = 500.0;
  config.wavenumber_high = 800.0;
  config.accuracy_fourier_series = 1e-4;
  config.num_phi = 1;

  if (spherical) {
    config.bottom_radius = 6.371e6;
  }

  config.allocate();
  config.phi_user[0] = 0.0;

  for (int lc = 0; lc < nlyr; ++lc) {
    config.delta_tau[lc]          = 0.4;
    config.single_scat_albedo[lc] = ssa;
    config.phaseFunctionMoments(0, lc) = 1.0;
    if (config.nmomNstr() >= 1) config.phaseFunctionMoments(1, lc) = 0.3 * ssa;
  }
  for (int lev = 0; lev <= nlyr; ++lev) config.temperature[lev] = Tlev[lev];

  if (spherical) {
    // Simple equally-spaced altitudes (only used for the beam path, which is
    // inactive here; present so allocate()/validate() are satisfied).
    for (int lev = 0; lev <= nlyr; ++lev)
      config.level_altitudes[lev] = 1.0e5 * (nlyr - lev);
  }

  return config;
}

// Validate the analytic temperature Jacobian against a central finite difference
// of the forward solve, over every temperature DOF and every output level.
static void checkThermalJacobian(const std::vector<double>& Tlev, double Tsurf,
                                 int nstr, double ssa, double beam_flux,
                                 double rtol, double atol) {
  const int nlyr = static_cast<int>(Tlev.size()) - 1;
  const int ndof = nlyr + 2;     // levels 0..nlyr + surface

  DisortConfig cfg = buildThermalConfig(Tlev, Tsurf, nstr, ssa, beam_flux);
  cfg.flags.compute_temperature_jacobian = true;
  DisortSolver solver;
  DisortResult res = solver.solve(cfg);
  const int ntau = res.num_user_tau();

  for (int dof = 0; dof < ndof; ++dof) {
    const double T0 = (dof <= nlyr) ? Tlev[dof] : Tsurf;
    const double h  = 1e-3 * std::max(std::abs(T0), 1.0);

    std::vector<double> Tp = Tlev, Tm = Tlev;
    double Tsp = Tsurf, Tsm = Tsurf;
    if (dof <= nlyr) { Tp[dof] += h; Tm[dof] -= h; }
    else             { Tsp += h;     Tsm -= h;     }

    DisortConfig cp = buildThermalConfig(Tp, Tsp, nstr, ssa, beam_flux);
    DisortConfig cm = buildThermalConfig(Tm, Tsm, nstr, ssa, beam_flux);
    DisortSolver sp, sm;
    DisortResult rp = sp.solve(cp);
    DisortResult rm = sm.solve(cm);

    for (int lu = 0; lu < ntau; ++lu) {
      auto chk = [&](double ana, double fp, double fm) {
        const double fd = (fp - fm) / (2.0 * h);
        EXPECT_NEAR(ana, fd, rtol * std::max(std::abs(fd), std::abs(ana)) + atol);
      };
      chk(res.flux_up_temperature_jac[lu][dof],             rp.flux_up[lu],             rm.flux_up[lu]);
      chk(res.flux_down_temperature_jac[lu][dof],           rp.flux_down[lu],           rm.flux_down[lu]);
      chk(res.mean_intensity_temperature_jac[lu][dof],      rp.mean_intensity[lu],      rm.mean_intensity[lu]);
      chk(res.mean_intensity_down_temperature_jac[lu][dof], rp.mean_intensity_down[lu], rm.mean_intensity_down[lu]);
      chk(res.mean_intensity_up_temperature_jac[lu][dof],   rp.mean_intensity_up[lu],   rm.mean_intensity_up[lu]);
      chk(res.flux_divergence_temperature_jac[lu][dof],     rp.flux_tau_divergence[lu], rm.flux_tau_divergence[lu]);
    }
  }
}

// --- Flux-only solver (DisortFluxSolver) thermal config + FD check -----------

static DisortFluxConfig buildFluxThermalConfig(const std::vector<double>& Tlev, double Tsurf,
                                               int nstr, double ssa, double beam_flux = 0.0) {
  const int nlyr = static_cast<int>(Tlev.size()) - 1;
  DisortFluxConfig c(nlyr, nstr);

  c.use_thermal_emission = true;
  c.direct_beam_flux  = beam_flux;
  c.direct_beam_mu    = 0.5;
  c.surface_albedo    = 0.1;
  c.temperature_bottom = Tsurf;
  c.temperature_top    = 0.0;
  c.emissivity_top     = 1.0;
  c.wavenumber_low  = 500.0;
  c.wavenumber_high = 800.0;

  c.allocate();
  for (int lc = 0; lc < nlyr; ++lc) {
    c.delta_tau[lc]          = 0.4;
    c.single_scat_albedo[lc] = ssa;
    c.phaseFunctionMoments(0, lc) = 1.0;
    if (c.nmomNstr() >= 1) c.phaseFunctionMoments(1, lc) = 0.3 * ssa;
  }
  for (int lev = 0; lev <= nlyr; ++lev) c.temperature[lev] = Tlev[lev];
  return c;
}

template<int NStr>
static void checkFluxThermalJacobian(const std::vector<double>& Tlev, double Tsurf,
                                     double ssa, double beam_flux, double rtol, double atol) {
  const int nlyr = static_cast<int>(Tlev.size()) - 1;
  const int ndof = nlyr + 2;

  DisortFluxConfig cfg = buildFluxThermalConfig(Tlev, Tsurf, NStr, ssa, beam_flux);
  cfg.compute_temperature_jacobian = true;
  DisortFluxSolver<NStr> solver;
  FluxResult res = solver.solve(cfg);
  const int ntau = res.num_levels();

  for (int dof = 0; dof < ndof; ++dof) {
    const double T0 = (dof <= nlyr) ? Tlev[dof] : Tsurf;
    const double h  = 1e-3 * std::max(std::abs(T0), 1.0);
    std::vector<double> Tp = Tlev, Tm = Tlev;
    double Tsp = Tsurf, Tsm = Tsurf;
    if (dof <= nlyr) { Tp[dof] += h; Tm[dof] -= h; }
    else             { Tsp += h;     Tsm -= h;     }

    DisortFluxConfig cp = buildFluxThermalConfig(Tp, Tsp, NStr, ssa, beam_flux);
    DisortFluxConfig cm = buildFluxThermalConfig(Tm, Tsm, NStr, ssa, beam_flux);
    DisortFluxSolver<NStr> sp, sm;
    FluxResult rp = sp.solve(cp);
    FluxResult rm = sm.solve(cm);

    for (int lu = 0; lu < ntau; ++lu) {
      auto chk = [&](double ana, double fp, double fm) {
        const double fd = (fp - fm) / (2.0 * h);
        EXPECT_NEAR(ana, fd, rtol * std::max(std::abs(fd), std::abs(ana)) + atol);
      };
      chk(res.flux_up_temperature_jac[lu][dof],         rp.flux_up[lu],             rm.flux_up[lu]);
      chk(res.flux_down_temperature_jac[lu][dof],       rp.flux_down[lu],           rm.flux_down[lu]);
      chk(res.mean_intensity_temperature_jac[lu][dof],  rp.mean_intensity[lu],      rm.mean_intensity[lu]);
      chk(res.flux_divergence_temperature_jac[lu][dof], rp.flux_tau_divergence[lu], rm.flux_tau_divergence[lu]);
    }
  }
}

// ============================================================================
// 1. Planck temperature derivative
// ============================================================================

TEST(JacobianTest, PlanckDerivVsFD) {
  struct Band { double lo, hi; };
  const Band bands[] = {{500.0, 800.0}, {100.0, 110.0}, {1.0, 3000.0}, {2000.0, 2001.0}};
  const double temps[] = {50.0, 150.0, 300.0, 1200.0};

  for (const auto& b : bands) {
    for (double T : temps) {
      const double h  = 1e-4 * T;
      const double fd = (planckFunction2(b.lo, b.hi, T + h) -
                         planckFunction2(b.lo, b.hi, T - h)) / (2.0 * h);
      const double ana = planckFunctionDeriv2(b.lo, b.hi, T);
      EXPECT_NEAR(ana, fd, 1e-5 * std::max(std::abs(fd), std::abs(ana)) + 1e-12);
    }
  }
}

// ============================================================================
// 2. Single-layer thermal
// ============================================================================

TEST(JacobianTest, SingleLayerThermal) {
  checkThermalJacobian({/*Tlev*/ 200.0, 260.0}, /*Tsurf*/ 280.0,
                       /*nstr*/ 8, /*ssa*/ 0.0, /*beam*/ 0.0,
                       /*rtol*/ 1e-3, /*atol*/ 1e-9);
}

// ============================================================================
// 3. Multi-layer thermal profile (full Jacobian vs FD)
// ============================================================================

TEST(JacobianTest, MultiLayerThermalProfile) {
  std::vector<double> Tlev;
  for (int i = 0; i <= 10; ++i) Tlev.push_back(200.0 + 8.0 * i);   // 200..280 K
  checkThermalJacobian(Tlev, /*Tsurf*/ 285.0,
                       /*nstr*/ 16, /*ssa*/ 0.0, /*beam*/ 0.0,
                       /*rtol*/ 1e-3, /*atol*/ 1e-9);
}

// ============================================================================
// 4. Thermal with scattering
// ============================================================================

TEST(JacobianTest, ThermalWithScatteringHalf) {
  std::vector<double> Tlev = {210.0, 235.0, 255.0, 270.0, 280.0};
  checkThermalJacobian(Tlev, 290.0, /*nstr*/ 16, /*ssa*/ 0.5, /*beam*/ 0.0,
                       /*rtol*/ 1.5e-3, /*atol*/ 1e-9);
}

TEST(JacobianTest, ThermalWithScatteringHigh) {
  std::vector<double> Tlev = {210.0, 235.0, 255.0, 270.0, 280.0};
  checkThermalJacobian(Tlev, 290.0, /*nstr*/ 16, /*ssa*/ 0.9, /*beam*/ 0.0,
                       /*rtol*/ 2e-3, /*atol*/ 1e-9);
}

// ============================================================================
// 5. Surface temperature Jacobian (exercised together with a direct beam)
// ============================================================================

TEST(JacobianTest, ThermalPlusBeam) {
  std::vector<double> Tlev = {200.0, 240.0, 270.0};
  checkThermalJacobian(Tlev, /*Tsurf*/ 288.0, /*nstr*/ 16, /*ssa*/ 0.6,
                       /*beam*/ 1.0, /*rtol*/ 1.5e-3, /*atol*/ 1e-9);
}

// ============================================================================
// 6. Pseudo-spherical thermal Jacobian equals plane-parallel
// ============================================================================

TEST(JacobianTest, SphericalEqualsPlaneParallelThermal) {
  std::vector<double> Tlev = {210.0, 240.0, 265.0, 282.0};
  const double Tsurf = 290.0;
  const int nstr = 16;
  const double ssa = 0.4;

  DisortConfig cpp_ = buildThermalConfig(Tlev, Tsurf, nstr, ssa, 0.0, /*spherical*/ false);
  DisortConfig csp_ = buildThermalConfig(Tlev, Tsurf, nstr, ssa, 0.0, /*spherical*/ true);
  cpp_.flags.compute_temperature_jacobian = true;
  csp_.flags.compute_temperature_jacobian = true;

  DisortSolver s1, s2;
  DisortResult rp = s1.solve(cpp_);
  DisortResult rs = s2.solve(csp_);

  const int ntau = rp.num_user_tau();
  const int ndof = static_cast<int>(Tlev.size()) + 1;   // nlyr+2
  for (int lu = 0; lu < ntau; ++lu) {
    for (int d = 0; d < ndof; ++d) {
      EXPECT_NEAR(rp.flux_up_temperature_jac[lu][d],         rs.flux_up_temperature_jac[lu][d],         1e-10);
      EXPECT_NEAR(rp.flux_down_temperature_jac[lu][d],       rs.flux_down_temperature_jac[lu][d],       1e-10);
      EXPECT_NEAR(rp.flux_divergence_temperature_jac[lu][d], rs.flux_divergence_temperature_jac[lu][d], 1e-10);
    }
  }
}

// ============================================================================
// 8. Off by default: zero overhead, empty Jacobian arrays
// ============================================================================

TEST(JacobianTest, JacobianOffByDefault) {
  std::vector<double> Tlev = {220.0, 250.0, 275.0};
  DisortConfig cfg = buildThermalConfig(Tlev, 285.0, 8, 0.0, 0.0);
  // compute_temperature_jacobian defaults to false
  EXPECT_FALSE(cfg.flags.compute_temperature_jacobian);

  DisortSolver solver;
  DisortResult res = solver.solve(cfg);

  EXPECT_TRUE(res.flux_up_temperature_jac.empty());
  EXPECT_TRUE(res.flux_down_temperature_jac.empty());
  EXPECT_TRUE(res.mean_intensity_temperature_jac.empty());
  EXPECT_TRUE(res.flux_divergence_temperature_jac.empty());

  // Enabling the flag must not change the forward fluxes.
  DisortConfig cfg2 = buildThermalConfig(Tlev, 285.0, 8, 0.0, 0.0);
  cfg2.flags.compute_temperature_jacobian = true;
  DisortSolver solver2;
  DisortResult res2 = solver2.solve(cfg2);

  for (int lu = 0; lu < res.num_user_tau(); ++lu) {
    EXPECT_NEAR(res.flux_up[lu],             res2.flux_up[lu],             1e-12);
    EXPECT_NEAR(res.flux_down[lu],           res2.flux_down[lu],           1e-12);
    EXPECT_NEAR(res.flux_tau_divergence[lu], res2.flux_tau_divergence[lu], 1e-12);
  }
  EXPECT_FALSE(res2.flux_up_temperature_jac.empty());
}

// ============================================================================
// index_from_bottom: Jacobians reversed consistently with the flux outputs
// ============================================================================

TEST(JacobianTest, IndexFromBottomReversal) {
  std::vector<double> Tlev = {210.0, 240.0, 265.0, 285.0};  // TOA -> BOA
  const double Tsurf = 295.0;
  const int nstr = 16;
  const double ssa = 0.3;
  const int nlyr = static_cast<int>(Tlev.size()) - 1;

  // Reference: native top-down ordering.
  DisortConfig c0 = buildThermalConfig(Tlev, Tsurf, nstr, ssa, 0.0);
  c0.flags.compute_temperature_jacobian = true;
  DisortSolver s0;
  DisortResult r0 = s0.solve(c0);

  // Bottom-up: same physical problem, inputs supplied BOA->TOA + index_from_bottom.
  // (Layer optical properties are uniform here, so only the temperature ordering
  //  actually flips.)
  std::vector<double> Tlev_bu(Tlev.rbegin(), Tlev.rend());
  DisortConfig c1 = buildThermalConfig(Tlev_bu, Tsurf, nstr, ssa, 0.0);
  c1.flags.index_from_bottom = true;
  c1.flags.compute_temperature_jacobian = true;
  DisortSolver s1;
  DisortResult r1 = s1.solve(c1);

  const int ntau = r0.num_user_tau();
  const int ndof = nlyr + 2;
  for (int lu = 0; lu < ntau; ++lu) {
    for (int dof = 0; dof < ndof; ++dof) {
      const int lu0  = ntau - 1 - lu;                       // reversed row
      const int dof0 = (dof <= nlyr) ? (nlyr - dof) : dof;  // reversed level col; surface stays last
      auto cmp = [&](const std::vector<std::vector<double>>& A,
                     const std::vector<std::vector<double>>& B) {
        EXPECT_NEAR(A[lu][dof], B[lu0][dof0],
                    1e-9 * std::max(std::abs(A[lu][dof]), std::abs(B[lu0][dof0])) + 1e-12);
      };
      cmp(r1.flux_up_temperature_jac,         r0.flux_up_temperature_jac);
      cmp(r1.flux_down_temperature_jac,       r0.flux_down_temperature_jac);
      cmp(r1.flux_divergence_temperature_jac, r0.flux_divergence_temperature_jac);
    }
  }
}

// ============================================================================
// 7. DisortFluxSolver temperature Jacobian
// ============================================================================

TEST(JacobianTest, FluxSolverSingleLayerThermal) {
  checkFluxThermalJacobian<8>({200.0, 260.0}, 280.0, /*ssa*/ 0.0, /*beam*/ 0.0,
                              /*rtol*/ 1e-3, /*atol*/ 1e-9);
}

TEST(JacobianTest, FluxSolverMultiLayerScattering) {
  std::vector<double> Tlev;
  for (int i = 0; i <= 8; ++i) Tlev.push_back(205.0 + 9.0 * i);
  checkFluxThermalJacobian<16>(Tlev, 290.0, /*ssa*/ 0.7, /*beam*/ 0.0,
                               /*rtol*/ 2e-3, /*atol*/ 1e-9);
}

TEST(JacobianTest, FluxSolverThermalPlusBeam) {
  std::vector<double> Tlev = {210.0, 245.0, 275.0};
  checkFluxThermalJacobian<16>(Tlev, 288.0, /*ssa*/ 0.5, /*beam*/ 1.0,
                               /*rtol*/ 1.5e-3, /*atol*/ 1e-9);
}

// DisortFluxSolver and DisortSolver must agree on the temperature Jacobian for a
// shared thermal problem (independent implementations cross-check).
TEST(JacobianTest, FluxSolverMatchesDisortSolver) {
  std::vector<double> Tlev = {210.0, 240.0, 265.0, 282.0};
  const double Tsurf = 290.0;
  const int nstr = 16;
  const double ssa = 0.0;   // ssa=0 minimises eigensolver discrepancy between solvers

  DisortConfig cfg_full = buildThermalConfig(Tlev, Tsurf, nstr, ssa, 0.0);
  cfg_full.flags.compute_temperature_jacobian = true;
  DisortSolver full;
  DisortResult rf = full.solve(cfg_full);

  DisortFluxConfig cfg_flux = buildFluxThermalConfig(Tlev, Tsurf, nstr, ssa, 0.0);
  cfg_flux.compute_temperature_jacobian = true;
  DisortFluxSolver<nstr> flux;
  FluxResult rx = flux.solve(cfg_flux);

  const int ntau = rx.num_levels();
  const int ndof = static_cast<int>(Tlev.size()) + 1;   // nlyr+2
  for (int lu = 0; lu < ntau; ++lu) {
    for (int d = 0; d < ndof; ++d) {
      auto cmp = [&](double a, double b) {
        EXPECT_NEAR(a, b, 1e-6 * std::max(std::abs(a), std::abs(b)) + 1e-9);
      };
      cmp(rf.flux_up_temperature_jac[lu][d],         rx.flux_up_temperature_jac[lu][d]);
      cmp(rf.flux_down_temperature_jac[lu][d],       rx.flux_down_temperature_jac[lu][d]);
      cmp(rf.flux_divergence_temperature_jac[lu][d], rx.flux_divergence_temperature_jac[lu][d]);
    }
  }
}
