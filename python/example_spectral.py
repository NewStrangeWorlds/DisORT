#!/usr/bin/env python3
"""
DisORT++ Python Interface — Spectral Loop Example

Demonstrates the spectral loop functions that run the wavenumber iteration
entirely in C++, avoiding per-call Python overhead:

  1. Thermal emission spectrum (solve_spectral_bands)
  2. Wavelength-dependent scattering with Python loop
  3. Performance comparison (C++ loop vs Python loop)
  4. Per-wavelength overrides with solve_flux_spectral_bands
  5. Thermal emission with per-wavelength overrides (solve_spectral_bands)

Examples 4 and 5 show how to pass wavelength-dependent arrays (delta_tau,
single_scat_albedo, surface_albedo, emissivity_top, etc.) directly to
the C++ spectral loop, so the entire computation stays in C++.

Build the Python module first:
    cd DisORT/build
    cmake .. -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)

Then run this script from the build directory:
    python3 ../python/example_spectral.py
"""

import sys
import os

# Add the build directory to the path so we can import the module
build_dir = os.path.join(os.path.dirname(__file__), "..", "build")
sys.path.insert(0, os.path.abspath(build_dir))

import disortpp
import numpy as np


# ============================================================================
# Example 1: Thermal emission spectrum with the general solver
# ============================================================================

def example_thermal_spectrum():
    """
    Compute the upward thermal emission spectrum of a 4-layer atmosphere
    across the infrared, from 100 to 2000 cm^-1 in 50 cm^-1 bands.
    """
    print("=" * 70)
    print("Example 1: Thermal emission spectrum (DisortSolver)")
    print("=" * 70)

    nlyr = 4
    nstr = 8

    cfg = disortpp.DisortConfig(nlyr, nstr)
    cfg.flags.use_lambertian_surface = True
    cfg.flags.use_thermal_emission = True
    cfg.flags.comp_only_fluxes = True
    cfg.allocate()

    # Optical properties (absorption-dominated atmosphere)
    cfg.delta_tau = [0.2, 0.5, 1.0, 2.0]
    cfg.single_scat_albedo = [0.3, 0.3, 0.3, 0.3]
    cfg.set_isotropic()

    # Temperature profile (warm surface, cold TOA)
    cfg.temperature = [180.0, 210.0, 240.0, 270.0, 300.0]

    # Boundary conditions
    cfg.bc.direct_beam_flux = 0.0  # no stellar beam (nightside)
    cfg.bc.temperature_top = 180.0
    cfg.bc.temperature_bottom = 300.0
    cfg.bc.surface_albedo = 0.0  # perfect emitter

    # Define wavenumber bands: 100-2000 cm^-1 in 50 cm^-1 steps
    band_width = 50.0
    wn_edges = np.arange(100.0, 2001.0, band_width)
    bands = [(wn_edges[i], wn_edges[i + 1]) for i in range(len(wn_edges) - 1)]
    band_centers = np.array([(lo + hi) / 2 for lo, hi in bands])

    # Solve all bands in one C++ call
    results = disortpp.solve_spectral_bands(cfg, bands)

    # Extract fluxes
    flux_up_toa = np.array([r.flux_up[0] for r in results])
    flux_down_boa = np.array([r.flux_down[-1] for r in results])
    flux_up_boa = np.array([r.flux_up[-1] for r in results])

    # Print a selection of results
    print(f"\n  {len(bands)} wavenumber bands from {wn_edges[0]:.0f} "
          f"to {wn_edges[-1]:.0f} cm^-1")
    print(f"\n  {'Band [cm^-1]':>16}  {'Flux Up TOA':>12}  "
          f"{'Flux Down BOA':>14}  {'Flux Up BOA':>12}")
    print("  " + "-" * 58)

    # Print every 4th band
    for i in range(0, len(bands), 4):
        lo, hi = bands[i]
        print(f"  {lo:7.0f}-{hi:5.0f}  {flux_up_toa[i]:12.4f}  "
              f"{flux_down_boa[i]:14.4f}  {flux_up_boa[i]:12.4f}")

    # Total integrated flux (sum over all bands)
    total_up_toa = np.sum(flux_up_toa)
    total_up_boa = np.sum(flux_up_boa)
    print(f"\n  Total upward flux at TOA: {total_up_toa:.4f}")
    print(f"  Total upward flux at BOA: {total_up_boa:.4f}")
    print()


# ============================================================================
# Example 2: Wavelength-dependent scattering with the flux solver
# ============================================================================

def example_scattering_spectrum():
    """
    Compute fluxes for an atmosphere where the optical depth varies with
    wavenumber (e.g. Rayleigh scattering, where tau ~ wn^4).

    Uses the fast DisortFluxSolver via solve_flux_spectral_bands,
    updating the optical properties for each band.
    """
    print("=" * 70)
    print("Example 2: Wavelength-dependent scattering (DisortFluxSolver)")
    print("=" * 70)

    nlyr = 3
    nstr = 16

    # Wavenumber grid: 5000-25000 cm^-1 (0.4-2.0 um) in 500 cm^-1 bands
    band_width = 500.0
    wn_edges = np.arange(5000.0, 25001.0, band_width)
    bands = [(wn_edges[i], wn_edges[i + 1]) for i in range(len(wn_edges) - 1)]
    band_centers = np.array([(lo + hi) / 2 for lo, hi in bands])

    # Since optical properties change with wavenumber, we solve each band
    # individually using the spectral loop with per-band config updates.
    # For this case, we loop in Python but still benefit from solver reuse.
    solver = disortpp.create_flux_solver(nstr)

    flux_up_toa = np.zeros(len(bands))
    flux_down_boa = np.zeros(len(bands))

    # Reference optical depth at 10000 cm^-1
    tau_ref = np.array([0.05, 0.1, 0.2])
    wn_ref = 10000.0

    for i, (wn_lo, wn_hi) in enumerate(bands):
        wn_center = (wn_lo + wn_hi) / 2

        cfg = disortpp.DisortFluxConfig(nlyr, nstr)
        cfg.direct_beam_flux = 1.0
        cfg.direct_beam_mu = 0.5
        cfg.surface_albedo = 0.1
        cfg.wavenumber_low = wn_lo
        cfg.wavenumber_high = wn_hi
        cfg.allocate()

        # Rayleigh-like scaling: tau ~ (wn/wn_ref)^4
        scale = (wn_center / wn_ref) ** 4
        cfg.delta_tau = list(tau_ref * scale)
        cfg.single_scat_albedo = [1.0, 1.0, 1.0]  # pure scattering
        cfg.set_rayleigh()

        result = solver.solve(cfg)
        flux_up_toa[i] = result.flux_up[0]
        flux_down_boa[i] = result.total_flux_down(nlyr)

    # Convert wavenumber to wavelength for display
    wavelength_um = 1e4 / band_centers  # cm^-1 -> um

    print(f"\n  {len(bands)} wavenumber bands from {wn_edges[0]:.0f} "
          f"to {wn_edges[-1]:.0f} cm^-1")
    print(f"\n  {'Wavelength [um]':>16}  {'Wn [cm^-1]':>12}  "
          f"{'Flux Up TOA':>12}  {'Flux Down BOA':>14}")
    print("  " + "-" * 58)

    for i in range(0, len(bands), 4):
        print(f"  {wavelength_um[i]:16.3f}  {band_centers[i]:12.0f}  "
              f"{flux_up_toa[i]:12.6f}  {flux_down_boa[i]:14.6f}")

    # Planetary albedo spectrum
    flux_in = 1.0 * 0.5  # fbeam * mu0 = incident flux
    albedo = flux_up_toa / flux_in

    print(f"\n  Planetary albedo at 2.0 um (wn= 5000): {albedo[0]:.4f}")
    print(f"  Planetary albedo at 0.5 um (wn=20000): {albedo[-4]:.4f}")
    print(f"  (Rayleigh scattering is stronger at shorter wavelengths)")
    print()


# ============================================================================
# Example 3: Comparing C++ loop vs Python loop performance
# ============================================================================

def example_performance():
    """
    Compare the performance of the C++ spectral loop (solve_spectral_bands)
    against an equivalent Python loop calling solver.solve() repeatedly.
    """
    import time

    print("=" * 70)
    print("Example 3: Performance comparison (C++ loop vs Python loop)")
    print("=" * 70)

    nlyr = 6
    nstr = 16

    # Build a thermal config
    cfg = disortpp.DisortConfig(nlyr, nstr)
    cfg.flags.use_lambertian_surface = True
    cfg.flags.use_thermal_emission = True
    cfg.flags.comp_only_fluxes = True
    cfg.allocate()

    cfg.delta_tau = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    cfg.single_scat_albedo = [0.5] * nlyr
    cfg.set_henyey_greenstein(g=0.5)
    cfg.temperature = [180.0, 200.0, 220.0, 250.0, 270.0, 290.0, 300.0]
    cfg.bc.temperature_top = 180.0
    cfg.bc.temperature_bottom = 300.0

    # 500 wavenumber bands
    n_bands = 500
    wn_edges = np.linspace(100.0, 3000.0, n_bands + 1)
    bands = [(wn_edges[i], wn_edges[i + 1]) for i in range(n_bands)]

    # --- C++ spectral loop ---
    t0 = time.monotonic()
    results_cpp = disortpp.solve_spectral_bands(cfg, bands)
    t_cpp = (time.monotonic() - t0) * 1000

    # --- Python loop ---
    solver = disortpp.DisortSolver()
    t0 = time.monotonic()
    results_py = []
    for wn_lo, wn_hi in bands:
        cfg.wavenumber_low = wn_lo
        cfg.wavenumber_high = wn_hi
        results_py.append(solver.solve(cfg))
    t_py = (time.monotonic() - t0) * 1000

    # Verify results match
    max_diff = max(abs(results_cpp[i].flux_up[0] - results_py[i].flux_up[0])
                   for i in range(n_bands))

    print(f"\n  {n_bands} wavenumber bands, {nlyr} layers, {nstr} streams")
    print(f"\n  C++ spectral loop: {t_cpp:8.1f} ms  ({t_cpp/n_bands:.3f} ms/band)")
    print(f"  Python loop:       {t_py:8.1f} ms  ({t_py/n_bands:.3f} ms/band)")
    print(f"  Speedup:           {t_py/t_cpp:.2f}x")
    print(f"  Max difference:    {max_diff:.2e}")
    print()


# ============================================================================
# Example 4: Wavelength-dependent overrides with solve_flux_spectral_bands
# ============================================================================

def example_spectral_overrides():
    """
    Demonstrate per-wavelength overrides for the spectral loop functions.

    This repeats the Rayleigh scattering scenario from Example 2, but instead
    of looping in Python, we pass per-wavelength arrays (delta_tau,
    single_scat_albedo, surface_albedo) directly to solve_flux_spectral_bands.
    The entire computation runs in C++ (and OpenMP-parallel if available).
    """
    print("=" * 70)
    print("Example 4: Wavelength-dependent overrides (solve_flux_spectral_bands)")
    print("=" * 70)

    nlyr = 3
    nstr = 16

    # Wavenumber grid: 5000-25000 cm^-1 (0.4-2.0 um) in 500 cm^-1 bands
    band_width = 500.0
    wn_edges = np.arange(5000.0, 25001.0, band_width)
    bands = [(wn_edges[i], wn_edges[i + 1]) for i in range(len(wn_edges) - 1)]
    band_centers = np.array([(lo + hi) / 2 for lo, hi in bands])
    n_bands = len(bands)

    # --- Build per-wavelength arrays ---

    # Reference optical depth at 10000 cm^-1
    tau_ref = np.array([0.05, 0.1, 0.2])
    wn_ref = 10000.0

    # delta_tau: Rayleigh scaling tau ~ (wn/wn_ref)^4  [n_bands, nlyr]
    delta_tau = []
    for wn in band_centers:
        scale = (wn / wn_ref) ** 4
        delta_tau.append(list(tau_ref * scale))

    # single_scat_albedo: pure scattering everywhere [n_bands, nlyr]
    single_scat_albedo = [[1.0] * nlyr] * n_bands

    # surface_albedo: wavelength-dependent, e.g. increasing toward longer
    # wavelengths (mimicking vegetation or ocean albedo variations) [n_bands]
    surface_albedo = list(0.02 + 0.18 * (band_centers - 5000.0) / 20000.0)

    # --- Set up the base config (quantities that don't change with wn) ---
    cfg = disortpp.DisortFluxConfig(nlyr, nstr)
    cfg.direct_beam_flux = 1.0
    cfg.direct_beam_mu = 0.5
    cfg.allocate()

    # Set a dummy delta_tau / ssa / phase function (will be overridden)
    cfg.delta_tau = list(tau_ref)
    cfg.single_scat_albedo = [1.0] * nlyr
    cfg.set_rayleigh()

    # --- Solve all bands in one C++ call with overrides ---
    results = disortpp.solve_flux_spectral_bands(
        cfg, bands,
        delta_tau=delta_tau,
        single_scat_albedo=single_scat_albedo,
        surface_albedo=surface_albedo,
    )

    # Extract fluxes
    flux_up_toa = np.array([r.flux_up[0] for r in results])
    flux_down_boa = np.array([r.total_flux_down(nlyr) for r in results])
    wavelength_um = 1e4 / band_centers

    print(f"\n  {n_bands} wavenumber bands from {wn_edges[0]:.0f} "
          f"to {wn_edges[-1]:.0f} cm^-1")
    print(f"\n  Per-wavelength overrides: delta_tau, single_scat_albedo, surface_albedo")
    print(f"\n  {'Wavelength [um]':>16}  {'Wn [cm^-1]':>12}  "
          f"{'Albedo_sfc':>10}  {'Flux Up TOA':>12}  {'Flux Down BOA':>14}")
    print("  " + "-" * 68)

    for i in range(0, n_bands, 4):
        print(f"  {wavelength_um[i]:16.3f}  {band_centers[i]:12.0f}  "
              f"{surface_albedo[i]:10.4f}  {flux_up_toa[i]:12.6f}  "
              f"{flux_down_boa[i]:14.6f}")

    # Planetary albedo spectrum
    flux_in = 1.0 * 0.5  # fbeam * mu0
    albedo = flux_up_toa / flux_in
    print(f"\n  Planetary albedo at 2.0 um (wn= 5000): {albedo[0]:.4f}")
    print(f"  Planetary albedo at 0.5 um (wn=20000): {albedo[-4]:.4f}")
    print()


# ============================================================================
# Example 5: Thermal emission with per-wavelength overrides (general solver)
# ============================================================================

def example_thermal_overrides():
    """
    Demonstrate per-wavelength overrides for the general solver
    (solve_spectral_bands) with thermal emission.

    Models an atmosphere where the optical depth and emissivity vary across
    the infrared spectrum.
    """
    print("=" * 70)
    print("Example 5: Thermal overrides (solve_spectral_bands)")
    print("=" * 70)

    nlyr = 4
    nstr = 8

    # Wavenumber bands: 200-2000 cm^-1 in 100 cm^-1 steps
    band_width = 100.0
    wn_edges = np.arange(200.0, 2001.0, band_width)
    bands = [(wn_edges[i], wn_edges[i + 1]) for i in range(len(wn_edges) - 1)]
    band_centers = np.array([(lo + hi) / 2 for lo, hi in bands])
    n_bands = len(bands)

    # --- Base config ---
    cfg = disortpp.DisortConfig(nlyr, nstr)
    cfg.flags.use_lambertian_surface = True
    cfg.flags.use_thermal_emission = True
    cfg.flags.comp_only_fluxes = True
    cfg.allocate()

    # Base optical properties (will be overridden per wavelength)
    cfg.delta_tau = [0.5, 0.5, 0.5, 0.5]
    cfg.single_scat_albedo = [0.3, 0.3, 0.3, 0.3]
    cfg.set_isotropic()

    # Temperature profile (constant across wavelength)
    cfg.temperature = [180.0, 210.0, 240.0, 270.0, 300.0]
    cfg.bc.direct_beam_flux = 0.0
    cfg.bc.temperature_top = 180.0
    cfg.bc.temperature_bottom = 300.0
    cfg.bc.surface_albedo = 0.0

    # --- Per-wavelength overrides ---

    # Optical depth increases toward longer wavelengths (lower wavenumbers)
    # Mimics a gas with stronger absorption at lower wavenumbers
    tau_base = np.array([0.2, 0.5, 1.0, 2.0])
    delta_tau = []
    for wn in band_centers:
        scale = (1000.0 / wn) ** 2  # stronger absorption at lower wn
        delta_tau.append(list(tau_base * scale))

    # Emissivity at top: drops at window regions (higher wavenumbers)
    emissivity_top = list(0.5 + 0.5 * np.exp(-((band_centers - 600) / 300) ** 2))

    # --- Solve ---
    results = disortpp.solve_spectral_bands(
        cfg, bands,
        delta_tau=delta_tau,
        emissivity_top=emissivity_top,
    )

    flux_up_toa = np.array([r.flux_up[0] for r in results])
    flux_down_boa = np.array([r.flux_down[-1] for r in results])

    print(f"\n  {n_bands} wavenumber bands from {wn_edges[0]:.0f} "
          f"to {wn_edges[-1]:.0f} cm^-1")
    print(f"\n  Per-wavelength overrides: delta_tau, emissivity_top")
    print(f"\n  {'Band [cm^-1]':>16}  {'Emiss_top':>10}  "
          f"{'Flux Up TOA':>12}  {'Flux Down BOA':>14}")
    print("  " + "-" * 56)

    for i in range(0, n_bands, 2):
        lo, hi = bands[i]
        print(f"  {lo:7.0f}-{hi:5.0f}  {emissivity_top[i]:10.4f}  "
              f"{flux_up_toa[i]:12.4f}  {flux_down_boa[i]:14.4f}")

    total_up = np.sum(flux_up_toa)
    print(f"\n  Total upward flux at TOA: {total_up:.4f}")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    example_thermal_spectrum()
    example_scattering_spectrum()
    example_performance()
    example_spectral_overrides()
    example_thermal_overrides()
    print("All examples completed successfully.")
