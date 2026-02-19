#!/usr/bin/env python3
"""
DisORT++ Python Interface — Example Calculations

Demonstrates the two solvers:
  1. DisortSolver      — full radiative transfer (fluxes + intensities)
  2. DisortFluxSolver   — fast flux-only solver (compile-time stream count)

Build the Python module first:
    cd DisORT/build
    cmake .. -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)

Then run this script from the build directory:
    python3 ../python/example.py
"""

import sys
import os

# Add the build directory to the path so we can import the module
build_dir = os.path.join(os.path.dirname(__file__), "..", "build")
sys.path.insert(0, os.path.abspath(build_dir))

import disortpp
import numpy as np


# ============================================================================
# Example 1: Flux-only calculation with the general solver
# ============================================================================

def example_flux_only():
    """
    Compute fluxes for a 6-layer atmosphere illuminated by a direct beam.

    Setup:
      - 16 streams
      - Henyey-Greenstein phase function (g = 0.8)
      - Lambertian surface with albedo 0.1
      - Direct beam with flux pi, incident at mu0 = 0.5
    """
    print("=" * 70)
    print("Example 1: Flux-only calculation (DisortSolver)")
    print("=" * 70)

    nlyr = 6
    nstr = 16

    # Create and configure
    cfg = disortpp.DisortConfig(nlyr, nstr)
    cfg.flags.use_lambertian_surface = True
    cfg.flags.comp_only_fluxes = True
    cfg.allocate()

    # Layer optical properties
    cfg.delta_tau = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    cfg.single_scat_albedo = [0.9] * nlyr
    cfg.set_henyey_greenstein(g=0.8)

    # Boundary conditions
    cfg.bc.direct_beam_flux = np.pi
    cfg.bc.direct_beam_mu = 0.5
    cfg.bc.surface_albedo = 0.1

    # Solve
    solver = disortpp.DisortSolver()
    result = solver.solve(cfg)

    # Convert results to numpy arrays
    flux_up = np.array(result.flux_up)
    flux_down = np.array(result.flux_down)
    flux_direct = np.array(result.flux_direct_beam)

    # Print results at each level
    print(f"\n  {'Level':>5}  {'Direct Beam':>12}  {'Diffuse Down':>12}  {'Diffuse Up':>12}  {'Net Flux':>12}")
    print("  " + "-" * 57)
    for lev in range(nlyr + 1):
        net = result.net_flux(lev)
        print(f"  {lev:5d}  {flux_direct[lev]:12.6f}  {flux_down[lev]:12.6f}  "
              f"{flux_up[lev]:12.6f}  {net:12.6f}")

    print(f"\n  Planetary albedo (flux_up[TOA] / flux_down[TOA]): "
          f"{flux_up[0] / (flux_direct[0] + flux_down[0]):.6f}")
    print()


# ============================================================================
# Example 2: Full intensity calculation
# ============================================================================

def example_intensity():
    """
    Compute intensities at user-specified angles and optical depths.

    Setup:
      - 2-layer atmosphere
      - 16 streams with Nakajima-Tanaka intensity correction
      - User angles: mu = -1, -0.5, 0.5, 1.0
      - User azimuthal angles: phi = 0, 90, 180 degrees
      - Output at TOA, layer boundary, and BOA
    """
    print("=" * 70)
    print("Example 2: Full intensity calculation (DisortSolver)")
    print("=" * 70)

    nlyr = 2
    nstr = 16
    numu = 4
    nphi = 3

    cfg = disortpp.DisortConfig(nlyr, nstr)
    cfg.flags.use_lambertian_surface = True
    cfg.flags.use_user_tau = True
    cfg.flags.use_user_mu = True
    cfg.flags.comp_only_fluxes = False
    cfg.flags.intensity_corr_nakajima = True

    cfg.num_user_tau = nlyr + 1
    cfg.num_user_mu = numu
    cfg.num_phi = nphi
    cfg.allocate()

    # Optical properties
    cfg.delta_tau = [1.0, 2.0]
    cfg.single_scat_albedo = [0.95, 0.85]
    cfg.set_henyey_greenstein(g=0.75, lc=0)  # layer 0
    cfg.set_henyey_greenstein(g=0.60, lc=1)  # layer 1

    # User output specification
    cfg.tau_user = [0.0, 1.0, 3.0]  # TOA, boundary, BOA
    cfg.mu_user = [-1.0, -0.5, 0.5, 1.0]  # negative = downward, positive = upward
    cfg.phi_user = [0.0, 90.0, 180.0]

    # Boundary conditions
    cfg.bc.direct_beam_flux = np.pi
    cfg.bc.direct_beam_mu = 0.5
    cfg.bc.direct_beam_phi = 0.0
    cfg.bc.surface_albedo = 0.1

    # Solve
    solver = disortpp.DisortSolver()
    result = solver.solve(cfg)

    # Intensities as a 3D numpy array: (num_user_tau, num_user_mu, num_phi)
    intensity = np.array(result.intensity)
    mu_angles = np.array(result.mu_angles)
    phi_angles = np.array(cfg.phi_user)
    tau_user = np.array(cfg.tau_user)

    print(f"\n  Intensity array shape: {intensity.shape}")
    print(f"  (num_user_tau={result.num_user_tau()}, "
          f"num_user_mu={result.num_user_mu()}, num_phi={result.num_phi()})")

    # Print intensities at TOA (tau=0) for all angles
    print(f"\n  Intensities at TOA (tau = {tau_user[0]}):")
    header = "mu \\ phi"
    print(f"  {header:>10}", end="")
    for phi in phi_angles:
        print(f"  {phi:>10.1f}°", end="")
    print()
    print("  " + "-" * (12 + 12 * nphi))

    for iu, mu in enumerate(mu_angles):
        print(f"  {mu:10.4f}", end="")
        for jp in range(nphi):
            print(f"  {intensity[0, iu, jp]:12.6f}", end="")
        print()

    # Print intensities at BOA
    print(f"\n  Intensities at BOA (tau = {tau_user[-1]}):")
    print(f"  {header:>10}", end="")
    for phi in phi_angles:
        print(f"  {phi:>10.1f}°", end="")
    print()
    print("  " + "-" * (12 + 12 * nphi))

    for iu, mu in enumerate(mu_angles):
        print(f"  {mu:10.4f}", end="")
        for jp in range(nphi):
            print(f"  {intensity[-1, iu, jp]:12.6f}", end="")
        print()
    print()


# ============================================================================
# Example 3: Fast flux-only solver (DisortFluxSolver)
# ============================================================================

def example_flux_solver():
    """
    Use the templated flux-only solver for maximum performance.

    The flux solver uses compile-time stream counts for faster eigenvalue
    computation. Create it via create_flux_solver(nstr).
    """
    print("=" * 70)
    print("Example 3: Fast flux-only solver (DisortFluxSolver)")
    print("=" * 70)

    nlyr = 6
    nstr = 16

    cfg = disortpp.DisortFluxConfig(nlyr, nstr)
    cfg.direct_beam_flux = np.pi
    cfg.direct_beam_mu = 0.5
    cfg.surface_albedo = 0.1
    cfg.allocate()

    cfg.delta_tau = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    cfg.single_scat_albedo = [0.9] * nlyr
    cfg.set_henyey_greenstein(g=0.8)

    # Create solver via factory (dispatches to DisortFluxSolver<16>)
    solver = disortpp.create_flux_solver(nstr)
    result = solver.solve(cfg)

    flux_up = np.array(result.flux_up)
    flux_down = np.array(result.flux_down)
    flux_direct = np.array(result.flux_direct_beam)

    print(f"\n  {'Level':>5}  {'Direct Beam':>12}  {'Diffuse Down':>12}  {'Diffuse Up':>12}")
    print("  " + "-" * 45)
    for lev in range(nlyr + 1):
        print(f"  {lev:5d}  {flux_direct[lev]:12.6f}  {flux_down[lev]:12.6f}  "
              f"{flux_up[lev]:12.6f}")

    print(f"\n  Supported stream counts: 4, 6, 8, 10, 12, 14, 16, 32, 64")
    print()


# ============================================================================
# Example 4: Thermal emission
# ============================================================================

def example_thermal():
    """
    Compute fluxes with thermal emission in a hot atmosphere.

    Setup:
      - 4-layer atmosphere with temperature profile
      - No direct beam (nightside or infrared)
      - Wavenumber band 200-400 cm^-1 (thermal infrared)
    """
    print("=" * 70)
    print("Example 4: Thermal emission")
    print("=" * 70)

    nlyr = 4
    nstr = 8

    cfg = disortpp.DisortConfig(nlyr, nstr)
    cfg.flags.use_lambertian_surface = True
    cfg.flags.use_thermal_emission = True
    cfg.flags.comp_only_fluxes = True

    # Wavenumber band (cm^-1)
    cfg.wavenumber_low = 200.0
    cfg.wavenumber_high = 400.0

    cfg.allocate()

    # Optical properties
    cfg.delta_tau = [0.5, 1.0, 2.0, 4.0]
    cfg.single_scat_albedo = [0.5, 0.5, 0.5, 0.5]
    cfg.set_henyey_greenstein(g=0.0)  # isotropic for simplicity

    # Temperature profile (levels: TOA to BOA)
    cfg.temperature = [200.0, 220.0, 250.0, 280.0, 300.0]

    # Boundary conditions (no beam, thermal only)
    cfg.bc.direct_beam_flux = 0.0
    cfg.bc.temperature_top = 200.0      # cosmic background or space
    cfg.bc.temperature_bottom = 300.0   # surface temperature
    cfg.bc.emissivity_top = 1.0
    cfg.bc.surface_albedo = 0.0         # perfect emitter

    solver = disortpp.DisortSolver()
    result = solver.solve(cfg)

    flux_up = np.array(result.flux_up)
    flux_down = np.array(result.flux_down)
    mean_int = np.array(result.mean_intensity)

    print(f"\n  {'Level':>5}  {'Temp [K]':>10}  {'Flux Down':>12}  {'Flux Up':>12}  {'Mean Int.':>12}")
    print("  " + "-" * 52)
    for lev in range(nlyr + 1):
        print(f"  {lev:5d}  {cfg.temperature[lev]:10.1f}  {flux_down[lev]:12.6f}  "
              f"{flux_up[lev]:12.6f}  {mean_int[lev]:12.6f}")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    example_flux_only()
    example_intensity()
    example_flux_solver()
    example_thermal()
    print("All examples completed successfully.")
