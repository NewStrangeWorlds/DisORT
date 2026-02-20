Examples
========

Complete example scripts are provided in the ``python/`` directory. This
page walks through the key patterns.

.. contents:: On this page
   :local:
   :depth: 1


Flux-only calculation
---------------------

Compute fluxes for a 6-layer atmosphere illuminated by a direct beam.

.. code-block:: python

   import disortpp
   import numpy as np

   nlyr = 6
   nstr = 16

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

   flux_up = np.array(result.flux_up)
   flux_down = np.array(result.flux_down)
   flux_direct = np.array(result.flux_direct_beam)

   for lev in range(nlyr + 1):
       net = result.net_flux(lev)
       print(f"Level {lev}: direct={flux_direct[lev]:.4f}  "
             f"down={flux_down[lev]:.4f}  up={flux_up[lev]:.4f}  "
             f"net={net:.4f}")


Full intensity calculation
--------------------------

Compute intensities at user-specified angles and optical depths.

.. code-block:: python

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

   # Optical properties (different phase function per layer)
   cfg.delta_tau = [1.0, 2.0]
   cfg.single_scat_albedo = [0.95, 0.85]
   cfg.set_henyey_greenstein(g=0.75, lc=0)
   cfg.set_henyey_greenstein(g=0.60, lc=1)

   # User output specification
   cfg.tau_user = [0.0, 1.0, 3.0]
   cfg.mu_user = [-1.0, -0.5, 0.5, 1.0]
   cfg.phi_user = [0.0, 90.0, 180.0]

   # Boundary conditions
   cfg.bc.direct_beam_flux = np.pi
   cfg.bc.direct_beam_mu = 0.5
   cfg.bc.direct_beam_phi = 0.0
   cfg.bc.surface_albedo = 0.1

   solver = disortpp.DisortSolver()
   result = solver.solve(cfg)

   # 3D intensity array: (num_user_tau, num_user_mu, num_phi)
   intensity = np.array(result.intensity)
   print(f"Intensity shape: {intensity.shape}")
   print(f"Intensity at TOA, mu=1.0, phi=0: {intensity[0, 3, 0]:.6f}")


Fast flux solver
----------------

The ``DisortFluxSolver`` uses compile-time stream counts for faster
execution.

.. code-block:: python

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

   # Use factory function to create the right solver
   solver = disortpp.create_flux_solver(nstr)
   result = solver.solve(cfg)

   for lev in range(nlyr + 1):
       print(f"Level {lev}: up={result.flux_up[lev]:.4f}  "
             f"down={result.total_flux_down(lev):.4f}")


Thermal emission
----------------

Compute thermal fluxes in the infrared with no direct beam.

.. code-block:: python

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

   cfg.delta_tau = [0.5, 1.0, 2.0, 4.0]
   cfg.single_scat_albedo = [0.5] * nlyr
   cfg.set_henyey_greenstein(g=0.0)

   # Temperature profile: TOA (200 K) to BOA (300 K)
   cfg.temperature = [200.0, 220.0, 250.0, 280.0, 300.0]
   cfg.bc.temperature_top = 200.0
   cfg.bc.temperature_bottom = 300.0
   cfg.bc.emissivity_top = 1.0
   cfg.bc.surface_albedo = 0.0  # perfect emitter

   solver = disortpp.DisortSolver()
   result = solver.solve(cfg)

   for lev in range(nlyr + 1):
       print(f"Level {lev} ({cfg.temperature[lev]:.0f} K): "
             f"down={result.flux_down[lev]:.4f}  "
             f"up={result.flux_up[lev]:.4f}")


Spectral loop (thermal emission spectrum)
------------------------------------------

Compute thermal emission across many wavenumber bands efficiently using the
C++ spectral loop with OpenMP parallelisation.

.. code-block:: python

   nlyr = 4
   nstr = 8

   cfg = disortpp.DisortConfig(nlyr, nstr)
   cfg.flags.use_lambertian_surface = True
   cfg.flags.use_thermal_emission = True
   cfg.flags.comp_only_fluxes = True
   cfg.allocate()

   cfg.delta_tau = [0.2, 0.5, 1.0, 2.0]
   cfg.single_scat_albedo = [0.3] * nlyr
   cfg.set_isotropic()

   cfg.temperature = [180.0, 210.0, 240.0, 270.0, 300.0]
   cfg.bc.temperature_top = 180.0
   cfg.bc.temperature_bottom = 300.0

   # Define 38 bands: 100--2000 cm^-1 in 50 cm^-1 steps
   wn_edges = np.arange(100.0, 2001.0, 50.0)
   bands = [(wn_edges[i], wn_edges[i + 1]) for i in range(len(wn_edges) - 1)]

   # Solve all bands in one C++ call (OpenMP-parallelised)
   results = disortpp.solve_spectral_bands(cfg, bands)

   # Extract upward flux at TOA for each band
   flux_up_toa = np.array([r.flux_up[0] for r in results])
   band_centers = np.array([(lo + hi) / 2 for lo, hi in bands])

   print(f"Total upward flux at TOA: {np.sum(flux_up_toa):.4f}")


Wavelength-dependent scattering (Python loop)
----------------------------------------------

When optical properties change with wavenumber (e.g., Rayleigh scattering
where :math:`\tau \propto \nu^4`), one approach is to create a new
configuration per band and use the flux solver in a Python loop.

.. code-block:: python

   nlyr = 3
   nstr = 16

   solver = disortpp.create_flux_solver(nstr)

   tau_ref = np.array([0.05, 0.1, 0.2])
   wn_ref = 10000.0

   bands = [(5000.0 + i * 500, 5500.0 + i * 500) for i in range(40)]

   for wn_lo, wn_hi in bands:
       wn_center = (wn_lo + wn_hi) / 2

       cfg = disortpp.DisortFluxConfig(nlyr, nstr)
       cfg.direct_beam_flux = 1.0
       cfg.direct_beam_mu = 0.5
       cfg.surface_albedo = 0.1
       cfg.wavenumber_low = wn_lo
       cfg.wavenumber_high = wn_hi
       cfg.allocate()

       # Rayleigh-like scaling
       scale = (wn_center / wn_ref) ** 4
       cfg.delta_tau = list(tau_ref * scale)
       cfg.single_scat_albedo = [1.0] * nlyr
       cfg.set_rayleigh()

       result = solver.solve(cfg)
       print(f"{wn_center:.0f} cm^-1: flux_up_TOA = {result.flux_up[0]:.6f}")

A more efficient approach uses per-wavelength overrides, as shown in the
next section.


Wavelength-dependent scattering (spectral overrides)
----------------------------------------------------

The spectral loop functions accept optional per-wavelength arrays that
override the base configuration for each spectral point. This keeps the
entire computation in C++ (with OpenMP parallelisation) while still allowing
optical properties to vary across the spectrum.

The following example repeats the Rayleigh scattering calculation above,
but passes the wavelength-dependent ``delta_tau``, ``single_scat_albedo``,
and ``surface_albedo`` directly to ``solve_flux_spectral_bands``.

.. code-block:: python

   nlyr = 3
   nstr = 16

   # Wavenumber bands: 5000-25000 cm^-1 in 500 cm^-1 steps
   wn_edges = np.arange(5000.0, 25001.0, 500.0)
   bands = [(wn_edges[i], wn_edges[i + 1]) for i in range(len(wn_edges) - 1)]
   band_centers = np.array([(lo + hi) / 2 for lo, hi in bands])
   n_bands = len(bands)

   # Per-wavelength optical depths: Rayleigh scaling tau ~ (wn/wn_ref)^4
   tau_ref = np.array([0.05, 0.1, 0.2])
   wn_ref = 10000.0
   delta_tau = [list(tau_ref * (wn / wn_ref) ** 4) for wn in band_centers]

   # Per-wavelength single-scatter albedo (pure scattering)
   single_scat_albedo = [[1.0] * nlyr] * n_bands

   # Per-wavelength surface albedo (increasing toward longer wavelengths)
   surface_albedo = list(0.02 + 0.18 * (band_centers - 5000.0) / 20000.0)

   # Base config (quantities that don't change with wavenumber)
   cfg = disortpp.DisortFluxConfig(nlyr, nstr)
   cfg.direct_beam_flux = 1.0
   cfg.direct_beam_mu = 0.5
   cfg.allocate()
   cfg.delta_tau = list(tau_ref)
   cfg.single_scat_albedo = [1.0] * nlyr
   cfg.set_rayleigh()

   # Solve all bands in one call -- overrides replace per-band values
   results = disortpp.solve_flux_spectral_bands(
       cfg, bands,
       delta_tau=delta_tau,
       single_scat_albedo=single_scat_albedo,
       surface_albedo=surface_albedo,
   )

   for i, r in enumerate(results):
       print(f"{band_centers[i]:.0f} cm^-1: flux_up_TOA = {r.flux_up[0]:.6f}")

Any keyword argument that is omitted (or passed as an empty list) uses the
value from the base ``config`` for all wavelengths, so existing code
without overrides continues to work unchanged.


Thermal emission with spectral overrides
-----------------------------------------

Per-wavelength overrides work with the general solver and thermal emission
too. Here the optical depth and top-boundary emissivity vary across the
infrared spectrum:

.. code-block:: python

   nlyr = 4
   nstr = 8

   # 18 bands: 200-2000 cm^-1 in 100 cm^-1 steps
   wn_edges = np.arange(200.0, 2001.0, 100.0)
   bands = [(wn_edges[i], wn_edges[i + 1]) for i in range(len(wn_edges) - 1)]
   band_centers = np.array([(lo + hi) / 2 for lo, hi in bands])

   # Base config
   cfg = disortpp.DisortConfig(nlyr, nstr)
   cfg.flags.use_lambertian_surface = True
   cfg.flags.use_thermal_emission = True
   cfg.flags.comp_only_fluxes = True
   cfg.allocate()

   cfg.delta_tau = [0.5, 0.5, 0.5, 0.5]
   cfg.single_scat_albedo = [0.3] * nlyr
   cfg.set_isotropic()
   cfg.temperature = [180.0, 210.0, 240.0, 270.0, 300.0]
   cfg.bc.temperature_top = 180.0
   cfg.bc.temperature_bottom = 300.0

   # Per-wavelength optical depth (stronger at lower wavenumbers)
   tau_base = np.array([0.2, 0.5, 1.0, 2.0])
   delta_tau = [list(tau_base * (1000.0 / wn) ** 2) for wn in band_centers]

   # Per-wavelength top emissivity (Gaussian profile)
   emissivity_top = list(
       0.5 + 0.5 * np.exp(-((band_centers - 600) / 300) ** 2)
   )

   results = disortpp.solve_spectral_bands(
       cfg, bands,
       delta_tau=delta_tau,
       emissivity_top=emissivity_top,
   )

   flux_up_toa = np.array([r.flux_up[0] for r in results])
   print(f"Total upward flux at TOA: {np.sum(flux_up_toa):.4f}")
