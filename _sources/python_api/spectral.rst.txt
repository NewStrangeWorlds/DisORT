Spectral Loop Functions
=======================

Radiative transfer calculations typically loop over many wavenumber bands.
Calling the solver repeatedly from Python incurs per-call overhead (GIL
acquire/release, type conversions). The spectral loop functions move the
entire wavenumber iteration into C++, releasing the GIL once and running all
bands natively.

When OpenMP is available, the spectral loops are **parallelised across
threads**, with each thread using its own solver and configuration instance.
The number of threads can be controlled via the ``OMP_NUM_THREADS``
environment variable.

.. contents:: On this page
   :local:
   :depth: 1


Per-wavelength overrides
------------------------

All four spectral loop functions accept optional keyword arguments that
supply per-wavelength values for quantities that commonly depend on
wavelength. When an override array is provided, it replaces the
corresponding value in the configuration for each spectral point. When
omitted (the default), the value from the base configuration is used for
all wavelengths.

.. list-table:: Available overrides
   :header-rows: 1
   :widths: 25 20 55

   * - Keyword
     - Shape
     - Description
   * - ``delta_tau``
     - ``[n_wl, num_layers]``
     - Optical depth per layer
   * - ``single_scat_albedo``
     - ``[n_wl, num_layers]``
     - Single-scattering albedo per layer
   * - ``phase_function_moments``
     - ``[n_wl, num_layers, nmom+1]``
     - Phase function Legendre moments per layer
   * - ``surface_albedo``
     - ``[n_wl]``
     - Surface albedo
   * - ``emissivity_top``
     - ``[n_wl]``
     - Top-of-atmosphere emissivity
   * - ``direct_beam_flux``
     - ``[n_wl]``
     - Direct beam flux at the top boundary
   * - ``isotropic_flux_top``
     - ``[n_wl]``
     - Isotropic flux at the top boundary
   * - ``isotropic_flux_bottom``
     - ``[n_wl]``
     - Isotropic flux at the bottom boundary

Here ``n_wl`` is the number of wavenumber points (or bands). The outer
index always corresponds to the spectral point.

.. note::

   The override arrays are shared read-only across OpenMP threads.
   Each thread applies the overrides to its own local copy of the
   configuration, so thread safety is guaranteed.


General solver
--------------

solve_spectral
^^^^^^^^^^^^^^

.. function:: disortpp.solve_spectral(config, wavenumbers, *, delta_tau=[], single_scat_albedo=[], phase_function_moments=[], surface_albedo=[], emissivity_top=[], direct_beam_flux=[], isotropic_flux_top=[], isotropic_flux_bottom=[])

   Solve for a list of single wavenumbers using ``DisortSolver``.

   For each wavenumber, ``wavenumber_low`` and ``wavenumber_high`` are both
   set to the given value. This is appropriate for monochromatic
   calculations where thermal emission is not used.

   :param config: ``DisortConfig`` instance. A copy is made internally;
      the original is not modified.
   :param wavenumbers: List of wavenumber values [cm\ :sup:`-1`].
   :type wavenumbers: list[float]
   :param delta_tau: Per-wavelength optical depths ``[n_wl][num_layers]``.
      Empty (default) uses the config value for all wavelengths.
   :type delta_tau: list[list[float]]
   :param single_scat_albedo: Per-wavelength SSA ``[n_wl][num_layers]``.
   :type single_scat_albedo: list[list[float]]
   :param phase_function_moments: Per-wavelength moments
      ``[n_wl][num_layers][nmom+1]``.
   :type phase_function_moments: list[list[list[float]]]
   :param surface_albedo: Per-wavelength surface albedo ``[n_wl]``.
   :type surface_albedo: list[float]
   :param emissivity_top: Per-wavelength top emissivity ``[n_wl]``.
   :type emissivity_top: list[float]
   :param direct_beam_flux: Per-wavelength beam flux ``[n_wl]``.
   :type direct_beam_flux: list[float]
   :param isotropic_flux_top: Per-wavelength isotropic top flux ``[n_wl]``.
   :type isotropic_flux_top: list[float]
   :param isotropic_flux_bottom: Per-wavelength isotropic bottom flux
      ``[n_wl]``.
   :type isotropic_flux_bottom: list[float]
   :returns: List of ``DisortResult``, one per wavenumber.
   :rtype: list[DisortResult]

   .. code-block:: python

      results = disortpp.solve_spectral(cfg, [5000.0, 10000.0, 15000.0])
      flux_up_toa = [r.flux_up[0] for r in results]


solve_spectral_bands
^^^^^^^^^^^^^^^^^^^^

.. function:: disortpp.solve_spectral_bands(config, wavenumber_bands, *, delta_tau=[], single_scat_albedo=[], phase_function_moments=[], surface_albedo=[], emissivity_top=[], direct_beam_flux=[], isotropic_flux_top=[], isotropic_flux_bottom=[])

   Solve for a list of wavenumber bands using ``DisortSolver``.

   For each band, ``wavenumber_low`` and ``wavenumber_high`` are set to the
   lower and upper edges respectively. This is the appropriate mode for
   calculations with thermal emission.

   :param config: ``DisortConfig`` instance (copied internally).
   :param wavenumber_bands: List of (low, high) wavenumber pairs
      [cm\ :sup:`-1`].
   :type wavenumber_bands: list[tuple[float, float]]
   :param delta_tau: Per-band optical depths ``[n_bands][num_layers]``.
   :type delta_tau: list[list[float]]
   :param single_scat_albedo: Per-band SSA ``[n_bands][num_layers]``.
   :type single_scat_albedo: list[list[float]]
   :param phase_function_moments: Per-band moments
      ``[n_bands][num_layers][nmom+1]``.
   :type phase_function_moments: list[list[list[float]]]
   :param surface_albedo: Per-band surface albedo ``[n_bands]``.
   :type surface_albedo: list[float]
   :param emissivity_top: Per-band top emissivity ``[n_bands]``.
   :type emissivity_top: list[float]
   :param direct_beam_flux: Per-band beam flux ``[n_bands]``.
   :type direct_beam_flux: list[float]
   :param isotropic_flux_top: Per-band isotropic top flux ``[n_bands]``.
   :type isotropic_flux_top: list[float]
   :param isotropic_flux_bottom: Per-band isotropic bottom flux
      ``[n_bands]``.
   :type isotropic_flux_bottom: list[float]
   :returns: List of ``DisortResult``, one per band.
   :rtype: list[DisortResult]

   .. code-block:: python

      bands = [(100.0, 200.0), (200.0, 300.0), (300.0, 400.0)]
      results = disortpp.solve_spectral_bands(cfg, bands)


Flux solver
-----------

solve_flux_spectral
^^^^^^^^^^^^^^^^^^^

.. function:: disortpp.solve_flux_spectral(config, wavenumbers, *, delta_tau=[], single_scat_albedo=[], phase_function_moments=[], surface_albedo=[], emissivity_top=[], direct_beam_flux=[], isotropic_flux_top=[], isotropic_flux_bottom=[])

   Solve for fluxes at a list of single wavenumbers using the flux solver.
   Dispatches to the appropriate ``DisortFluxSolver<NStr>`` based on
   ``config.num_streams``.

   :param config: ``DisortFluxConfig`` instance.
   :param wavenumbers: List of wavenumber values [cm\ :sup:`-1`].
   :type wavenumbers: list[float]
   :param delta_tau: Per-wavelength optical depths ``[n_wl][num_layers]``.
   :type delta_tau: list[list[float]]
   :param single_scat_albedo: Per-wavelength SSA ``[n_wl][num_layers]``.
   :type single_scat_albedo: list[list[float]]
   :param phase_function_moments: Per-wavelength moments
      ``[n_wl][num_layers][nmom+1]``.
   :type phase_function_moments: list[list[list[float]]]
   :param surface_albedo: Per-wavelength surface albedo ``[n_wl]``.
   :type surface_albedo: list[float]
   :param emissivity_top: Per-wavelength top emissivity ``[n_wl]``.
   :type emissivity_top: list[float]
   :param direct_beam_flux: Per-wavelength beam flux ``[n_wl]``.
   :type direct_beam_flux: list[float]
   :param isotropic_flux_top: Per-wavelength isotropic top flux ``[n_wl]``.
   :type isotropic_flux_top: list[float]
   :param isotropic_flux_bottom: Per-wavelength isotropic bottom flux
      ``[n_wl]``.
   :type isotropic_flux_bottom: list[float]
   :returns: List of ``FluxResult``, one per wavenumber.
   :rtype: list[FluxResult]


solve_flux_spectral_bands
^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: disortpp.solve_flux_spectral_bands(config, wavenumber_bands, *, delta_tau=[], single_scat_albedo=[], phase_function_moments=[], surface_albedo=[], emissivity_top=[], direct_beam_flux=[], isotropic_flux_top=[], isotropic_flux_bottom=[])

   Solve for fluxes at a list of wavenumber bands using the flux solver.

   :param config: ``DisortFluxConfig`` instance.
   :param wavenumber_bands: List of (low, high) wavenumber pairs
      [cm\ :sup:`-1`].
   :type wavenumber_bands: list[tuple[float, float]]
   :param delta_tau: Per-band optical depths ``[n_bands][num_layers]``.
   :type delta_tau: list[list[float]]
   :param single_scat_albedo: Per-band SSA ``[n_bands][num_layers]``.
   :type single_scat_albedo: list[list[float]]
   :param phase_function_moments: Per-band moments
      ``[n_bands][num_layers][nmom+1]``.
   :type phase_function_moments: list[list[list[float]]]
   :param surface_albedo: Per-band surface albedo ``[n_bands]``.
   :type surface_albedo: list[float]
   :param emissivity_top: Per-band top emissivity ``[n_bands]``.
   :type emissivity_top: list[float]
   :param direct_beam_flux: Per-band beam flux ``[n_bands]``.
   :type direct_beam_flux: list[float]
   :param isotropic_flux_top: Per-band isotropic top flux ``[n_bands]``.
   :type isotropic_flux_top: list[float]
   :param isotropic_flux_bottom: Per-band isotropic bottom flux
      ``[n_bands]``.
   :type isotropic_flux_bottom: list[float]
   :returns: List of ``FluxResult``, one per band.
   :rtype: list[FluxResult]


Performance considerations
--------------------------

The C++ spectral loop avoids Python overhead for each wavenumber band. The
speedup depends on the number of bands and the per-band computation time:

- For small per-band cost (few layers, few streams), the overhead reduction
  is significant -- typical speedups of 2--5x over an equivalent Python
  loop.
- For large per-band cost (many layers, many streams), the overhead is a
  smaller fraction and the speedup is more modest.
- With OpenMP enabled, the spectral loops parallelise across CPU cores,
  providing near-linear scaling up to the number of available cores.

Example measuring the speedup:

.. code-block:: python

   import time

   # C++ spectral loop
   t0 = time.monotonic()
   results_cpp = disortpp.solve_spectral_bands(cfg, bands)
   t_cpp = time.monotonic() - t0

   # Equivalent Python loop
   solver = disortpp.DisortSolver()
   t0 = time.monotonic()
   results_py = []
   for wn_lo, wn_hi in bands:
       cfg.wavenumber_low = wn_lo
       cfg.wavenumber_high = wn_hi
       results_py.append(solver.solve(cfg))
   t_py = time.monotonic() - t0

   print(f"C++ loop: {t_cpp:.3f} s, Python loop: {t_py:.3f} s")
   print(f"Speedup: {t_py / t_cpp:.1f}x")
