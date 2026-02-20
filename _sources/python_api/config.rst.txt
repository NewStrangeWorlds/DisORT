Python Configuration Classes
============================

.. contents:: On this page
   :local:
   :depth: 2


DisortFlags
-----------

.. code-block:: python

   flags = disortpp.DisortFlags()

All members are read/write attributes:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Description
   * - ``use_user_tau``
     - ``bool``
     - Return output at user-specified optical depths.
   * - ``use_user_mu``
     - ``bool``
     - Return intensities at user-specified polar angles.
   * - ``ibcnd``
     - ``BoundaryConditionType``
     - Boundary condition type (``General`` or ``Special``).
   * - ``use_lambertian_surface``
     - ``bool``
     - Use Lambertian surface reflection.
   * - ``use_thermal_emission``
     - ``bool``
     - Include thermal emission.
   * - ``use_spherical_beam``
     - ``bool``
     - Pseudo-spherical beam geometry.
   * - ``comp_only_fluxes``
     - ``bool``
     - Compute only fluxes (skip angular intensities).
   * - ``brdf_type``
     - ``BrdfType``
     - BRDF model for the surface.
   * - ``intensity_corr_buras``
     - ``bool``
     - Buras & Emde intensity correction.
   * - ``intensity_corr_nakajima``
     - ``bool``
     - Nakajima & Tanaka intensity correction.
   * - ``use_delta_m_plus``
     - ``bool``
     - Use Delta-M+ scaling.
   * - ``output_fourier_expansion``
     - ``bool``
     - Return Fourier expansion of intensity.


BoundaryConditions
------------------

.. code-block:: python

   bc = disortpp.BoundaryConditions()

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Attribute
     - Type
     - Description
   * - ``direct_beam_flux``
     - ``float``
     - Incident parallel beam intensity.
   * - ``direct_beam_mu``
     - ``float``
     - Cosine of beam polar angle.
   * - ``direct_beam_phi``
     - ``float``
     - Azimuth angle of beam [deg].
   * - ``isotropic_flux_top``
     - ``float``
     - Isotropic illumination at top.
   * - ``isotropic_flux_bottom``
     - ``float``
     - Isotropic illumination at bottom.
   * - ``temperature_top``
     - ``float``
     - Top boundary temperature [K].
   * - ``temperature_bottom``
     - ``float``
     - Bottom boundary temperature [K].
   * - ``emissivity_top``
     - ``float``
     - Top boundary emissivity.
   * - ``surface_albedo``
     - ``float``
     - Lambertian surface albedo.


BRDF specifications
-------------------

.. code-block:: python

   rpv = disortpp.RpvBrdfSpec()           # rho0, k, theta, sigma, t1, t2, scale
   ambrals = disortpp.AmbralsBrdfSpec()   # iso, vol, geo
   cox_munk = disortpp.CoxMunkBrdfSpec()  # u10, pcl, xsal, refractive_index, do_shadow
   hapke = disortpp.HapkeBrdfSpec()       # b0, hh, w

   brdf = disortpp.BrdfSpecification()
   brdf.set_rpv(rpv)
   brdf.set_ambrals(ambrals)
   brdf.set_cox_munk(cox_munk)
   brdf.set_hapke(hapke)

All BRDF struct members are accessible as read/write attributes. See the
:doc:`C++ BRDF documentation </cpp_api/config>` for details on each
parameter.


DisortConfig
------------

Main configuration class for the general solver.

.. code-block:: python

   # Construct with dimensions
   cfg = disortpp.DisortConfig(num_layers, num_streams)
   cfg = disortpp.DisortConfig(num_layers, num_streams, num_phase_func_moments)

   # Or default-construct and set dimensions manually
   cfg = disortpp.DisortConfig()
   cfg.num_layers = 4
   cfg.num_streams = 16

**Nested structs** (read/write):

- ``cfg.flags`` -- ``DisortFlags``
- ``cfg.bc`` -- ``BoundaryConditions``
- ``cfg.brdf`` -- ``BrdfSpecification``

**Dimension attributes:**

- ``num_layers``, ``num_streams``, ``num_phase_func_moments``
- ``num_user_tau``, ``num_user_mu``, ``num_phi``
- ``num_phase_func_angles``

**Physical parameters:**

- ``wavenumber_low``, ``wavenumber_high`` (``float``)
- ``accuracy_fourier_series`` (``float``)
- ``bottom_radius`` (``float``)

**Array attributes** (set as Python lists after calling ``allocate()``):

- ``delta_tau`` -- Layer optical depths.
- ``single_scat_albedo`` -- Layer single-scattering albedos.
- ``phase_function_moments`` -- Nested list of Legendre moments.
- ``temperature`` -- Level temperatures.
- ``level_altitudes`` -- Level altitudes.
- ``tau_user`` -- User optical depths.
- ``mu_user`` -- User polar angle cosines.
- ``phi_user`` -- User azimuthal angles [deg].
- ``mu_phase_function`` -- Scattering angle grid.
- ``phase_function`` -- Tabulated phase function values.

**Methods:**

.. method:: cfg.nmom_nstr()

   Returns ``max(num_phase_func_moments, num_streams)``.

.. method:: cfg.allocate()

   Allocate internal arrays. Call after setting all dimensions and flags.

.. method:: cfg.validate()

   Validate the configuration. Raises ``ValueError`` on invalid input.

.. method:: cfg.set_henyey_greenstein(g, lc=-1)

   Fill Henyey-Greenstein phase function moments (:math:`\chi_k = g^k`).
   Set ``lc`` to fill only a specific layer (0-based), or ``-1`` for all
   layers.

.. method:: cfg.set_isotropic(lc=-1)

   Set isotropic scattering.

.. method:: cfg.set_rayleigh(lc=-1)

   Set Rayleigh scattering.

.. method:: cfg.set_haze_garcia_siewert(lc=-1)

   Set Haze-L phase function.

.. method:: cfg.set_cloud_garcia_siewert(lc=-1)

   Set Cloud C.1 phase function.

.. method:: cfg.set_phase_function(type, g=0.0, lc=-1)

   Set phase function by enum type.


DisortFluxConfig
----------------

Lightweight configuration for the flux-only solver.

.. code-block:: python

   cfg = disortpp.DisortFluxConfig(num_layers, num_streams)

All fields are flat attributes (no nested structs):

**Dimensions:** ``num_layers``, ``num_streams``, ``num_phase_func_moments``

**Flags:** ``use_thermal_emission``, ``use_spherical_beam``,
``use_delta_m_plus``

**Boundary conditions:** ``direct_beam_flux``, ``direct_beam_mu``,
``isotropic_flux_top``, ``isotropic_flux_bottom``, ``temperature_top``,
``temperature_bottom``, ``emissivity_top``, ``surface_albedo``

**Physical parameters:** ``wavenumber_low``, ``wavenumber_high``,
``bottom_radius``

**Arrays:** ``delta_tau``, ``single_scat_albedo``,
``phase_function_moments``, ``temperature``, ``level_altitudes``

**Methods:** Same as ``DisortConfig``: ``nmom_nstr()``, ``allocate()``,
``validate()``, ``set_henyey_greenstein()``, ``set_isotropic()``,
``set_rayleigh()``, ``set_haze_garcia_siewert()``,
``set_cloud_garcia_siewert()``, ``set_phase_function()``.
