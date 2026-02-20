Configuration Classes
=====================

Defined in ``DisortConfig.hpp`` and ``DisortFluxConfig.hpp``.

.. contents:: On this page
   :local:
   :depth: 2


DisortFlags
-----------

Control flags for the general solver.

.. code-block:: cpp

   struct DisortFlags {
       bool use_user_tau              = false;
       bool use_user_mu               = false;
       BoundaryConditionType ibcnd    = BoundaryConditionType::General;
       bool use_lambertian_surface    = true;
       bool use_thermal_emission      = false;
       bool use_spherical_beam        = false;
       bool comp_only_fluxes          = false;
       BrdfType brdf_type             = BrdfType::None;
       bool intensity_corr_buras      = false;
       bool intensity_corr_nakajima   = false;
       bool use_delta_m_plus          = false;
       bool output_fourier_expansion  = false;
   };

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Member
     - Default
     - Description
   * - ``use_user_tau``
     - ``false``
     - Return radiant quantities at user-specified optical depths
       (``tau_user``). When ``false``, output is at computational layer
       boundaries.
   * - ``use_user_mu``
     - ``false``
     - Return intensities at user-specified polar angles (``mu_user``).
       When ``false``, output is at the quadrature angles.
   * - ``ibcnd``
     - ``General``
     - Boundary condition type. ``Special`` returns only the medium's albedo
       and transmissivity.
   * - ``use_lambertian_surface``
     - ``true``
     - Use an isotropically reflecting (Lambertian) bottom boundary with
       reflectivity ``surface_albedo``.
   * - ``use_thermal_emission``
     - ``false``
     - Include thermal emission. Requires ``temperature``,
       ``temperature_top``, ``temperature_bottom``, and a valid wavenumber
       range.
   * - ``use_spherical_beam``
     - ``false``
     - Use pseudo-spherical geometry for the direct beam attenuation. The
       diffuse calculation remains plane-parallel.
   * - ``comp_only_fluxes``
     - ``false``
     - Compute only fluxes and mean intensities (skip angular intensities).
       Significantly faster when intensities are not needed.
   * - ``brdf_type``
     - ``None``
     - Type of BRDF model for the surface. When not ``None``, overrides
       the Lambertian surface.
   * - ``intensity_corr_buras``
     - ``false``
     - Apply the Buras & Emde single-scattering intensity correction.
   * - ``intensity_corr_nakajima``
     - ``false``
     - Apply the Nakajima & Tanaka (1988) TMS/IMS intensity correction.
   * - ``use_delta_m_plus``
     - ``false``
     - Use Delta-M+ scaling (Lin et al. 2018) instead of standard Delta-M.
   * - ``output_fourier_expansion``
     - ``false``
     - Return the Fourier expansion of the intensity as a separate output
       in ``intensity_fourier_expansion``.


BoundaryConditions
------------------

Boundary condition parameters.

.. code-block:: cpp

   struct BoundaryConditions {
       double direct_beam_flux     = 0.0;
       double direct_beam_mu       = 0.5;
       double direct_beam_phi      = 0.0;
       double isotropic_flux_top   = 0.0;
       double isotropic_flux_bottom = 0.0;
       double temperature_top      = 0.0;
       double temperature_bottom   = 0.0;
       double emissivity_top       = 1.0;
       double surface_albedo       = 0.0;
   };

.. list-table::
   :header-rows: 1
   :widths: 28 10 62

   * - Member
     - Default
     - Description
   * - ``direct_beam_flux``
     - 0.0
     - Intensity of the incident parallel beam at the top boundary.
   * - ``direct_beam_mu``
     - 0.5
     - Cosine of the polar angle of the incident beam (positive, measured
       from the normal).
   * - ``direct_beam_phi``
     - 0.0
     - Azimuth angle of the incident beam [degrees, 0--360].
   * - ``isotropic_flux_top``
     - 0.0
     - Intensity of isotropic illumination at the top boundary.
   * - ``isotropic_flux_bottom``
     - 0.0
     - Intensity of isotropic illumination at the bottom boundary.
   * - ``temperature_top``
     - 0.0
     - Temperature of the top boundary [K].
   * - ``temperature_bottom``
     - 0.0
     - Temperature of the bottom boundary [K].
   * - ``emissivity_top``
     - 1.0
     - Emissivity of the top boundary.
   * - ``surface_albedo``
     - 0.0
     - Lambertian albedo of the bottom boundary.


BRDF specifications
-------------------

RpvBrdfSpec
^^^^^^^^^^^

Rahman-Pinty-Verstraete BRDF parameters.

.. code-block:: cpp

   struct RpvBrdfSpec {
       double rho0  = 0.0;
       double k     = 0.0;
       double theta = 0.0;
       double sigma = 0.0;   // snow extension
       double t1    = 0.0;   // snow extension
       double t2    = 0.0;   // snow extension
       double scale = 1.0;
   };

AmbralsBrdfSpec
^^^^^^^^^^^^^^^

Ambrals (Ross-Li) kernel-based BRDF parameters.

.. code-block:: cpp

   struct AmbralsBrdfSpec {
       double iso = 0.0;   // isotropic component
       double vol = 0.0;   // volumetric component
       double geo = 0.0;   // geometric component
   };

CoxMunkBrdfSpec
^^^^^^^^^^^^^^^

Cox & Munk ocean surface BRDF parameters.

.. code-block:: cpp

   struct CoxMunkBrdfSpec {
       double u10              = 0.0;    // wind speed at 10 m [m/s]
       double pcl              = 0.0;    // pigment concentration
       double xsal             = 0.0;    // salinity
       double refractive_index = 1.34;   // refractive index of water
       bool   do_shadow        = true;   // enable Tsang shadowing
   };

HapkeBrdfSpec
^^^^^^^^^^^^^

Hapke opposition-effect BRDF parameters.

.. code-block:: cpp

   struct HapkeBrdfSpec {
       double b0 = 1.0;    // opposition effect amplitude
       double hh = 0.06;   // opposition effect angular width
       double w  = 0.6;    // single-scattering albedo of surface
   };

BrdfSpecification
^^^^^^^^^^^^^^^^^

Container holding the active BRDF model.

.. code-block:: cpp

   struct BrdfSpecification {
       std::optional<RpvBrdfSpec>     rpv;
       std::optional<AmbralsBrdfSpec> ambrals;
       std::optional<CoxMunkBrdfSpec> cox_munk;
       std::optional<HapkeBrdfSpec>   hapke;

       void setRpv(const RpvBrdfSpec& spec);
       void setAmbrals(const AmbralsBrdfSpec& spec);
       void setCoxMunk(const CoxMunkBrdfSpec& spec);
       void setHapke(const HapkeBrdfSpec& spec);
   };

Set the appropriate BRDF specification using the setter methods. The BRDF
type must match ``DisortFlags::brdf_type``.


DisortConfig
------------

Main configuration class for the general solver.

Defined in ``DisortConfig.hpp``.

Constructors
^^^^^^^^^^^^

.. code-block:: cpp

   DisortConfig();
   DisortConfig(int num_layers, int num_streams, int num_phase_func_moments = 0);

The parameterised constructor sets the three core dimensions. Call
``allocate()`` after setting all dimension and flag fields to allocate the
internal arrays.


Nested structs
^^^^^^^^^^^^^^

.. code-block:: cpp

   DisortFlags        flags;   // control flags
   BoundaryConditions bc;      // boundary conditions
   BrdfSpecification  brdf;    // BRDF specification


Dimension members
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Description
   * - ``int num_layers``
     - Number of atmospheric layers.
   * - ``int num_streams``
     - Number of computational streams (must be even, >= 4).
   * - ``int num_phase_func_moments``
     - Number of phase function Legendre moments (not counting the zeroth).
   * - ``int num_user_tau``
     - Number of user-specified optical depths for output.
   * - ``int num_user_mu``
     - Number of user-specified polar angle cosines for intensity output.
   * - ``int num_phi``
     - Number of azimuthal output angles.
   * - ``int num_phase_func_angles``
     - Number of scattering angle grid points (for tabulated phase
       functions).


Physical parameters
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Description
   * - ``double wavenumber_low``
     - Lower wavenumber bound [cm\ :sup:`-1`] for Planck function
       integration.
   * - ``double wavenumber_high``
     - Upper wavenumber bound [cm\ :sup:`-1`].
   * - ``double accuracy_fourier_series``
     - Convergence criterion for the azimuthal Fourier series (0 = use
       default).
   * - ``double bottom_radius``
     - Planetary radius [km] for pseudo-spherical geometry.


Array members
^^^^^^^^^^^^^

These are allocated by ``allocate()`` and must be filled before calling the
solver.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Member
     - Description
   * - ``vector<double> delta_tau``
     - Optical depth of each layer [``num_layers``].
   * - ``vector<double> single_scat_albedo``
     - Single-scattering albedo of each layer [``num_layers``].
   * - ``vector<vector<double>> phase_function_moments``
     - Phase function Legendre moments
       [``num_layers``][``nmomNstr()+1``]. Zeroth moment must be 1.0.
   * - ``vector<double> temperature``
     - Temperature at each level [``num_layers+1``], from TOA to BOA.
       Required when ``use_thermal_emission = true``.
   * - ``vector<double> level_altitudes``
     - Altitude at each level [``num_layers+1``]. Required when
       ``use_spherical_beam = true``.
   * - ``vector<double> tau_user``
     - User-specified optical depths [``num_user_tau``].
   * - ``vector<double> mu_user``
     - User-specified polar angle cosines [``num_user_mu``]. Negative
       values = downward, positive = upward.
   * - ``vector<double> phi_user``
     - User-specified azimuthal angles [degrees] [``num_phi``].
   * - ``vector<double> mu_phase_function``
     - Scattering angle cosines for tabulated phase functions
       [``num_phase_func_angles``].
   * - ``vector<vector<double>> phase_function``
     - Tabulated phase function values
       [``num_layers``][``num_phase_func_angles``].


Methods
^^^^^^^

.. function:: int nmomNstr() const

   Returns ``max(num_phase_func_moments, num_streams)``.

.. function:: void allocate()

   Allocate all internal arrays based on the current dimension and flag
   settings. Must be called after setting dimensions and before filling
   arrays.

.. function:: void validate() const

   Validate the configuration. Throws ``std::invalid_argument`` if any
   parameter is out of range or inconsistent.

.. function:: void setHenyeyGreenstein(double g, int lc = -1)

   Fill the phase function moments with the Henyey-Greenstein function
   (:math:`\chi_k = g^k`). If ``lc >= 0``, only fill layer ``lc``;
   otherwise fill all layers.

.. function:: void setIsotropic(int lc = -1)

   Set isotropic scattering for the specified layer(s).

.. function:: void setRayleigh(int lc = -1)

   Set Rayleigh scattering for the specified layer(s).

.. function:: void setHazeGarciaSiewert(int lc = -1)

   Set the Haze-L phase function (Garcia & Siewert, 1985).

.. function:: void setCloudGarciaSiewert(int lc = -1)

   Set the Cloud C.1 phase function (Garcia & Siewert, 1985).

.. function:: void setPhaseFunction(PhaseFunction type, double g = 0.0, int lc = -1)

   Set the phase function by enum type. The parameter *g* is used only for
   ``HenyeyGreenstein``.


DisortFluxConfig
----------------

Lightweight configuration for the flux-only solver. All fields are flat
(no nested structs) for minimal overhead.

Defined in ``DisortFluxConfig.hpp``.


Constructors
^^^^^^^^^^^^

.. code-block:: cpp

   DisortFluxConfig();
   DisortFluxConfig(int num_layers, int num_streams, int num_phase_func_moments = 0);


Members
^^^^^^^

**Dimensions:**

- ``int num_layers`` -- Number of atmospheric layers.
- ``int num_streams`` -- Number of streams (must match the template
  parameter of ``DisortFluxSolver<NStr>``).
- ``int num_phase_func_moments`` -- Number of phase function moments.

**Flags:**

- ``bool use_thermal_emission`` -- Include thermal emission (default: ``false``).
- ``bool use_spherical_beam`` -- Pseudo-spherical beam geometry (default: ``false``).
- ``bool use_delta_m_plus`` -- Delta-M+ scaling (default: ``false``).

**Boundary conditions:**

- ``double direct_beam_flux`` -- Incident beam intensity (default: 0).
- ``double direct_beam_mu`` -- Beam polar angle cosine (default: 0.5).
- ``double isotropic_flux_top`` -- Isotropic top illumination (default: 0).
- ``double isotropic_flux_bottom`` -- Isotropic bottom illumination (default: 0).
- ``double temperature_top`` -- Top boundary temperature [K] (default: 0).
- ``double temperature_bottom`` -- Bottom boundary temperature [K] (default: 0).
- ``double emissivity_top`` -- Top boundary emissivity (default: 1).
- ``double surface_albedo`` -- Lambertian surface albedo (default: 0).

**Physical parameters:**

- ``double wavenumber_low`` -- Lower wavenumber [cm\ :sup:`-1`] (default: 0).
- ``double wavenumber_high`` -- Upper wavenumber [cm\ :sup:`-1`] (default: 0).
- ``double bottom_radius`` -- Planetary radius [km] (default: 0).

**Arrays** (allocated by ``allocate()``):

- ``vector<double> delta_tau`` -- Layer optical depths [``num_layers``].
- ``vector<double> single_scat_albedo`` -- Layer single-scattering albedos [``num_layers``].
- ``vector<vector<double>> phase_function_moments`` -- Phase function moments [``num_layers``][``nmomNstr()+1``].
- ``vector<double> temperature`` -- Level temperatures [``num_layers+1``].
- ``vector<double> level_altitudes`` -- Level altitudes [``num_layers+1``].


Methods
^^^^^^^

``DisortFluxConfig`` provides the same methods as ``DisortConfig``:
``nmomNstr()``, ``allocate()``, ``validate()``, ``setHenyeyGreenstein()``,
``setIsotropic()``, ``setRayleigh()``, ``setHazeGarciaSiewert()``,
``setCloudGarciaSiewert()``, and ``setPhaseFunction()``.
