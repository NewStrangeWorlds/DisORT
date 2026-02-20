Result Classes
==============

Defined in ``DisortResult.hpp`` and ``FluxResult.hpp``.

.. contents:: On this page
   :local:
   :depth: 1


DisortResult
------------

Contains all computed radiant quantities from ``DisortSolver::solve()``.

Defined in ``DisortResult.hpp``.


Flux vectors
^^^^^^^^^^^^

All flux vectors have size ``num_user_tau`` (or ``num_layers + 1`` when
``use_user_tau = false``). Index 0 corresponds to the topmost output level.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Member
     - Description
   * - ``vector<double> flux_direct_beam``
     - Direct-beam flux (without Delta-M scaling).
   * - ``vector<double> flux_down``
     - Diffuse downward flux.
   * - ``vector<double> flux_up``
     - Diffuse upward flux.
   * - ``vector<double> flux_tau_divergence``
     - Flux divergence :math:`d(\text{net flux})/d(\tau)`.
   * - ``vector<double> mean_intensity``
     - Mean intensity including the direct beam contribution.
   * - ``vector<double> mean_intensity_down``
     - Mean diffuse downward intensity.
   * - ``vector<double> mean_intensity_up``
     - Mean diffuse upward intensity.
   * - ``vector<double> mean_intensity_direct_beam``
     - Mean intensity of the direct beam only.


Special boundary condition mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These fields are populated only when ``ibcnd = Special``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Member
     - Description
   * - ``vector<double> albedo_medium``
     - Albedo of the medium at each user angle [``num_user_mu``].
   * - ``vector<double> transmissivity_medium``
     - Transmissivity of the medium at each user angle [``num_user_mu``].
   * - ``double spherical_albedo``
     - Spherical albedo of the medium.
   * - ``double spherical_transmissivity``
     - Spherical transmissivity of the medium.


Intensity arrays
^^^^^^^^^^^^^^^^

Populated when ``comp_only_fluxes = false``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - Description
   * - ``vector<double> mu_angles``
     - Polar angle cosines at which intensities are returned
       [``num_user_mu``].
   * - ``vector<vector<vector<double>>> intensity``
     - Intensity array [``num_user_tau``][``num_user_mu``][``num_phi``].
   * - ``vector<vector<double>> intensity_azimuth_avg``
     - Azimuthally-averaged intensity
       [``num_user_tau``][``num_user_mu``].
   * - ``vector<vector<vector<double>>> intensity_fourier_expansion``
     - Fourier expansion coefficients (when
       ``output_fourier_expansion = true``)
       [``num_user_tau``][``num_user_mu``][``num_streams+1``].


Accessor methods
^^^^^^^^^^^^^^^^

.. function:: int num_user_tau() const

   Number of output levels.

.. function:: int num_user_mu() const

   Number of output polar angles.

.. function:: int num_phi() const

   Number of azimuthal output angles.

.. function:: double totalFluxDown(int lu) const

   Total downward flux (direct + diffuse) at output level ``lu``.

.. function:: double netFlux(int lu) const

   Net flux (downward positive) at output level ``lu``:
   ``flux_direct_beam[lu] + flux_down[lu] - flux_up[lu]``.

.. function:: double intensities(int iu, int lu, int j) const

   Access the 3D intensity array at user angle ``iu``, output level ``lu``,
   and azimuthal angle ``j``. Indices are 0-based.

.. function:: double intensitiesAzimuthAvg(int iu, int lu) const

   Access the azimuthally-averaged intensity at user angle ``iu`` and
   output level ``lu``. Indices are 0-based.


FluxResult
----------

Lightweight flux-only result from ``DisortFluxSolver::solve()``.

Defined in ``FluxResult.hpp``.

All vectors have size ``num_layers + 1``. Index 0 = TOA, index
``num_layers`` = BOA.


Members
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Member
     - Description
   * - ``vector<double> flux_direct_beam``
     - Direct-beam flux at each level.
   * - ``vector<double> flux_down``
     - Diffuse downward flux at each level.
   * - ``vector<double> flux_up``
     - Diffuse upward flux at each level.
   * - ``vector<double> flux_tau_divergence``
     - Flux divergence at each level.
   * - ``vector<double> mean_intensity``
     - Mean intensity (including direct beam) at each level.
   * - ``vector<double> mean_intensity_down``
     - Mean diffuse downward intensity at each level.
   * - ``vector<double> mean_intensity_up``
     - Mean diffuse upward intensity at each level.
   * - ``vector<double> mean_intensity_direct_beam``
     - Mean direct beam intensity at each level.


Methods
^^^^^^^

.. function:: int num_levels() const

   Number of output levels (``num_layers + 1``).

.. function:: double totalFluxDown(int lev) const

   Total downward flux at level ``lev``:
   ``flux_direct_beam[lev] + flux_down[lev]``.

.. function:: double netFlux(int lev) const

   Net flux at level ``lev``:
   ``flux_direct_beam[lev] + flux_down[lev] - flux_up[lev]``.
