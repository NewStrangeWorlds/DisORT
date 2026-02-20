Python Result Classes
=====================

.. contents:: On this page
   :local:
   :depth: 1


DisortResult
------------

Returned by ``DisortSolver.solve()``. Contains all computed radiant
quantities.

Flux vectors
^^^^^^^^^^^^

All flux attributes are Python lists (convertible to NumPy arrays). Size
is ``num_user_tau`` (or ``num_layers + 1`` when ``use_user_tau = False``).

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Attribute
     - Description
   * - ``flux_direct_beam``
     - Direct-beam flux.
   * - ``flux_down``
     - Diffuse downward flux.
   * - ``flux_up``
     - Diffuse upward flux.
   * - ``flux_tau_divergence``
     - Flux divergence.
   * - ``mean_intensity``
     - Mean intensity (including direct beam).
   * - ``mean_intensity_down``
     - Mean diffuse downward intensity.
   * - ``mean_intensity_up``
     - Mean diffuse upward intensity.
   * - ``mean_intensity_direct_beam``
     - Mean direct beam intensity.


Special boundary condition mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Attribute
     - Description
   * - ``albedo_medium``
     - Medium albedo at each user angle.
   * - ``transmissivity_medium``
     - Medium transmissivity at each user angle.
   * - ``spherical_albedo``
     - Spherical albedo (scalar).
   * - ``spherical_transmissivity``
     - Spherical transmissivity (scalar).


Intensity arrays
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``mu_angles``
     - Polar angle cosines for intensity output.
   * - ``intensity``
     - 3D nested list [tau][mu][phi].
   * - ``intensity_azimuth_avg``
     - 2D nested list [tau][mu].
   * - ``intensity_fourier_expansion``
     - 3D nested list [tau][mu][fourier term].

Convert to NumPy for convenient indexing:

.. code-block:: python

   intensity = np.array(result.intensity)
   # intensity.shape == (num_user_tau, num_user_mu, num_phi)

   flux_up = np.array(result.flux_up)
   # flux_up.shape == (num_user_tau,)


Methods
^^^^^^^

.. method:: result.num_user_tau()

   Number of output levels.

.. method:: result.num_user_mu()

   Number of output polar angles.

.. method:: result.num_phi()

   Number of azimuthal angles.

.. method:: result.total_flux_down(lu)

   Total downward flux at output level ``lu`` (direct + diffuse).

.. method:: result.net_flux(lu)

   Net flux at output level ``lu`` (downward positive).

.. method:: result.intensities(iu, lu, j)

   Access the 3D intensity at user angle ``iu``, level ``lu``, azimuth
   ``j``. Indices are 0-based.

.. method:: result.intensities_azimuth_avg(iu, lu)

   Access the azimuthally-averaged intensity. Indices are 0-based.


FluxResult
----------

Returned by ``DisortFluxSolver.solve()``. Lightweight flux-only container.

All vectors have size ``num_layers + 1`` (index 0 = TOA, last = BOA).

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Attribute
     - Description
   * - ``flux_direct_beam``
     - Direct-beam flux at each level.
   * - ``flux_down``
     - Diffuse downward flux at each level.
   * - ``flux_up``
     - Diffuse upward flux at each level.
   * - ``flux_tau_divergence``
     - Flux divergence at each level.
   * - ``mean_intensity``
     - Mean intensity at each level.
   * - ``mean_intensity_down``
     - Mean diffuse downward intensity.
   * - ``mean_intensity_up``
     - Mean diffuse upward intensity.
   * - ``mean_intensity_direct_beam``
     - Mean direct beam intensity.

Methods
^^^^^^^

.. method:: result.num_levels()

   Number of output levels.

.. method:: result.total_flux_down(lev)
   :no-index:

   Total downward flux at level ``lev``.

.. method:: result.net_flux(lev)
   :no-index:

   Net flux at level ``lev``.
