Quick Start
===========

This page demonstrates the basic workflow for both the C++ and Python
interfaces.


C++ quick start
---------------

The following example computes fluxes for a 4-layer atmosphere with a
Henyey-Greenstein phase function and a Lambertian surface.

.. code-block:: cpp

   #include "DisortConfig.hpp"
   #include "DisortSolver.hpp"

   using namespace disortpp;

   int main() {
       int nlyr = 4;
       int nstr = 16;

       // Create and configure
       DisortConfig cfg(nlyr, nstr);
       cfg.flags.use_lambertian_surface = true;
       cfg.flags.comp_only_fluxes = true;
       cfg.allocate();

       // Layer optical properties
       cfg.delta_tau = {0.1, 0.5, 1.0, 2.0};
       cfg.single_scat_albedo = {0.9, 0.9, 0.9, 0.9};
       cfg.setHenyeyGreenstein(0.8);  // all layers

       // Boundary conditions
       cfg.bc.direct_beam_flux = 3.14159;
       cfg.bc.direct_beam_mu = 0.5;
       cfg.bc.surface_albedo = 0.1;

       // Solve
       DisortSolver solver;
       DisortResult result = solver.solve(cfg);

       // Access results
       double flux_up_toa = result.flux_up[0];
       double flux_down_boa = result.totalFluxDown(nlyr);
   }


Python quick start
------------------

The Python interface mirrors the C++ API closely:

.. code-block:: python

   import disortpp
   import numpy as np

   nlyr = 4
   nstr = 16

   # Create and configure
   cfg = disortpp.DisortConfig(nlyr, nstr)
   cfg.flags.use_lambertian_surface = True
   cfg.flags.comp_only_fluxes = True
   cfg.allocate()

   # Layer optical properties
   cfg.delta_tau = [0.1, 0.5, 1.0, 2.0]
   cfg.single_scat_albedo = [0.9, 0.9, 0.9, 0.9]
   cfg.set_henyey_greenstein(g=0.8)

   # Boundary conditions
   cfg.bc.direct_beam_flux = np.pi
   cfg.bc.direct_beam_mu = 0.5
   cfg.bc.surface_albedo = 0.1

   # Solve
   solver = disortpp.DisortSolver()
   result = solver.solve(cfg)

   # Access results
   flux_up = np.array(result.flux_up)
   flux_down = np.array(result.flux_down)
   print(f"Upward flux at TOA: {flux_up[0]:.6f}")


Workflow overview
-----------------

Every DisORT++ calculation follows the same three steps:

1. **Configure** -- create a configuration object (``DisortConfig`` or
   ``DisortFluxConfig``), set dimensions, flags, optical properties, boundary
   conditions, and call ``allocate()``.

2. **Solve** -- pass the configuration to a solver (``DisortSolver`` or
   ``DisortFluxSolver``).

3. **Extract results** -- read fluxes, mean intensities, and (if requested)
   angular intensities from the result object.


Choosing a solver
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Criterion
     - DisortSolver
     - DisortFluxSolver
   * - Output
     - Fluxes + intensities
     - Fluxes only
   * - Surface model
     - Lambertian + BRDF
     - Lambertian only
   * - Stream count
     - Any even number >= 4
     - 4, 6, 8, 10, 12, 14, 16, 32, 64
   * - Performance
     - Baseline
     - 30--50 % faster
   * - Config class
     - ``DisortConfig``
     - ``DisortFluxConfig``


Level and layer indexing
^^^^^^^^^^^^^^^^^^^^^^^^

DisORT++ uses a layered atmosphere model. **Levels** are the boundaries
between layers, numbered from 0 (top of atmosphere, TOA) to *N*
(bottom of atmosphere, BOA), where *N* is the number of layers:

.. code-block:: text

   Level 0  ──────────────  TOA
            │  Layer 0   │
   Level 1  ──────────────
            │  Layer 1   │
   Level 2  ──────────────
            │    ...     │
   Level N  ──────────────  BOA

- ``delta_tau[i]`` and ``single_scat_albedo[i]`` describe **layer** *i*
  (between levels *i* and *i + 1*).
- ``temperature[i]`` describes **level** *i*.
- Flux arrays (``flux_up``, ``flux_down``, etc.) are indexed by level.
