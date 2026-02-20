Solvers
=======

Defined in ``DisortSolver.hpp`` and ``FluxSolver.hpp``.

.. contents:: On this page
   :local:
   :depth: 1


DisortSolver
------------

The general-purpose discrete ordinates radiative transfer solver. Supports
full intensity calculations, user-specified output angles, BRDF surfaces,
thermal emission, and intensity corrections.

Defined in ``DisortSolver.hpp``.

.. code-block:: cpp

   class DisortSolver {
   public:
       DisortSolver();
       DisortResult solve(DisortConfig& config);
   };

Constructor
^^^^^^^^^^^

.. function:: DisortSolver()

   Default constructor. The solver is lightweight and allocates memory
   only during ``solve()``.

solve
^^^^^

.. function:: DisortResult solve(DisortConfig& config)

   Solve the radiative transfer problem.

   :param config: Input configuration. **Note:** the solver may modify
      the configuration to set automatic dimensions (e.g.,
      ``num_phase_func_moments`` when using a named phase function).
   :returns: ``DisortResult`` containing fluxes, mean intensities, and
      (optionally) angular intensities.
   :throws std\:\:invalid_argument: if the configuration is invalid.
   :throws std\:\:runtime_error: if a numerical problem occurs.

   The solver internally performs:

   1. Configuration validation and optional Delta-M(+) scaling.
   2. Eigenvalue/eigenvector computation for each layer.
   3. Boundary condition assembly and linear system solve.
   4. Flux and (optionally) intensity computation at all output levels.

   The solver object can be reused for multiple calls. Internal temporaries
   are re-allocated as needed.


DisortFluxSolver<NStr>
----------------------

A performance-optimised flux-only solver with compile-time stream count.
Uses fixed-size Eigen matrices for the eigenvalue problem, yielding
30--50 % faster execution compared to ``DisortSolver`` in flux-only mode.

Defined in ``FluxSolver.hpp``.

.. code-block:: cpp

   template<int NStr>
   class DisortFluxSolver {
   public:
       static constexpr int NN = NStr / 2;

       DisortFluxSolver();
       FluxResult solve(const DisortFluxConfig& config);
   };

Template parameter
^^^^^^^^^^^^^^^^^^

``NStr`` -- the number of streams. Must be even and >= 4. The following
instantiations are provided:

- ``DisortFluxSolver<4>``
- ``DisortFluxSolver<6>``
- ``DisortFluxSolver<8>``
- ``DisortFluxSolver<10>``
- ``DisortFluxSolver<12>``
- ``DisortFluxSolver<14>``
- ``DisortFluxSolver<16>``
- ``DisortFluxSolver<32>``
- ``DisortFluxSolver<64>``

Constructor
^^^^^^^^^^^

.. function:: DisortFluxSolver()

   Default constructor. Lightweight; no allocation until ``solve()`` is
   called.

solve
^^^^^

.. function:: FluxResult solve(const DisortFluxConfig& config)

   Solve for fluxes and mean intensities.

   :param config: Input configuration. Must have
      ``config.num_streams == NStr``. The configuration is not modified
      (taken by const reference).
   :returns: ``FluxResult`` with fluxes at all ``num_layers + 1`` level
      boundaries.
   :throws std\:\:invalid_argument: if validation fails.

Limitations
^^^^^^^^^^^

Compared to ``DisortSolver``, the flux solver does **not** support:

- Angular intensity output.
- User-specified output optical depths (output is always at all layer
  boundaries).
- BRDF surface models (Lambertian only).
- Intensity corrections (Nakajima-Tanaka, Buras-Emde).
- Special boundary condition mode.


Usage example
^^^^^^^^^^^^^

.. code-block:: cpp

   #include "DisortFluxConfig.hpp"
   #include "FluxSolver.hpp"

   using namespace disortpp;

   DisortFluxConfig cfg(6, 16);
   cfg.direct_beam_flux = 3.14159;
   cfg.direct_beam_mu = 0.5;
   cfg.surface_albedo = 0.1;
   cfg.allocate();

   cfg.delta_tau = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
   cfg.single_scat_albedo = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9};
   cfg.setHenyeyGreenstein(0.8);

   DisortFluxSolver<16> solver;
   FluxResult result = solver.solve(cfg);

   double flux_up_toa = result.flux_up[0];
