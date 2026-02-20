Python Solvers
==============

.. contents:: On this page
   :local:
   :depth: 1


DisortSolver
------------

General-purpose radiative transfer solver.

.. code-block:: python

   solver = disortpp.DisortSolver()
   result = solver.solve(cfg)   # cfg is a DisortConfig

.. method:: solver.solve(config)

   Solve the radiative transfer problem.

   :param config: ``DisortConfig`` instance. May be modified by the solver.
   :returns: ``DisortResult`` with fluxes and (optionally) intensities.
   :raises ValueError: if the configuration is invalid.

   The GIL is released during the C++ computation, allowing other Python
   threads to run.


DisortFluxSolver
----------------

Flux-only solver with compile-time stream count. Individual classes are
available for each supported stream count:

.. code-block:: python

   solver = disortpp.DisortFluxSolver4()    # 4 streams
   solver = disortpp.DisortFluxSolver6()    # 6 streams
   solver = disortpp.DisortFluxSolver8()    # 8 streams
   solver = disortpp.DisortFluxSolver10()   # 10 streams
   solver = disortpp.DisortFluxSolver12()   # 12 streams
   solver = disortpp.DisortFluxSolver14()   # 14 streams
   solver = disortpp.DisortFluxSolver16()   # 16 streams
   solver = disortpp.DisortFluxSolver32()   # 32 streams
   solver = disortpp.DisortFluxSolver64()   # 64 streams

.. method:: solver.solve(config)
   :no-index:

   Solve for fluxes only.

   :param config: ``DisortFluxConfig`` instance. Not modified.
   :returns: ``FluxResult`` with fluxes at all layer boundaries.


create_flux_solver
------------------

Factory function that creates the appropriate ``DisortFluxSolver`` for the
given stream count.

.. code-block:: python

   solver = disortpp.create_flux_solver(nstr)

.. function:: disortpp.create_flux_solver(nstr)

   :param nstr: Number of streams (4, 6, 8, 10, 12, 14, 16, 32, or 64).
   :returns: A ``DisortFluxSolverN`` instance.
   :raises ValueError: if ``nstr`` is not a supported value.

Example:

.. code-block:: python

   cfg = disortpp.DisortFluxConfig(6, 16)
   cfg.direct_beam_flux = np.pi
   cfg.direct_beam_mu = 0.5
   cfg.surface_albedo = 0.1
   cfg.allocate()

   cfg.delta_tau = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
   cfg.single_scat_albedo = [0.9] * 6
   cfg.set_henyey_greenstein(g=0.8)

   solver = disortpp.create_flux_solver(16)
   result = solver.solve(cfg)

   flux_up = np.array(result.flux_up)
