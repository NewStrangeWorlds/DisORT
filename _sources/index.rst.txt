DisORT++ Documentation
======================

**DisORT++** is a modern C++17 implementation of the DISORT (Discrete Ordinates
Radiative Transfer) algorithm. It solves the radiative transfer equation in a
plane-parallel or pseudo-spherical atmosphere, computing fluxes, mean
intensities, and angular intensities at arbitrary optical depths and directions.

DisORT++ provides two solver interfaces:

- **DisortSolver** -- the general-purpose solver supporting full intensity
  calculations, user-specified output angles, BRDF surface models, and
  multiple intensity correction methods.
- **DisortFluxSolver** -- a performance-optimised flux-only solver with
  compile-time stream counts, delivering 30--50 % faster execution for
  flux-only applications.

Both solvers are accessible from C++ and Python (via pybind11 bindings).


Features
--------

- Discrete-ordinate method with arbitrary even stream count
- Henyey-Greenstein, Rayleigh, and tabulated phase functions
- Delta-M and Delta-M+ scaling for forward-peaked scattering
- Thermal emission with Planck-function integration
- Lambertian and BRDF surface reflection (RPV, Cox-Munk, Ambrals, Hapke)
- Plane-parallel and pseudo-spherical beam geometry
- Nakajima-Tanaka and Buras-Emde intensity corrections
- Spectral loop functions with OpenMP parallelisation for efficient
  multi-wavenumber calculations
- Python bindings via pybind11


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 3
   :caption: C++ API Reference

   cpp_api/index

.. toctree::
   :maxdepth: 3
   :caption: Python API Reference

   python_api/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
