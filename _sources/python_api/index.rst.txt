Python API Reference
====================

The ``disortpp`` Python module exposes the full DisORT++ C++ API through
pybind11 bindings. Build with ``-DBUILD_PYTHON_BINDINGS=ON`` to compile the
module (see :doc:`/installation`).

.. code-block:: python

   import disortpp

The Python API closely mirrors the C++ interface, with the following naming
conventions:

- C++ ``camelCase`` methods become Python ``snake_case``
  (e.g., ``setHenyeyGreenstein`` becomes ``set_henyey_greenstein``).
- C++ ``enum class`` values are accessed as attributes
  (e.g., ``disortpp.PhaseFunction.Rayleigh``).
- C++ ``std::vector<double>`` maps to Python ``list[float]``. Results can
  be converted to NumPy arrays with ``np.array(result.flux_up)``.
- C++ ``std::vector<std::pair<double, double>>`` maps to Python
  ``list[tuple[float, float]]``.

All solver calls release the Python GIL, allowing concurrent Python threads
to proceed during computation.


Enumerations
------------

.. code-block:: python

   disortpp.PhaseFunction.Isotropic
   disortpp.PhaseFunction.Rayleigh
   disortpp.PhaseFunction.HenyeyGreenstein
   disortpp.PhaseFunction.HazeGarciaSiewert
   disortpp.PhaseFunction.CloudGarciaSiewert

   disortpp.BoundaryConditionType.General
   disortpp.BoundaryConditionType.Special

   disortpp.BrdfType.NONE
   disortpp.BrdfType.RPV
   disortpp.BrdfType.CoxMunk
   disortpp.BrdfType.Ambrals
   disortpp.BrdfType.Hapke


Contents
--------

.. toctree::
   :maxdepth: 2

   config
   solvers
   results
   spectral
