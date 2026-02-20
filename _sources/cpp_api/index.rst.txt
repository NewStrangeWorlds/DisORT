C++ API Reference
=================

The DisORT++ C++ API consists of configuration classes, solvers, result
containers, and utility functions. All types reside in the ``disortpp``
namespace.

Header files are located in the ``src/`` directory. Link against the
``disortpp`` CMake target to use the library.

.. code-block:: cpp

   #include "DisortConfig.hpp"
   #include "DisortSolver.hpp"
   // ... other headers as needed

   using namespace disortpp;

.. toctree::
   :maxdepth: 2

   types
   config
   solvers
   results
   utilities
