Installation
============

Requirements
------------

- **C++17** compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake** 3.15 or later
- **Eigen 3.4** (fetched automatically via CMake FetchContent)

Optional:

- **pybind11** 2.13+ (fetched automatically when building Python bindings)
- **OpenMP** (detected automatically; enables parallel spectral loops)
- **Python** 3.8+ with **NumPy** (for using the Python interface)


Building the C++ library
-------------------------

.. code-block:: bash

   cd DisORT
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)

CMake will download Eigen 3.4 into ``build/_deps/`` on the first build.
Subsequent builds reuse the cached download.

The build produces the static library ``libdisortpp.a`` and the test
executable ``tests/disortpp_tests``.


Build options
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Option
     - Default
     - Description
   * - ``CMAKE_BUILD_TYPE``
     - ``Release``
     - ``Release`` (``-O3 -march=native``) or ``Debug`` (``-g -Wall``)
   * - ``BUILD_PYTHON_BINDINGS``
     - ``OFF``
     - Build the ``disortpp`` Python module via pybind11


Building with Python bindings
-----------------------------

.. code-block:: bash

   cd DisORT
   mkdir build && cd build
   cmake .. -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)

This additionally produces the shared library
``disortpp.cpython-3XX-x86_64-linux-gnu.so`` (the exact filename depends on
your Python version and platform).

To use the module, either run Python from the ``build/`` directory or add
it to ``PYTHONPATH``:

.. code-block:: bash

   export PYTHONPATH=/path/to/DisORT/build:$PYTHONPATH
   python3 -c "import disortpp; print('OK')"


Running the tests
-----------------

.. code-block:: bash

   cd build
   ctest --output-on-failure

Or run the test binary directly for verbose output:

.. code-block:: bash

   ./tests/disortpp_tests


Using DisORT++ in your CMake project
------------------------------------

Add DisORT++ as a subdirectory or use ``FetchContent``:

.. code-block:: cmake

   add_subdirectory(DisORT)

   add_executable(my_app main.cpp)
   target_link_libraries(my_app PRIVATE disortpp)

The ``disortpp`` target transitively provides the Eigen include directories,
so no additional ``find_package(Eigen3)`` call is needed.
