Enumerations and Constants
==========================

Defined in ``DisortTypes.hpp``.

.. contents:: On this page
   :local:
   :depth: 1


PhaseFunction
-------------

.. code-block:: cpp

   enum class PhaseFunction {
       Isotropic          = 1,
       Rayleigh           = 2,
       HenyeyGreenstein   = 3,
       HazeGarciaSiewert  = 4,
       CloudGarciaSiewert = 5
   };

Selects the scattering phase function type used by the convenience methods
``setPhaseFunction()``, ``setHenyeyGreenstein()``, etc.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``Isotropic``
     - Isotropic scattering (all moments zero except the zeroth).
   * - ``Rayleigh``
     - Rayleigh scattering (moment 2 = 0.1).
   * - ``HenyeyGreenstein``
     - Henyey-Greenstein phase function with asymmetry parameter *g*.
       Moments follow :math:`\chi_k = g^k`.
   * - ``HazeGarciaSiewert``
     - Haze-L phase function from Garcia & Siewert (1985).
   * - ``CloudGarciaSiewert``
     - Cloud C.1 phase function from Garcia & Siewert (1985).


BoundaryConditionType
---------------------

.. code-block:: cpp

   enum class BoundaryConditionType {
       General = 0,
       Special = 1
   };

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Description
   * - ``General``
     - Standard boundary conditions with specified top/bottom illumination.
   * - ``Special``
     - Return only the medium's albedo and transmissivity (no specific
       illumination).


IlluminationType
----------------

.. code-block:: cpp

   enum class IlluminationType {
       Top    = 1,
       Bottom = 2
   };

Specifies the direction of external illumination.


BrdfType
--------

.. code-block:: cpp

   enum class BrdfType {
       None    = 0,
       RPV     = 1,
       CoxMunk = 2,
       Ambrals = 3,
       Hapke   = 4
   };

Selects the bidirectional reflectance distribution function (BRDF) model for
the bottom boundary.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Description
   * - ``None``
     - No BRDF; use Lambertian surface (``surface_albedo``).
   * - ``RPV``
     - Rahman-Pinty-Verstraete model.
   * - ``CoxMunk``
     - Cox & Munk ocean surface model.
   * - ``Ambrals``
     - Ambrals (Ross-Li) kernel-based model.
   * - ``Hapke``
     - Hapke opposition-effect model.


MessageType
-----------

.. code-block:: cpp

   enum class MessageType {
       Warning = 0,
       Error   = 1
   };

Internal diagnostic message severity.


VerbosityLevel
--------------

.. code-block:: cpp

   enum class VerbosityLevel {
       Verbose = 0,
       Quiet   = 1
   };

Controls solver diagnostic output.


Constants
---------

.. code-block:: cpp

   constexpr double DEG  = M_PI / 180.0;   // degrees-to-radians conversion
   constexpr int    NMUG = 200;             // quadrature points for BRDF integration
