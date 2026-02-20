Utility Functions
=================

.. contents:: On this page
   :local:
   :depth: 1


Planck function
---------------

Defined in ``Planck.hpp``.

.. function:: double planckFunction(double wnumlo, double wnumhi, double temp)

   Compute the Planck function integrated between two wavenumbers.

   :param wnumlo: Lower wavenumber [cm\ :sup:`-1`].
   :param wnumhi: Upper wavenumber [cm\ :sup:`-1`]. Must be > ``wnumlo``.
   :param temp: Temperature [K]. Must be > 0.
   :returns: Integrated Planck function [W/m\ :sup:`2`].
   :throws std\:\:invalid_argument: if parameters are out of range.

   Accuracy is at least 6 significant digits. Uses power series for
   efficiency at large arguments and the standard integral otherwise.

Constants (``PlanckConstants`` namespace):

- ``C2 = 1.438786`` -- :math:`hc/k` [cm K]
- ``SIGMA = 5.67032 \times 10^{-8}`` -- Stefan-Boltzmann constant
  [W/m\ :sup:`2`/K\ :sup:`4`]


Phase function utilities
------------------------

Defined in ``PhaseFunction.hpp``.

These functions fill the phase function Legendre moment arrays. They are
called internally by the ``setHenyeyGreenstein()`` etc. methods on the
configuration classes and are also available for direct use.

All functions operate on a 2D moments array
(``vector<vector<double>>&``). The parameter ``lc`` selects a specific
layer (0-based); when ``lc = -1``, all layers are filled.

.. function:: void phase_function::fillHenyeyGreenstein(moments, nmom_nstr, num_layers, g, lc = -1)

   Fill with Henyey-Greenstein moments: :math:`\chi_k = g^k`.

.. function:: void phase_function::fillIsotropic(moments, nmom_nstr, num_layers, lc = -1)

   Fill with isotropic scattering (all moments zero except zeroth).

.. function:: void phase_function::fillRayleigh(moments, nmom_nstr, num_layers, lc = -1)

   Fill with Rayleigh scattering moments (moment 2 = 0.1).

.. function:: void phase_function::fillHazeGarciaSiewert(moments, nmom_nstr, num_layers, lc = -1)

   Fill with the Haze-L phase function from Garcia & Siewert (1985).

.. function:: void phase_function::fillCloudGarciaSiewert(moments, nmom_nstr, num_layers, lc = -1)

   Fill with the Cloud C.1 phase function from Garcia & Siewert (1985).

.. function:: void phase_function::fillPhaseFunction(moments, nmom_nstr, num_layers, type, g = 0.0, lc = -1)

   Dispatch to one of the above functions based on the ``PhaseFunction``
   enum value.


Legendre polynomials
--------------------

Defined in ``Legendre.hpp``.

.. function:: void legendrePolynomials(int nmu, int m, int twonm1, const vector<double>& mu, vector<vector<double>>& ylm)

   Compute normalised associated Legendre polynomials.

   :param nmu: Number of angle cosines.
   :param m: Azimuthal order (0 for ordinary Legendre polynomials).
   :param twonm1: Highest polynomial degree to compute.
   :param mu: Angle cosines (values in [-1, 1]) [``nmu``].
   :param ylm: Output array [``twonm1+1``][``nmu``].


Linear algebra
--------------

Defined in ``LinearAlgebra.hpp``. These are Eigen-based wrappers that
replace the original LINPACK/LAPACK routines from the Fortran DISORT code.

Dense solvers
^^^^^^^^^^^^^

.. function:: VectorXd LinearAlgebra::solveDense(const MatrixXd& A, const VectorXd& b, double* rcond = nullptr)

   Solve the dense linear system :math:`Ax = b` using LU decomposition.
   Optionally returns the reciprocal condition number in ``rcond``.

.. function:: PartialPivLU<MatrixXd> LinearAlgebra::factorizeDense(MatrixXd& A)

   Compute the LU factorisation of *A*.

.. function:: VectorXd LinearAlgebra::solveDenseFactored(const PartialPivLU<MatrixXd>& lu, const VectorXd& b)

   Solve using a pre-computed LU factorisation.

Band solver
^^^^^^^^^^^

.. function:: VectorXd LinearAlgebra::solveBand(const MatrixXd& A, const VectorXd& b, int kl, int ku, double* rcond = nullptr)

   Solve a banded linear system with ``kl`` sub-diagonals and ``ku``
   super-diagonals.

Eigenvalue computation
^^^^^^^^^^^^^^^^^^^^^^

.. function:: void LinearAlgebra::computeEigensystem(const MatrixXd& A, VectorXd& eigenvalues, MatrixXd& eigenvectors, bool sort_ascending = true)

   Compute eigenvalues and eigenvectors of a real asymmetric matrix.
   Eigenvalues are sorted in ascending order by default.

Matrix utilities
^^^^^^^^^^^^^^^^

.. function:: double LinearAlgebra::matrixNorm1(const MatrixXd& A)

   Compute the 1-norm (maximum absolute column sum) of *A*.

.. function:: double LinearAlgebra::estimateRcond(const PartialPivLU<MatrixXd>& lu, double norm1_A)

   Estimate the reciprocal condition number from a pre-computed LU
   factorisation and the 1-norm of the original matrix.
