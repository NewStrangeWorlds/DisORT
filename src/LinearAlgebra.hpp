#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <stdexcept>

namespace disortpp {

/**
 * @brief Linear algebra operations using Eigen
 *
 * This module provides modern C++ wrappers around Eigen library operations,
 * replacing the original LINPACK/BLAS routines used in CDISORT.
 *
 * Design notes:
 * - Uses Eigen for all matrix operations
 * - Maintains similar interfaces to original C functions for easier integration
 * - All functions use 0-based indexing (unlike original 1-based C code)
 * - Exceptions thrown for invalid inputs
 */

namespace LinearAlgebra {

// ============================================================================
// BLAS Level 1 Operations
// ============================================================================

/**
 * @brief Sum of absolute values (BLAS: SASUM)
 *
 * Computes: sum(|x_i|)
 *
 * @param x Input vector
 * @return Sum of absolute values
 */
inline double vectorAbsSum(const Eigen::VectorXd& x) {
  return x.cwiseAbs().sum();
}

/**
 * @brief Scaled vector addition (BLAS: SAXPY)
 *
 * Computes: y = a*x + y
 *
 * @param a Scalar multiplier
 * @param x Input vector
 * @param y Vector to be updated (modified in place)
 */
inline void vectorSaxpy(double a, const Eigen::VectorXd& x, Eigen::VectorXd& y) {
  if (x.size() != y.size()) {
    throw std::invalid_argument("vectorSaxpy: x and y must have same size");
  }
  y += a * x;
}

/**
 * @brief Dot product (BLAS: SDOT)
 *
 * Computes: sum(x_i * y_i)
 *
 * @param x First vector
 * @param y Second vector
 * @return Dot product
 */
inline double vectorDot(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
  if (x.size() != y.size()) {
    throw std::invalid_argument("vectorDot: x and y must have same size");
  }
  return x.dot(y);
}

/**
 * @brief Scale vector (BLAS: SSCAL)
 *
 * Computes: x = a*x
 *
 * @param a Scalar multiplier
 * @param x Vector to be scaled (modified in place)
 */
inline void vectorScale(double a, Eigen::VectorXd& x) {
  x *= a;
}

/**
 * @brief Index of maximum absolute value (BLAS: ISAMAX)
 *
 * @param x Input vector
 * @return Index of element with maximum absolute value (0-based)
 */
inline int vectorMaxAbsIndex(const Eigen::VectorXd& x) {
  int max_idx = 0;
  x.cwiseAbs().maxCoeff(&max_idx);
  return max_idx;
}

// ============================================================================
// Dense Matrix Operations (replacing LINPACK SGECO/SGEFA/SGESL)
// ============================================================================

/**
 * @brief Dense matrix solver with condition number estimation
 *
 * Solves A*x = b for dense matrix A, with condition number estimation.
 * Replaces LINPACK's sgeco/sgefa/sgesl sequence.
 *
 * @param A Coefficient matrix (n x n)
 * @param b Right-hand side vector (n)
 * @param rcond [out] Reciprocal condition number estimate
 * @return Solution vector x
 * @throws std::runtime_error if matrix is singular or nearly singular
 */
Eigen::VectorXd solveDense(const Eigen::MatrixXd& A,
               const Eigen::VectorXd& b,
               double* rcond = nullptr);

/**
 * @brief Dense matrix LU factorization
 *
 * Computes LU factorization with partial pivoting: P*A = L*U
 * Replaces LINPACK's sgefa.
 *
 * @param A Matrix to factor (modified in place)
 * @return LU decomposition object
 */
Eigen::PartialPivLU<Eigen::MatrixXd> factorizeDense(Eigen::MatrixXd& A);

/**
 * @brief Solve using pre-factored dense matrix
 *
 * Solves A*x = b where A has already been factored.
 * Replaces LINPACK's sgesl.
 *
 * @param lu Pre-computed LU factorization
 * @param b Right-hand side vector
 * @return Solution vector x
 */
Eigen::VectorXd solveDenseFactored(const Eigen::PartialPivLU<Eigen::MatrixXd>& lu,
                   const Eigen::VectorXd& b);

// ============================================================================
// Band Matrix Operations (replacing LINPACK SGBCO/SGBFA/SGBSL)
// ============================================================================

/**
 * @brief Band matrix solver
 *
 * Solves A*x = b for band matrix A.
 * Replaces LINPACK's sgbco/sgbfa/sgbsl sequence.
 *
 * Band storage format:
 * - kl: number of subdiagonals
 * - ku: number of superdiagonals
 * - Total bandwidth: kl + ku + 1
 *
 * @param A Band matrix (n x n) with bandwidth kl + ku + 1
 * @param b Right-hand side vector (n)
 * @param kl Number of subdiagonals
 * @param ku Number of superdiagonals
 * @param rcond [out] Reciprocal condition number estimate
 * @return Solution vector x
 */
Eigen::VectorXd solveBand(const Eigen::MatrixXd& A,
              const Eigen::VectorXd& b,
              int kl, int ku,
              double* rcond = nullptr);

// ============================================================================
// Eigenvalue/Eigenvector Computation (replacing c_asymmetric_matrix)
// ============================================================================

/**
 * @brief Eigenvalue and eigenvector computation for real asymmetric matrices
 *
 * Computes eigenvalues and eigenvectors of a real asymmetric matrix.
 * Assumes all eigenvalues are real (appropriate for DISORT's use case).
 *
 * Replaces c_asymmetric_matrix() which used:
 * - Balancing (BALANC)
 * - Hessenberg reduction (ELMHES)
 * - QR algorithm (HQR, ELTRAN)
 *
 * @param A Input matrix (n x n)
 * @param eigenvalues [out] Real eigenvalues (sorted)
 * @param eigenvectors [out] Corresponding eigenvectors (columns)
 * @param sort_ascending If true, sort eigenvalues in ascending order
 * @throws std::runtime_error if eigenvalues have imaginary parts
 */
void computeEigensystem(const Eigen::MatrixXd& A,
            Eigen::VectorXd& eigenvalues,
            Eigen::MatrixXd& eigenvectors,
            bool sort_ascending = true);

/**
 * @brief Check if matrix has real eigenvalues
 *
 * Helper function to validate that eigenvalues are real (imaginary parts negligible).
 * Used by computeEigensystem to ensure assumptions are met.
 *
 * @param eigenvalues Complex eigenvalues from Eigen solver
 * @param tolerance Maximum allowed imaginary part (default: 1e-10)
 * @return true if all eigenvalues are real within tolerance
 */
bool hasRealEigenvalues(const Eigen::VectorXcd& eigenvalues,
            double tolerance = 1e-10);

/**
 * @brief Extract real parts of eigenvalues
 *
 * Extracts real parts and validates imaginary parts are negligible.
 *
 * @param eigenvalues Complex eigenvalues
 * @param tolerance Maximum allowed imaginary part
 * @return Real parts of eigenvalues
 * @throws std::runtime_error if imaginary parts exceed tolerance
 */
Eigen::VectorXd extractRealEigenvalues(const Eigen::VectorXcd& eigenvalues,
                     double tolerance = 1e-10);

// ============================================================================
// Matrix Utilities
// ============================================================================

/**
 * @brief Compute matrix 1-norm (maximum absolute column sum)
 *
 * Used for condition number estimation.
 *
 * @param A Input matrix
 * @return 1-norm of A
 */
inline double matrixNorm1(const Eigen::MatrixXd& A) {
  double max_sum = 0.0;
  for (int j = 0; j < A.cols(); ++j) {
    double col_sum = A.col(j).cwiseAbs().sum();
    max_sum = std::max(max_sum, col_sum);
  }
  return max_sum;
}

/**
 * @brief Estimate reciprocal condition number
 *
 * Estimates 1/cond(A) where cond(A) = ||A|| * ||A^-1||.
 * Used to detect near-singularity.
 *
 * @param lu LU decomposition of matrix
 * @param norm1_A 1-norm of original matrix
 * @return Estimated reciprocal condition number
 */
double estimateRcond(const Eigen::PartialPivLU<Eigen::MatrixXd>& lu,
           double norm1_A);

} // namespace LinearAlgebra
} // namespace disortpp
