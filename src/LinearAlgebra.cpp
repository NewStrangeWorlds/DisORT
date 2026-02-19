#include "LinearAlgebra.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace disortpp {
namespace LinearAlgebra {

// ============================================================================
// Dense Matrix Operations
// ============================================================================

Eigen::VectorXd solveDense(const Eigen::MatrixXd& A,
               const Eigen::VectorXd& b,
               double* rcond) 
{
  const int n = A.rows();

  if (A.cols() != n) {
    throw std::invalid_argument("solveDense: Matrix must be square");
  }
  if (b.size() != n) {
    throw std::invalid_argument("solveDense: b must match matrix size");
  }

  // Compute 1-norm of A (for condition number estimation)
  double norm1_A = matrixNorm1(A);

  // Compute LU decomposition with partial pivoting
  Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);

  // Estimate condition number if requested
  if (rcond != nullptr) {
    *rcond = estimateRcond(lu, norm1_A);

    // Warn if matrix is ill-conditioned
    if (*rcond < std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
        "solveDense: Matrix is singular or nearly singular (rcond = " +
        std::to_string(*rcond) + ")");
    }
  }

  // Solve A*x = b
  Eigen::VectorXd x = lu.solve(b);

  // Verify solution quality
  double relative_error = (A * x - b).norm() / b.norm();
  if (relative_error > 1e-6) {
    throw std::runtime_error(
      "solveDense: Solution has large residual error (rel_err = " +
      std::to_string(relative_error) + ")");
  }

  return x;
}

Eigen::PartialPivLU<Eigen::MatrixXd> factorizeDense(Eigen::MatrixXd& A) 
{
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("factorizeDense: Matrix must be square");
  }

  return Eigen::PartialPivLU<Eigen::MatrixXd>(A);
}

Eigen::VectorXd solveDenseFactored(const Eigen::PartialPivLU<Eigen::MatrixXd>& lu,
                   const Eigen::VectorXd& b) 
{
  return lu.solve(b);
}

// ============================================================================
// Band Matrix Operations
// ============================================================================

Eigen::VectorXd solveBand(const Eigen::MatrixXd& A,
              const Eigen::VectorXd& b,
              int kl, int ku,
              double* rcond) 
{
  const int n = A.rows();

  if (A.cols() != n) {
    throw std::invalid_argument("solveBand: Matrix must be square");
  }
  if (b.size() != n) {
    throw std::invalid_argument("solveBand: b must match matrix size");
  }
  if (kl < 0 || ku < 0) {
    throw std::invalid_argument("solveBand: Bandwidths must be non-negative");
  }

  // For now, use dense solver
  // TODO: Optimize using Eigen's band matrix support if profiling shows need
  //
  // Note: Eigen doesn't have direct LU for band matrices in the same way LINPACK does.
  // Options:
  // 1. Use dense solver (simple, correct, but slower for large sparse matrices)
  // 2. Use Eigen::SparseMatrix with sparse LU (more complex setup)
  // 3. Manually implement band LU (most work)
  //
  // For CDISORT, matrices are typically small (num_streams = 16-32), so dense is fine.

  return solveDense(A, b, rcond);
}

// ============================================================================
// Eigenvalue/Eigenvector Computation
// ============================================================================

bool hasRealEigenvalues(const Eigen::VectorXcd& eigenvalues, double tolerance) 
{
  for (int i = 0; i < eigenvalues.size(); ++i) {
    if (std::abs(eigenvalues(i).imag()) > tolerance) {
      return false;
    }
  }
  return true;
}

Eigen::VectorXd extractRealEigenvalues(const Eigen::VectorXcd& eigenvalues,
                     double tolerance) 
{
  if (!hasRealEigenvalues(eigenvalues, tolerance)) {
    // Find max imaginary part for error message
    double max_imag = 0.0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
      max_imag = std::max(max_imag, std::abs(eigenvalues(i).imag()));
    }

    throw std::runtime_error(
      "extractRealEigenvalues: Matrix has complex eigenvalues "
      "(max imag part = " + std::to_string(max_imag) + ")");
  }

  Eigen::VectorXd real_eigenvalues(eigenvalues.size());
  for (int i = 0; i < eigenvalues.size(); ++i) {
    real_eigenvalues(i) = eigenvalues(i).real();
  }

  return real_eigenvalues;
}

void computeEigensystem(const Eigen::MatrixXd& A,
            Eigen::VectorXd& eigenvalues,
            Eigen::MatrixXd& eigenvectors,
            bool sort_ascending) 
{
  const int n = A.rows();

  if (A.cols() != n) {
    throw std::invalid_argument("computeEigensystem: Matrix must be square");
  }

  // Compute eigenvalues and eigenvectors
  // Note: EigenSolver handles general real matrices (not necessarily symmetric)
  Eigen::EigenSolver<Eigen::MatrixXd> solver(A, /* computeEigenvectors = */ true);

  // Extract eigenvalues (ensure they are real)
  Eigen::VectorXcd complex_eigenvalues = solver.eigenvalues();
  eigenvalues = extractRealEigenvalues(complex_eigenvalues);

  // Extract eigenvectors (take real parts)
  Eigen::MatrixXcd complex_eigenvectors = solver.eigenvectors();
  eigenvectors.resize(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      eigenvectors(i, j) = complex_eigenvectors(i, j).real();
    }
  }

  // Sort eigenvalues and eigenvectors if requested
  if (sort_ascending) {
    // Create index array for sorting
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) {
      indices[i] = i;
    }

    // Sort indices based on eigenvalue magnitude (absolute value)
    // This matches the C code's c_asymmetric_matrix sorting convention
    std::sort(indices.begin(), indices.end(),
          [&eigenvalues](int i, int j) {
            return std::abs(eigenvalues(i)) < std::abs(eigenvalues(j));
          });

    // Reorder eigenvalues and eigenvectors
    Eigen::VectorXd sorted_eigenvalues(n);
    Eigen::MatrixXd sorted_eigenvectors(n, n);

    for (int i = 0; i < n; ++i) {
      sorted_eigenvalues(i) = eigenvalues(indices[i]);
      sorted_eigenvectors.col(i) = eigenvectors.col(indices[i]);
    }

    eigenvalues = sorted_eigenvalues;
    eigenvectors = sorted_eigenvectors;
  }
}

// ============================================================================
// Matrix Utilities
// ============================================================================

double estimateRcond(const Eigen::PartialPivLU<Eigen::MatrixXd>& lu,
           double norm1_A) 
{
  // Eigen doesn't provide LINPACK-style condition number estimation directly.
  // We use a simplified approach:
  //
  // 1. Full approach (expensive): rcond = 1 / (||A|| * ||A^-1||)
  // 2. Simplified (cheap): Check for small pivots in LU factorization
  //
  // For now, use simplified approach based on determinant
  // (small determinant indicates near-singularity)

  const Eigen::MatrixXd& L = lu.matrixLU().triangularView<Eigen::Lower>();
  const Eigen::MatrixXd& U = lu.matrixLU().triangularView<Eigen::Upper>();

  // Compute product of diagonal elements (determinant)
  double det = 1.0;
  int n = lu.matrixLU().rows();
  for (int i = 0; i < n; ++i) {
    det *= lu.matrixLU()(i, i);  // Diagonal elements of U
  }
  det = std::abs(det);

  // Simple estimate: if determinant is very small relative to norm, matrix is ill-conditioned
  if (det < std::numeric_limits<double>::epsilon() * norm1_A) {
    return det / norm1_A;  // Very small rcond
  }

  // Better estimate: Use inverse matrix norm (expensive but accurate)
  // rcond = 1 / (||A|| * ||A^-1||)
  //
  // We approximate ||A^-1|| by solving A*X = I for a few columns
  // and taking the maximum column sum (1-norm)

  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  Eigen::MatrixXd A_inv = lu.solve(I);

  double norm1_A_inv = matrixNorm1(A_inv);
  double rcond = 1.0 / (norm1_A * norm1_A_inv);

  return rcond;
}

} // namespace LinearAlgebra
} // namespace disortpp
