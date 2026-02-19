#include "testing.hpp"
#include "LinearAlgebra.hpp"
#include <Eigen/Dense>
#include <cmath>

using namespace disortpp::LinearAlgebra;
using namespace Eigen;

// ============================================================================
// Tests for BLAS Level 1 Operations
// ============================================================================

TEST(BlasTest, VectorAbsSum) {
  VectorXd x(5);
  x << 1.0, -2.0, 3.0, -4.0, 5.0;

  double sum = vectorAbsSum(x);
  EXPECT_DOUBLE_EQ(sum, 15.0);  // |1| + |-2| + |3| + |-4| + |5| = 15
}

TEST(BlasTest, VectorAbsSumZero) {
  VectorXd x = VectorXd::Zero(10);
  double sum = vectorAbsSum(x);
  EXPECT_DOUBLE_EQ(sum, 0.0);
}

TEST(BlasTest, VectorSaxpy) {
  VectorXd x(3);
  VectorXd y(3);
  x << 1.0, 2.0, 3.0;
  y << 4.0, 5.0, 6.0;

  vectorSaxpy(2.0, x, y);  // y = 2*x + y

  EXPECT_DOUBLE_EQ(y(0), 6.0);   // 2*1 + 4
  EXPECT_DOUBLE_EQ(y(1), 9.0);   // 2*2 + 5
  EXPECT_DOUBLE_EQ(y(2), 12.0);  // 2*3 + 6
}

TEST(BlasTest, VectorSaxpyZeroScalar) {
  VectorXd x(3);
  VectorXd y(3);
  x << 1.0, 2.0, 3.0;
  y << 4.0, 5.0, 6.0;

  VectorXd y_original = y;
  vectorSaxpy(0.0, x, y);

  // y should be unchanged
  EXPECT_TRUE(y.isApprox(y_original));
}

TEST(BlasTest, VectorDot) {
  VectorXd x(4);
  VectorXd y(4);
  x << 1.0, 2.0, 3.0, 4.0;
  y << 5.0, 6.0, 7.0, 8.0;

  double dot = vectorDot(x, y);
  EXPECT_DOUBLE_EQ(dot, 70.0);  // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
}

TEST(BlasTest, VectorDotOrthogonal) {
  VectorXd x(2);
  VectorXd y(2);
  x << 1.0, 0.0;
  y << 0.0, 1.0;

  double dot = vectorDot(x, y);
  EXPECT_DOUBLE_EQ(dot, 0.0);
}

TEST(BlasTest, VectorScale) {
  VectorXd x(3);
  x << 1.0, 2.0, 3.0;

  vectorScale(3.0, x);

  EXPECT_DOUBLE_EQ(x(0), 3.0);
  EXPECT_DOUBLE_EQ(x(1), 6.0);
  EXPECT_DOUBLE_EQ(x(2), 9.0);
}

TEST(BlasTest, VectorMaxAbsIndex) {
  VectorXd x(5);
  x << 1.0, -5.0, 3.0, 2.0, -4.0;

  int idx = vectorMaxAbsIndex(x);
  EXPECT_EQ(idx, 1);  // Index 1 has value -5, |5| is maximum
}

TEST(BlasTest, VectorMaxAbsIndexPositive) {
  VectorXd x(4);
  x << 1.0, 2.0, 8.0, 3.0;

  int idx = vectorMaxAbsIndex(x);
  EXPECT_EQ(idx, 2);  // Index 2 has value 8
}

// ============================================================================
// Tests for Dense Matrix Operations
// ============================================================================

TEST(DenseMatrixTest, SolveSimple) {
  // Solve: A*x = b where A = [[2, 1], [1, 3]], b = [5, 7]
  // From 2x + y = 5 and x + 3y = 7
  // Solution: x = 1.6, y = 1.8
  MatrixXd A(2, 2);
  A << 2.0, 1.0,
     1.0, 3.0;

  VectorXd b(2);
  b << 5.0, 7.0;

  VectorXd x = solveDense(A, b);

  EXPECT_NEAR(x(0), 1.6, 1e-10);
  EXPECT_NEAR(x(1), 1.8, 1e-10);

  // Verify: A*x = b
  VectorXd result = A * x;
  EXPECT_NEAR(result(0), 5.0, 1e-10);
  EXPECT_NEAR(result(1), 7.0, 1e-10);
}

TEST(DenseMatrixTest, SolveIdentity) {
  // Solve: I*x = b, solution should be x = b
  MatrixXd I = MatrixXd::Identity(3, 3);
  VectorXd b(3);
  b << 1.0, 2.0, 3.0;

  VectorXd x = solveDense(I, b);

  EXPECT_TRUE(x.isApprox(b, 1e-10));
}

TEST(DenseMatrixTest, SolveLargerSystem) {
  // 3x3 system
  MatrixXd A(3, 3);
  A << 4.0, 1.0, 0.0,
     1.0, 4.0, 1.0,
     0.0, 1.0, 4.0;

  VectorXd b(3);
  b << 1.0, 2.0, 3.0;

  VectorXd x = solveDense(A, b);

  // Verify solution: A*x should equal b
  VectorXd residual = A * x - b;
  EXPECT_LT(residual.norm(), 1e-10);
}

TEST(DenseMatrixTest, FactorizeAndSolve) {
  MatrixXd A(2, 2);
  A << 3.0, 1.0,
     1.0, 2.0;

  VectorXd b(2);
  b << 4.0, 3.0;

  // Factorize once
  auto lu = factorizeDense(A);

  // Solve multiple times with same factorization
  VectorXd x1 = solveDenseFactored(lu, b);

  VectorXd b2(2);
  b2 << 5.0, 4.0;
  VectorXd x2 = solveDenseFactored(lu, b2);

  // Verify first solution
  VectorXd residual1 = A * x1 - b;
  EXPECT_LT(residual1.norm(), 1e-10);

  // Verify second solution
  VectorXd residual2 = A * x2 - b2;
  EXPECT_LT(residual2.norm(), 1e-10);
}

TEST(DenseMatrixTest, ConditionNumberEstimate) {
  // Well-conditioned matrix
  MatrixXd A(2, 2);
  A << 2.0, 0.0,
     0.0, 2.0;

  VectorXd b(2);
  b << 1.0, 1.0;

  double rcond;
  VectorXd x = solveDense(A, b, &rcond);

  // Well-conditioned matrix should have rcond close to 1
  EXPECT_GT(rcond, 0.1);
}

TEST(DenseMatrixTest, MatrixNorm1) {
  MatrixXd A(3, 3);
  A << 1.0, -2.0, 3.0,
     4.0,  5.0, -6.0,
     -7.0, 8.0, 9.0;

  // 1-norm is max column sum
  // Col 0: |1| + |4| + |-7| = 12
  // Col 1: |-2| + |5| + |8| = 15
  // Col 2: |3| + |-6| + |9| = 18
  double norm = matrixNorm1(A);
  EXPECT_DOUBLE_EQ(norm, 18.0);
}

// ============================================================================
// Tests for Band Matrix Operations
// ============================================================================

TEST(BandMatrixTest, SolveTridiagonal) {
  // Tridiagonal matrix: kl=1, ku=1
  // [2  1  0  0]
  // [1  2  1  0]
  // [0  1  2  1]
  // [0  0  1  2]
  MatrixXd A(4, 4);
  A << 2.0, 1.0, 0.0, 0.0,
     1.0, 2.0, 1.0, 0.0,
     0.0, 1.0, 2.0, 1.0,
     0.0, 0.0, 1.0, 2.0;

  VectorXd b(4);
  b << 1.0, 2.0, 3.0, 4.0;

  VectorXd x = solveBand(A, b, 1, 1);

  // Verify solution
  VectorXd residual = A * x - b;
  EXPECT_LT(residual.norm(), 1e-10);
}

TEST(BandMatrixTest, SolveDiagonal) {
  // Diagonal matrix: kl=0, ku=0
  MatrixXd A = MatrixXd::Identity(5, 5) * 3.0;
  VectorXd b(5);
  b << 3.0, 6.0, 9.0, 12.0, 15.0;

  VectorXd x = solveBand(A, b, 0, 0);

  // Solution should be b/3
  VectorXd expected = b / 3.0;
  EXPECT_TRUE(x.isApprox(expected, 1e-10));
}

// ============================================================================
// Tests for Eigenvalue/Eigenvector Computation
// ============================================================================

TEST(EigenTest, SymmetricMatrix2x2) {
  // Symmetric matrix with known eigenvalues
  // [3  1]
  // [1  3]
  // Eigenvalues: 4, 2
  MatrixXd A(2, 2);
  A << 3.0, 1.0,
     1.0, 3.0;

  VectorXd eigenvalues;
  MatrixXd eigenvectors;

  computeEigensystem(A, eigenvalues, eigenvectors, true);

  // Check eigenvalues (sorted ascending)
  EXPECT_NEAR(eigenvalues(0), 2.0, 1e-10);
  EXPECT_NEAR(eigenvalues(1), 4.0, 1e-10);

  // Verify eigenvector equation: A*v = lambda*v
  for (int i = 0; i < 2; ++i) {
    VectorXd v = eigenvectors.col(i);
    VectorXd Av = A * v;
    VectorXd lambda_v = eigenvalues(i) * v;
    EXPECT_TRUE(Av.isApprox(lambda_v, 1e-10));
  }
}

TEST(EigenTest, DiagonalMatrix) {
  // Diagonal matrix - eigenvalues are diagonal elements
  MatrixXd A = MatrixXd::Zero(3, 3);
  A(0, 0) = 5.0;
  A(1, 1) = 2.0;
  A(2, 2) = 8.0;

  VectorXd eigenvalues;
  MatrixXd eigenvectors;

  computeEigensystem(A, eigenvalues, eigenvectors, true);

  // Eigenvalues should be 2, 5, 8 (sorted)
  EXPECT_NEAR(eigenvalues(0), 2.0, 1e-10);
  EXPECT_NEAR(eigenvalues(1), 5.0, 1e-10);
  EXPECT_NEAR(eigenvalues(2), 8.0, 1e-10);
}

TEST(EigenTest, IdentityMatrix) {
  MatrixXd I = MatrixXd::Identity(4, 4);

  VectorXd eigenvalues;
  MatrixXd eigenvectors;

  computeEigensystem(I, eigenvalues, eigenvectors, true);

  // All eigenvalues should be 1
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(eigenvalues(i), 1.0, 1e-10);
  }
}

TEST(EigenTest, AsymmetricMatrixRealEigenvalues) {
  // Upper triangular matrix with real eigenvalues
  MatrixXd A(3, 3);
  A << 1.0, 2.0, 3.0,
     0.0, 4.0, 5.0,
     0.0, 0.0, 6.0;

  VectorXd eigenvalues;
  MatrixXd eigenvectors;

  computeEigensystem(A, eigenvalues, eigenvectors, true);

  // Eigenvalues of triangular matrix are diagonal elements
  EXPECT_NEAR(eigenvalues(0), 1.0, 1e-10);
  EXPECT_NEAR(eigenvalues(1), 4.0, 1e-10);
  EXPECT_NEAR(eigenvalues(2), 6.0, 1e-10);
}

TEST(EigenTest, HasRealEigenvalues) {
  VectorXcd complex_eigs(3);
  complex_eigs << std::complex<double>(1.0, 0.0),
          std::complex<double>(2.0, 1e-12),
          std::complex<double>(3.0, 0.0);

  EXPECT_TRUE(hasRealEigenvalues(complex_eigs, 1e-10));
}

TEST(EigenTest, HasComplexEigenvalues) {
  VectorXcd complex_eigs(2);
  complex_eigs << std::complex<double>(1.0, 0.5),
          std::complex<double>(1.0, -0.5);

  EXPECT_FALSE(hasRealEigenvalues(complex_eigs, 1e-10));
}

TEST(EigenTest, ExtractRealEigenvalues) {
  VectorXcd complex_eigs(3);
  complex_eigs << std::complex<double>(1.5, 1e-15),
          std::complex<double>(2.5, 0.0),
          std::complex<double>(3.5, -1e-15);

  VectorXd real_eigs = extractRealEigenvalues(complex_eigs);

  EXPECT_DOUBLE_EQ(real_eigs(0), 1.5);
  EXPECT_DOUBLE_EQ(real_eigs(1), 2.5);
  EXPECT_DOUBLE_EQ(real_eigs(2), 3.5);
}

TEST(EigenTest, ExtractComplexEigenvaluesThrows) {
  VectorXcd complex_eigs(2);
  complex_eigs << std::complex<double>(1.0, 0.1),
          std::complex<double>(2.0, 0.0);

  EXPECT_THROW(extractRealEigenvalues(complex_eigs), std::runtime_error);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(IntegrationTest, SolveMultipleRHS) {
  // Test solving A*X = B where B has multiple columns
  MatrixXd A(3, 3);
  A << 4.0, 1.0, 0.0,
     1.0, 4.0, 1.0,
     0.0, 1.0, 4.0;

  // Factorize once
  auto lu = factorizeDense(A);

  // Solve for multiple right-hand sides
  VectorXd b1(3);
  b1 << 1.0, 0.0, 0.0;
  VectorXd x1 = solveDenseFactored(lu, b1);

  VectorXd b2(3);
  b2 << 0.0, 1.0, 0.0;
  VectorXd x2 = solveDenseFactored(lu, b2);

  VectorXd b3(3);
  b3 << 0.0, 0.0, 1.0;
  VectorXd x3 = solveDenseFactored(lu, b3);

  // Verify all solutions
  EXPECT_LT((A * x1 - b1).norm(), 1e-10);
  EXPECT_LT((A * x2 - b2).norm(), 1e-10);
  EXPECT_LT((A * x3 - b3).norm(), 1e-10);
}

TEST(IntegrationTest, EigenvalueAndSolve) {
  // Create a matrix, compute eigenvalues, then solve a system
  MatrixXd A(3, 3);
  A << 5.0, 1.0, 0.0,
     1.0, 5.0, 1.0,
     0.0, 1.0, 5.0;

  // Compute eigenvalues
  VectorXd eigenvalues;
  MatrixXd eigenvectors;
  computeEigensystem(A, eigenvalues, eigenvectors);

  // Eigenvalues should all be positive (A is positive definite)
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(eigenvalues(i), 0.0);
  }

  // Now solve A*x = b
  VectorXd b(3);
  b << 1.0, 2.0, 3.0;
  VectorXd x = solveDense(A, b);

  // Verify solution
  EXPECT_LT((A * x - b).norm(), 1e-10);
}
