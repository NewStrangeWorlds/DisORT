#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>

namespace disortpp {
namespace detail {

// ============================================================================
// Direct C++ port of LINPACK sgbfa/sgbsl banded LU factorization and solve.
//
// Faithful translation of the original Fortran LINPACK routines (Moler, 1982)
// as used in CDISORT. No external BLAS dependency — all operations inlined.
//
// Storage:  Column-major, lda = 2*ml + mu + 1 rows per column.
//           Diagonal at 1-based row m = ml+mu+1, i.e. 0-based row ml+mu.
//           Extra ml rows at top for fill-in during pivoting.
//
// Pivot convention: ipvt[k] stores 1-based original matrix row index of pivot
// for column k+1 (matching LINPACK convention). Both bandFactor and bandSolve
// use this convention consistently.
// ============================================================================

/**
 * Factor banded matrix in-place (partial pivoting).
 * Direct port of LINPACK sgbfa with pointer-optimized inner loops.
 */
inline void bandFactor(double* abd, int lda, int n, int ml, int mu, int* ipvt) {
  // 0-based diagonal offset in each column
  const int m0 = ml + mu;  // ABD(m,k) in 1-based = abd[m0 + (k-1)*lda] in 0-based
  const int m1 = m0 + 1;   // m in 1-based LINPACK = ml + mu + 1

  // Zero initial fill-in columns (1-based j from mu+2 to min(n, m1)-1)
  {
    int j1 = std::min(n, m1) - 1;
    for (int j = mu + 2; j <= j1; ++j) {
      int i0_1based = m1 + 1 - j;   // first 1-based row to zero
      int nzero = ml - i0_1based + 1;
      if (nzero > 0) {
        std::memset(abd + (i0_1based - 1) + (j - 1) * lda, 0, nzero * sizeof(double));
      }
    }
  }

  int jz = std::min(n, m1) - 1;
  int ju = 0;

  for (int k = 1; k <= n - 1; ++k) {
    // Zero next fill-in column
    jz++;
    if (jz <= n && ml > 0) {
      std::memset(abd + (jz - 1) * lda, 0, ml * sizeof(double));
    }

    int lm = std::min(ml, n - k);

    // Column k pointer (0-based): abd + (k-1)*lda
    double* col_k = abd + (k - 1) * lda;

    // Find pivot in col_k[m0..m0+lm]
    int l_offset = 0;  // offset from m0
    {
      double smax = std::abs(col_k[m0]);
      for (int i = 1; i <= lm; ++i) {
        double xmag = std::abs(col_k[m0 + i]);
        if (xmag > smax) {
          smax = xmag;
          l_offset = i;
        }
      }
    }
    int l = m1 + l_offset;  // 1-based band row of pivot
    ipvt[k - 1] = l + k - m1;  // 1-based original matrix row

    if (col_k[m0 + l_offset] == 0.0) {
      continue;  // singular column
    }

    // Swap pivot with diagonal
    if (l_offset != 0) {
      double tmp = col_k[m0 + l_offset];
      col_k[m0 + l_offset] = col_k[m0];
      col_k[m0] = tmp;
    }

    // Compute multipliers: col_k[m0+1..m0+lm] *= -1/col_k[m0]
    {
      double t = -1.0 / col_k[m0];
      double* mults = col_k + m0 + 1;
      for (int i = 0; i < lm; ++i) {
        mults[i] *= t;
      }
    }

    // Row elimination with column indexing
    ju = std::min(std::max(ju, mu + ipvt[k - 1]), n);
    int ll = l;   // tracks pivot row in subsequent columns (1-based band row)
    int mm = m1;  // tracks diagonal row in subsequent columns (1-based band row)
    const double* src = col_k + m0 + 1;  // multipliers source: col_k[m0+1..m0+lm]

    for (int j = k + 1; j <= ju; ++j) {
      ll--;
      mm--;
      double* col_j = abd + (j - 1) * lda;
      int ll0 = ll - 1;  // 0-based
      int mm0 = mm - 1;  // 0-based

      double t = col_j[ll0];
      if (ll0 != mm0) {
        col_j[ll0] = col_j[mm0];
        col_j[mm0] = t;
      }

      if (t != 0.0) {
        // axpy: col_j[mm0+1..mm0+lm] += t * src[0..lm-1]
        double* dst = col_j + mm0 + 1;
        for (int i = 0; i < lm; ++i) {
          dst[i] += t * src[i];
        }
      }
    }
  }

  ipvt[n - 1] = n;  // 1-based
}

/**
 * Solve A*x = b using LU factors from bandFactor.
 * Direct port of LINPACK sgbsl (job=0 case only).
 */
inline void bandSolve(const double* abd, int lda, int n, int ml, int mu,
                      const int* ipvt, double* b) {
  const int m0 = mu + ml;  // 0-based diagonal row

  // Forward elimination: solve L*y = b
  if (ml > 0) {
    for (int k = 1; k <= n - 1; ++k) {
      int lm = std::min(ml, n - k);
      int l = ipvt[k - 1];  // 1-based matrix row
      double t = b[l - 1];
      if (l != k) {
        b[l - 1] = b[k - 1];
        b[k - 1] = t;
      }
      // axpy: b[k..k+lm-1] += t * multipliers
      const double* mults = abd + (k - 1) * lda + m0 + 1;
      double* bi = b + k;
      for (int i = 0; i < lm; ++i) {
        bi[i] += t * mults[i];
      }
    }
  }

  // Back substitution: solve U*x = y
  const int m1 = m0 + 1;  // 1-based diagonal row
  for (int kb = 1; kb <= n; ++kb) {
    int k = n + 1 - kb;
    const double* col_k = abd + (k - 1) * lda;
    b[k - 1] /= col_k[m0];
    int lm = std::min(k, m1) - 1;
    if (lm > 0) {
      double t = -b[k - 1];
      // axpy: b[k-lm-1..k-2] += t * col_k[m0-lm..m0-1]
      const double* src = col_k + m0 - lm;
      double* dst = b + k - 1 - lm;
      for (int i = 0; i < lm; ++i) {
        dst[i] += t * src[i];
      }
    }
  }
}

} // namespace detail
} // namespace disortpp
