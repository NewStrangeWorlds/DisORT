#include "Legendre.hpp"
#include <cmath>
#include <stdexcept>

namespace disortpp {

void legendrePolynomials(int nmu, int m, int twonm1,
            const std::vector<double>& mu,
            std::vector<std::vector<double>>& ylm) 
{
  // Validate inputs
  if (nmu <= 0) {
    throw std::invalid_argument("nmu must be positive");
  }

  if (m < 0) {
    throw std::invalid_argument("m must be non-negative");
  }

  if (static_cast<int>(mu.size()) < nmu) {
    throw std::invalid_argument("mu array too small");
  }

  // Resize output array
  ylm.resize(twonm1 + 1);
  for (int l = 0; l <= twonm1; ++l) {
    ylm[l].resize(nmu);
  }

  if (m == 0) {
    // Upward recurrence for ordinary Legendre polynomials
    for (int i = 0; i < nmu; ++i) {
      ylm[0][i] = 1.0;
      ylm[1][i] = mu[i];
    }

    for (int l = 2; l <= twonm1; ++l) {
      for (int i = 0; i < nmu; ++i) {
        ylm[l][i] = ((2.0 * l - 1.0) * mu[i] * ylm[l-1][i]
              - (l - 1.0) * ylm[l-2][i]) / l;
      }
    }
  } else {
    // Associated Legendre polynomials (m > 0)
    for (int i = 0; i < nmu; ++i) {
      // Y-sub-m-super-m: Dave/Armstrong eqs. (11,12), STWL(58c)
      ylm[m][i] = -std::sqrt((1.0 - 1.0 / (2.0 * m)) * (1.0 - mu[i] * mu[i]))
            * ylm[m-1][i];

      // Y-sub-(m+1)-super-m: Dave/Armstrong eqs.(13,14), STWL(58f)
      ylm[m+1][i] = std::sqrt(2.0 * m + 1.0) * mu[i] * ylm[m][i];
    }

    // Upward recurrence: Dave/Armstrong eq.(10), STWL(58a)
    for (int l = m + 2; l <= twonm1; ++l) {
      double tmp1 = std::sqrt(static_cast<double>((l - m) * (l + m)));
      double tmp2 = std::sqrt(static_cast<double>((l - m - 1) * (l + m - 1)));

      for (int i = 0; i < nmu; ++i) {
        ylm[l][i] = ((2.0 * l - 1.0) * mu[i] * ylm[l-1][i]
              - tmp2 * ylm[l-2][i]) / tmp1;
      }
    }
  }
}

void legendrePolynomialsFlat(int nmu, int m, int maxmu, int twonm1,
               const double* mu, double* ylm) 
{
  // This version maintains compatibility with original C interface
  // Uses flat array with manual 2D indexing

  if (m == 0) {
    // Ordinary Legendre polynomials
    for (int i = 0; i < nmu; ++i) {
      ylm[0 * (maxmu + 1) + i] = 1.0;              // P_0
      ylm[1 * (maxmu + 1) + i] = mu[i];            // P_1
    }

    for (int l = 2; l <= twonm1; ++l) {
      for (int i = 0; i < nmu; ++i) {
        int idx_l   = l * (maxmu + 1) + i;
        int idx_l1  = (l - 1) * (maxmu + 1) + i;
        int idx_l2  = (l - 2) * (maxmu + 1) + i;

        ylm[idx_l] = ((2.0 * l - 1.0) * mu[i] * ylm[idx_l1]
               - (l - 1.0) * ylm[idx_l2]) / l;
      }
    }
  } else {
    // Associated Legendre polynomials
    for (int i = 0; i < nmu; ++i) {
      int idx_m   = m * (maxmu + 1) + i;
      int idx_m1  = (m - 1) * (maxmu + 1) + i;
      int idx_mp1 = (m + 1) * (maxmu + 1) + i;

      // P_m^m
      ylm[idx_m] = -std::sqrt((1.0 - 1.0 / (2.0 * m)) * (1.0 - mu[i] * mu[i]))
             * ylm[idx_m1];

      // P_{m+1}^m
      ylm[idx_mp1] = std::sqrt(2.0 * m + 1.0) * mu[i] * ylm[idx_m];
    }

    // Upward recurrence
    for (int l = m + 2; l <= twonm1; ++l) {
      double tmp1 = std::sqrt(static_cast<double>((l - m) * (l + m)));
      double tmp2 = std::sqrt(static_cast<double>((l - m - 1) * (l + m - 1)));

      for (int i = 0; i < nmu; ++i) {
        int idx_l  = l * (maxmu + 1) + i;
        int idx_l1 = (l - 1) * (maxmu + 1) + i;
        int idx_l2 = (l - 2) * (maxmu + 1) + i;

        ylm[idx_l] = ((2.0 * l - 1.0) * mu[i] * ylm[idx_l1]
               - tmp2 * ylm[idx_l2]) / tmp1;
      }
    }
  }
}

} // namespace disortpp
