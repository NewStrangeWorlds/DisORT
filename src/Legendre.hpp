#pragma once

#include <vector>
#include <cmath>

namespace disortpp {

/**
 * @brief Compute normalized associated Legendre polynomials
 *
 * Computes the normalized associated Legendre polynomial P_l^m(mu)
 * for all requested values of mu and polynomial degrees.
 *
 * Uses upward recurrence relations from:
 * - Dave/Armstrong equations (11-14) for m > 0
 * - Standard recurrence for m = 0
 * - STWL(58a,58c,58f) formulas
 *
 * @param nmu Number of angle cosines
 * @param m Azimuthal index (0 for ordinary Legendre polynomials)
 * @param twonm1 Highest polynomial degree to compute (typically 2*num_streams-1)
 * @param mu Angle cosines [nmu] (values in [-1, 1])
 * @param ylm Output: Legendre polynomials [twonm1+1][nmu]
 *           ylm[l][i] = P_l^m(mu[i])
 *
 * @note This function uses 0-based indexing for C++ vectors
 *       Original C version used 1-based indexing with macros
 */
void legendrePolynomials(int nmu, int m, int twonm1,
            const std::vector<double>& mu,
            std::vector<std::vector<double>>& ylm);

/**
 * @brief Compute normalized associated Legendre polynomials (flat array version)
 *
 * Same as legendrePolynomials() but uses a flat array for output.
 * This matches the original C interface more closely.
 *
 * @param nmu Number of angle cosines
 * @param m Azimuthal index
 * @param maxmu Maximum mu dimension (for array sizing)
 * @param twonm1 Highest polynomial degree
 * @param mu Input angle cosines [nmu] (0-based)
 * @param ylm Output flat array [(twonm1+1) * nmu] (0-based)
 *            Access as ylm[l * nmu + i] for P_l^m(mu[i])
 */
void legendrePolynomialsFlat(int nmu, int m, int maxmu, int twonm1,
               const double* mu, double* ylm);

} // namespace disortpp
