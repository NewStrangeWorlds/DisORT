#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace disortpp {

/**
 * @brief Binary search to locate value in sorted array
 *
 * Given a sorted array xx and a value x, returns an index j such that
 * x is between xx[j] and xx[j+1]. Works with both increasing and
 * decreasing sorted arrays.
 *
 * This is the C++ version of locate() from Numerical Recipes.
 * Converted to use 0-based indexing and std::vector.
 *
 * @param xx Sorted array (monotonic, either increasing or decreasing)
 * @param x Value to locate
 * @return Index j where xx[j] <= x < xx[j+1] (for increasing arrays)
 *         Returns -1 if x < xx[0], returns n-1 if x >= xx[n-1]
 *
 * @note Original C version used 1-based indexing. This uses 0-based.
 */
template<typename T>
int locate(const std::vector<T>& xx, T x) {
  const int n = static_cast<int>(xx.size());

  if (n == 0) {
    return -1;
  }

  if (n == 1) {
    return 0;
  }

  // Determine if array is increasing or decreasing
  bool increasing = (xx[1] > xx[0]);

  // Binary search
  int jl = 0;        // Lower limit
  int ju = n;        // Upper limit

  while (ju - jl > 1) {
    int jm = (ju + jl) >> 1;  // Midpoint

    if ((x >= xx[jm] && increasing) || (x <= xx[jm] && !increasing)) {
      jl = jm;
    } else {
      ju = jm;
    }
  }

  // Handle edge cases
  if (x == xx[0]) {
    return 0;
  } else if (x == xx[n-1]) {
    return n - 2;
  } else {
    return jl;
  }
}

/**
 * @brief Safe floating-point comparison using Knuth's algorithm
 *
 * Implements Knuth's recommendations for safe floating-point comparison.
 * Uses relative epsilon based on the magnitude of the numbers being compared.
 *
 * Reference: Knuth, D. E. (1998). The Art of Computer Programming.
 *            Volume 2: Seminumerical Algorithms. Section 4.2.2, p. 233.
 *
 * @param x1 First number
 * @param x2 Second number
 * @return -1 if x1 < x2, 0 if x1 == x2, 1 if x1 > x2
 */
inline int fcmp(double x1, double x2) {
  int exponent;
  double delta, difference;
  constexpr double epsilon = std::numeric_limits<double>::epsilon();

  // Get exponent of max(|x1|, |x2|)
  std::frexp(std::abs(x1) > std::abs(x2) ? x1 : x2, &exponent);

  // Form delta neighborhood around x2
  // delta = epsilon * 2^exponent
  delta = std::ldexp(epsilon, exponent);
  difference = x1 - x2;

  if (difference > delta) {
    return 1;   // x1 > x2
  } else if (difference < -delta) {
    return -1;  // x1 < x2
  } else {
    return 0;   // x1 == x2
  }
}

/**
 * @brief Safe division with overflow and underflow protection
 *
 * Computes a/b with protection against overflow and underflow.
 * Special cases:
 *   - If b == 0, returns 1 + a
 *   - If a == 0, returns 0
 *   - If both very small, returns 1
 *   - If ratio would overflow, returns max double
 *   - If ratio would underflow, returns min double
 *
 * @param a Numerator
 * @param b Denominator
 * @return a/b with overflow/underflow protection
 */
inline double ratio(double a, double b) {
  constexpr double tiny = std::numeric_limits<double>::min();
  constexpr double huge = std::numeric_limits<double>::max();
  const double powmax = std::log10(huge);
  const double powmin = std::log10(tiny);

  if (fcmp(b, 0.0) == 0) {
    return 1.0 + a;
  }

  if (fcmp(a, 0.0) == 0) {
    return 0.0;
  }

  double absa = std::abs(a);
  double absb = std::abs(b);
  double powa = std::log10(absa);
  double powb = std::log10(absb);

  double ans;

  if (fcmp(absa, tiny) < 0 && fcmp(absb, tiny) < 0) {
    ans = 1.0;
  } else if (fcmp(powa - powb, powmax) >= 0) {
    ans = huge;
  } else if (fcmp(powa - powb, powmin) <= 0) {
    ans = tiny;
  } else {
    ans = absa / absb;
  }

  // Determine sign (avoid a*b which may overflow)
  if ((a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0)) {
    ans *= -1.0;
  }

  return ans;
}

/**
 * @brief Interpolation utility (to be implemented if needed)
 *
 * Placeholder for c_inter() function if needed in future phases.
 */
// double interpolate(const std::vector<float>& xarr, const std::vector<double>& yarr,
//                   double x, int npoints, int itype);

} // namespace disortpp
