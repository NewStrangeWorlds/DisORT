#include "Planck.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <limits>

namespace disortpp {

using namespace PlanckConstants;

namespace {
  // Helper function: Planck function kernel
  inline double planckKernel(double v) {
    return v * v * v / (std::exp(v) - 1.0);
  }

  // Static variables for initialization
  struct PlanckData {
    double vmax;
    double sigdpi;
    double conc;
    double c1;  // For planckFunction2

    PlanckData() {
      vmax = std::log(std::numeric_limits<double>::max());
      sigdpi = SIGMA / M_PI;
      conc = 15.0 / std::pow(M_PI, 4.0);
      c1 = 1.1911e-8;
    }
  };

  static const PlanckData planck_data;
}

double planckFunction(double wnumlo, double wnumhi, double temp) 
{
  // Validate inputs
  if (temp < 0.0 || wnumhi <= wnumlo || wnumlo < 0.0) {
    throw std::invalid_argument(
      "planckFunction: Invalid arguments (temp=" + std::to_string(temp) +
      ", wnumlo=" + std::to_string(wnumlo) +
      ", wnumhi=" + std::to_string(wnumhi) + ")");
  }

  if (temp < 1.e-4) {
    return 0.0;
  }

  double v[2];
  v[0] = C2 * wnumlo / temp;
  v[1] = C2 * wnumhi / temp;

  // Check if wavenumbers are very close - use Simpson's rule
  if (v[0] > std::numeric_limits<double>::epsilon() &&
    v[1] < planck_data.vmax &&
    (wnumhi - wnumlo) / wnumhi < 1.e-2) {

    double hh = v[1] - v[0];
    double oldval = 0.0;
    double val0 = planckKernel(v[0]) + planckKernel(v[1]);
    bool converged = false;

    for (int n = 1; n <= 10; ++n) {
      double del = hh / (2.0 * n);
      double val = val0;

      for (int k = 1; k <= 2 * n - 1; ++k) {
        val += 2.0 * (1 + k % 2) * planckKernel(v[0] + k * del);
      }

      val *= del * A1;

      if (std::abs((val - oldval) / val) <= 1.e-6) {
        converged = true;
        return planck_data.sigdpi * std::pow(temp, 4.0) * planck_data.conc * val;
      }

      oldval = val;
    }

    if (!converged) {
      // Warning: Simpson's rule didn't converge, but return best estimate
      return planck_data.sigdpi * std::pow(temp, 4.0) * planck_data.conc * oldval;
    }
  }

  // General case: use power series or exponential series
  double p[2] = {0.0, 0.0};
  double d[2] = {0.0, 0.0};
  int smallv = 0;

  // Exponential series cutoff points
  constexpr double vcp[7] = {10.25, 5.7, 3.9, 2.9, 2.3, 1.9, 0.0};

  for (int i = 0; i < 2; ++i) {
    if (v[i] < VCUT) {
      // Use power series
      smallv++;
      double vsq = v[i] * v[i];
      p[i] = planck_data.conc * vsq * v[i] *
           (A1 + v[i] * (A2 + v[i] * (A3 + vsq * (A4 + vsq * (A5 + vsq * A6)))));
    } else {
      // Use exponential series
      // Find upper limit
      int mmax = 1;
      while (v[i] < vcp[mmax - 1]) {
        mmax++;
      }

      double ex = std::exp(-v[i]);
      double exm = 1.0;
      d[i] = 0.0;

      for (int m = 1; m <= mmax; ++m) {
        double mv = m * v[i];
        exm *= ex;
        d[i] += exm * (6.0 + mv * (6.0 + mv * (3.0 + mv))) / (m * m * m * m);
      }
      d[i] *= planck_data.conc;
    }
  }

  // Handle ill-conditioning
  double ans;
  if (smallv == 2) {
    // Both wnumlo and wnumhi small
    ans = p[1] - p[0];
  } else if (smallv == 1) {
    // wnumlo small, wnumhi large
    ans = 1.0 - p[0] - d[1];
  } else {
    // Both wnumlo and wnumhi large
    ans = d[0] - d[1];
  }

  ans *= planck_data.sigdpi * std::pow(temp, 4.0);

  if (ans == 0.0) {
    // Warning: underflow possible
    return 0.0;
  }

  return ans;
}

double planckFunction2(double wnumlo, double wnumhi, double temp) 
{
  // Validate inputs
  if (temp < 0.0 || wnumhi < wnumlo || wnumlo < 0.0) {
    throw std::invalid_argument(
      "planckFunction2: Invalid arguments (temp=" + std::to_string(temp) +
      ", wnumlo=" + std::to_string(wnumlo) +
      ", wnumhi=" + std::to_string(wnumhi) + ")");
  }

  if (temp < 1.e-4) {
    return 0.0;
  }

  // Special case: single wavenumber
  if (wnumhi == wnumlo) {
    double wvn = wnumhi;
    double arg = std::exp(-C2 * wvn / temp);
    return planck_data.c1 * wvn * wvn * wvn * arg / (1.0 - arg);
  }

  // Otherwise, same as planckFunction()
  return planckFunction(wnumlo, wnumhi, temp);
}

double planckFunctionDeriv2(double wnumlo, double wnumhi, double temp)
{
  // Validate inputs (mirrors planckFunction2)
  if (temp < 0.0 || wnumhi < wnumlo || wnumlo < 0.0) {
    throw std::invalid_argument(
      "planckFunctionDeriv2: Invalid arguments (temp=" + std::to_string(temp) +
      ", wnumlo=" + std::to_string(wnumlo) +
      ", wnumhi=" + std::to_string(wnumhi) + ")");
  }

  if (temp < 1.e-4) {
    return 0.0;
  }

  // Stable Planck shape function f(v) = v^3 / (e^v - 1), valid for all v >= 0.
  auto fkernel = [](double v) -> double {
    if (v < 1.e-6) return v * v;            // limit v^3/(e^v - 1) -> v^2 as v->0
    const double em = std::exp(-v);
    return v * v * v * em / (1.0 - em);     // = v^3/(e^v - 1), overflow-safe
  };

  // Single-wavenumber (spectral) case: b = c1 wvn^3 / (e^x - 1), x = C2 wvn/T.
  // db/dT = b * (x/T) / (1 - e^{-x}).
  if (wnumhi == wnumlo) {
    const double wvn = wnumhi;
    const double x = C2 * wvn / temp;
    const double em = std::exp(-x);
    const double b = planck_data.c1 * wvn * wvn * wvn * em / (1.0 - em);
    return b * (x / temp) / (1.0 - em);
  }

  // Band-integrated case. With B = (sigma/pi)(15/pi^4) T^4 I, I = integral_{v0}^{v1} f,
  //   dB/dT = 4 B / T  -  (sigma/pi)(15/pi^4) T^3 (v1 f(v1) - v0 f(v0)).
  const double sigdpi = SIGMA / M_PI;
  const double conc = 15.0 / std::pow(M_PI, 4.0);
  const double v0 = C2 * wnumlo / temp;
  const double v1 = C2 * wnumhi / temp;

  const double B = planckFunction2(wnumlo, wnumhi, temp);
  const double boundary = sigdpi * conc * temp * temp * temp
                          * (v1 * fkernel(v1) - v0 * fkernel(v0));

  return 4.0 * B / temp - boundary;
}

} // namespace disortpp
