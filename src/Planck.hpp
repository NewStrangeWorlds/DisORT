#pragma once

#include <cmath>

namespace disortpp {

/**
 * @brief Physical constants for Planck function calculations
 */
namespace PlanckConstants {
  constexpr double C2 = 1.438786;           // h*c/k in cm*K
  constexpr double SIGMA = 5.67032E-8;      // Stefan-Boltzmann constant W/m^2/K^4
  constexpr double VCUT = 1.5;              // Power series cutoff point

  // Power series coefficients
  constexpr double A1 = 1.0 / 3.0;
  constexpr double A2 = -1.0 / 8.0;
  constexpr double A3 = 1.0 / 60.0;
  constexpr double A4 = -1.0 / 5040.0;
  constexpr double A5 = 1.0 / 272160.0;
  constexpr double A6 = -1.0 / 13305600.0;
}

/**
 * @brief Compute Planck function integrated between two wavenumbers
 *
 * Computes the integral of the Planck function over a spectral interval:
 *   Integral(wnumlo to wnumhi) of 2hc^2 * nu^3 / (exp(hc*nu/kT) - 1) d(nu)
 *
 * where h = Planck's constant, c = speed of light, k = Boltzmann constant,
 * nu = wavenumber, T = temperature.
 *
 * **Method:**
 * 1. For close wavenumbers: Simpson's rule quadrature
 * 2. For small wavenumbers: Power series expansion
 * 3. For large wavenumbers: Exponential series expansion
 *
 * **Accuracy:** At least 6 significant digits
 *
 * **Reference:** Specifications of the Physical World: New Value of the
 *                Fundamental Constants, Dimensions/N.B.S., Jan. 1974
 *
 * @param wnumlo Lower wavenumber [cm^-1]
 * @param wnumhi Upper wavenumber [cm^-1]
 * @param temp Temperature [K]
 * @return Integrated Planck function [W/m^2]
 * @throws std::invalid_argument if temp < 0, wnumhi <= wnumlo, or wnumlo < 0
 */
double planckFunction(double wnumlo, double wnumhi, double temp);

/**
 * @brief Compute Planck function for single wavenumber or integrated
 *
 * Alternative Planck function implementation (c_planck_func2 in original code).
 * Can compute either:
 * - Single wavenumber Planck function
 * - Integrated Planck function over an interval
 *
 * @param wnumlo Lower wavenumber [cm^-1] (or single wavenumber if wnumhi == wnumlo)
 * @param wnumhi Upper wavenumber [cm^-1]
 * @param temp Temperature [K]
 * @return Planck function value [W/m^2]
 */
double planckFunction2(double wnumlo, double wnumhi, double temp);

/**
 * @brief Temperature derivative of the integrated Planck function
 *
 * Computes dB/dT where B is the band-integrated Planck function returned by
 * planckFunction2(wnumlo, wnumhi, temp). Used for analytic temperature
 * Jacobians of the radiation field.
 *
 * For a band [wnumlo, wnumhi] (wnumhi > wnumlo) the result uses the exact
 * closed form
 *   dB/dT = 4 B / T  -  (sigma/pi) * (15/pi^4) * T^3 * (v1 f(v1) - v0 f(v0)),
 * where v_i = C2 * wnum_i / T and f(v) = v^3 / (e^v - 1). This is exact
 * regardless of how B itself is evaluated internally.
 *
 * For the single-wavenumber case (wnumhi == wnumlo) the spectral derivative
 *   dB/dT = b * (x / T) / (1 - e^{-x}),  x = C2 * wnum / T,
 * is returned, with b the spectral Planck value.
 *
 * @param wnumlo Lower wavenumber [cm^-1]
 * @param wnumhi Upper wavenumber [cm^-1]
 * @param temp Temperature [K]
 * @return dB/dT [W/m^2/K]
 * @throws std::invalid_argument if temp < 0, wnumhi < wnumlo, or wnumlo < 0
 */
double planckFunctionDeriv2(double wnumlo, double wnumhi, double temp);

} // namespace disortpp
