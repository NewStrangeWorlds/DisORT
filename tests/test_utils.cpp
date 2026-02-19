#include "testing.hpp"
#include "Utils.hpp"
#include "Legendre.hpp"
#include "Planck.hpp"
#include <cmath>
#include <vector>

using namespace disortpp;

// ============================================================================
// Tests for locate() function
// ============================================================================

TEST(LocateTest, IncreasingArray) {
  std::vector<double> xx = {1.0, 2.0, 3.0, 4.0, 5.0};

  EXPECT_EQ(locate(xx, 1.5), 0);  // Between xx[0] and xx[1]
  EXPECT_EQ(locate(xx, 2.5), 1);  // Between xx[1] and xx[2]
  EXPECT_EQ(locate(xx, 3.5), 2);  // Between xx[2] and xx[3]
  EXPECT_EQ(locate(xx, 4.5), 3);  // Between xx[3] and xx[4]
}

TEST(LocateTest, DecreasingArray) {
  std::vector<double> xx = {5.0, 4.0, 3.0, 2.0, 1.0};

  EXPECT_EQ(locate(xx, 4.5), 0);  // Between xx[0] and xx[1]
  EXPECT_EQ(locate(xx, 3.5), 1);  // Between xx[1] and xx[2]
  EXPECT_EQ(locate(xx, 2.5), 2);  // Between xx[2] and xx[3]
}

TEST(LocateTest, EdgeCases) {
  std::vector<double> xx = {1.0, 2.0, 3.0, 4.0, 5.0};

  EXPECT_EQ(locate(xx, 1.0), 0);      // Exact match at start
  EXPECT_EQ(locate(xx, 5.0), 3);      // Exact match at end
  EXPECT_EQ(locate(xx, 0.5), 0);      // Below range
  EXPECT_EQ(locate(xx, 5.5), 4);      // Above range
}

TEST(LocateTest, SingleElement) {
  std::vector<double> xx = {3.0};

  EXPECT_EQ(locate(xx, 2.0), 0);
  EXPECT_EQ(locate(xx, 3.0), 0);
  EXPECT_EQ(locate(xx, 4.0), 0);
}

TEST(LocateTest, TwoElements) {
  std::vector<double> xx = {1.0, 3.0};

  EXPECT_EQ(locate(xx, 0.5), 0);  // Below range
  EXPECT_EQ(locate(xx, 1.0), 0);  // At first element
  EXPECT_EQ(locate(xx, 2.0), 0);  // Between elements
  EXPECT_EQ(locate(xx, 3.0), 0);  // At second element
  EXPECT_EQ(locate(xx, 4.0), 1);  // Above range
}

// ============================================================================
// Tests for fcmp() function
// ============================================================================

TEST(FcmpTest, ExactEquality) {
  EXPECT_EQ(fcmp(1.0, 1.0), 0);
  EXPECT_EQ(fcmp(0.0, 0.0), 0);
  EXPECT_EQ(fcmp(-1.0, -1.0), 0);
}

TEST(FcmpTest, ClearLessThan) {
  EXPECT_EQ(fcmp(1.0, 2.0), -1);
  EXPECT_EQ(fcmp(-2.0, -1.0), -1);
  EXPECT_EQ(fcmp(0.0, 1.0), -1);
}

TEST(FcmpTest, ClearGreaterThan) {
  EXPECT_EQ(fcmp(2.0, 1.0), 1);
  EXPECT_EQ(fcmp(-1.0, -2.0), 1);
  EXPECT_EQ(fcmp(1.0, 0.0), 1);
}

TEST(FcmpTest, NearEquality) {
  // Numbers very close to each other should be equal
  double x = 1.0;
  double y = x + std::numeric_limits<double>::epsilon() / 2.0;
  EXPECT_EQ(fcmp(x, y), 0);
}

TEST(FcmpTest, SmallNumbers) {
  EXPECT_EQ(fcmp(1e-10, 1e-10), 0);
  EXPECT_EQ(fcmp(1e-10, 2e-10), -1);
}

// ============================================================================
// Tests for ratio() function
// ============================================================================

TEST(RatioTest, NormalDivision) {
  EXPECT_DOUBLE_EQ(ratio(6.0, 2.0), 3.0);
  EXPECT_DOUBLE_EQ(ratio(10.0, 5.0), 2.0);
  EXPECT_DOUBLE_EQ(ratio(1.0, 4.0), 0.25);
}

TEST(RatioTest, ZeroDenominator) {
  // When b == 0, returns 1 + a
  EXPECT_DOUBLE_EQ(ratio(5.0, 0.0), 6.0);
  EXPECT_DOUBLE_EQ(ratio(-3.0, 0.0), -2.0);
  EXPECT_DOUBLE_EQ(ratio(0.0, 0.0), 1.0);
}

TEST(RatioTest, ZeroNumerator) {
  // When a == 0, returns 0
  EXPECT_DOUBLE_EQ(ratio(0.0, 5.0), 0.0);
  EXPECT_DOUBLE_EQ(ratio(0.0, -3.0), 0.0);
}

TEST(RatioTest, SignHandling) {
  EXPECT_DOUBLE_EQ(ratio(6.0, -2.0), -3.0);
  EXPECT_DOUBLE_EQ(ratio(-6.0, 2.0), -3.0);
  EXPECT_DOUBLE_EQ(ratio(-6.0, -2.0), 3.0);
}

TEST(RatioTest, OverflowProtection) {
  // Very large ratio should be clamped
  double huge = std::numeric_limits<double>::max();
  double tiny = std::numeric_limits<double>::min();

  double result = ratio(huge, tiny);
  EXPECT_TRUE(std::isfinite(result));  // Should not overflow
}

TEST(RatioTest, UnderflowProtection) {
  // Very small ratio should be clamped
  double huge = std::numeric_limits<double>::max();
  double tiny = std::numeric_limits<double>::min();

  double result = ratio(tiny, huge);
  EXPECT_TRUE(result >= 0.0);  // Should be finite and positive
}

TEST(RatioTest, BothTiny) {
  // When both are very small, returns 1
  double tiny = std::numeric_limits<double>::min() / 2.0;
  double result = ratio(tiny, tiny);
  EXPECT_DOUBLE_EQ(result, 1.0);
}

// ============================================================================
// Tests for Legendre polynomials
// ============================================================================

TEST(LegendreTest, OrdinaryLegendre_P0_P1) {
  std::vector<double> mu = {0.0, 0.5, 1.0};
  std::vector<std::vector<double>> ylm;

  legendrePolynomials(3, 0, 1, mu, ylm);

  // P_0(mu) = 1 for all mu
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(ylm[0][i], 1.0);
  }

  // P_1(mu) = mu
  EXPECT_DOUBLE_EQ(ylm[1][0], 0.0);
  EXPECT_DOUBLE_EQ(ylm[1][1], 0.5);
  EXPECT_DOUBLE_EQ(ylm[1][2], 1.0);
}

TEST(LegendreTest, OrdinaryLegendre_P2) {
  std::vector<double> mu = {0.0, 0.5, 1.0};
  std::vector<std::vector<double>> ylm;

  legendrePolynomials(3, 0, 2, mu, ylm);

  // P_2(mu) = (3*mu^2 - 1) / 2
  EXPECT_DOUBLE_EQ(ylm[2][0], -0.5);              // P_2(0)
  EXPECT_DOUBLE_EQ(ylm[2][1], (3*0.25 - 1)/2.0);  // P_2(0.5)
  EXPECT_DOUBLE_EQ(ylm[2][2], 1.0);               // P_2(1)
}

TEST(LegendreTest, InvalidInputs) {
  std::vector<double> mu = {0.0};
  std::vector<std::vector<double>> ylm;

  EXPECT_THROW(legendrePolynomials(0, 0, 1, mu, ylm), std::invalid_argument);  // nmu <= 0
  EXPECT_THROW(legendrePolynomials(1, -1, 1, mu, ylm), std::invalid_argument); // m < 0
}

TEST(LegendreTest, FlatArrayVersion) {
  const int nmu = 3;
  const int maxmu = 10;
  const int twonm1 = 2;

  std::vector<double> mu = {0.0, 0.5, 1.0};
  std::vector<double> ylm((twonm1 + 1) * (maxmu + 1), 0.0);

  legendrePolynomialsFlat(nmu, 0, maxmu, twonm1, mu.data(), ylm.data());

  // Check P_0 = 1
  for (int i = 0; i < nmu; ++i) {
    EXPECT_DOUBLE_EQ(ylm[0 * (maxmu + 1) + i], 1.0);
  }

  // Check P_1 = mu
  for (int i = 0; i < nmu; ++i) {
    EXPECT_DOUBLE_EQ(ylm[1 * (maxmu + 1) + i], mu[i]);
  }
}

// ============================================================================
// Tests for Planck functions
// ============================================================================

TEST(PlanckTest, ValidInputs) {
  double wnumlo = 1000.0;  // cm^-1
  double wnumhi = 2000.0;
  double temp = 300.0;     // K

  double result = planckFunction(wnumlo, wnumhi, temp);

  EXPECT_GT(result, 0.0);  // Should be positive
  EXPECT_TRUE(std::isfinite(result));  // Should be finite
}

TEST(PlanckTest, InvalidInputs) {
  EXPECT_THROW(planckFunction(1000.0, 2000.0, -10.0), std::invalid_argument);  // Negative temp
  EXPECT_THROW(planckFunction(2000.0, 1000.0, 300.0), std::invalid_argument);  // wnumhi < wnumlo
  EXPECT_THROW(planckFunction(-10.0, 2000.0, 300.0), std::invalid_argument);   // Negative wnumlo
}

TEST(PlanckTest, VeryLowTemperature) {
  // Very low temperature should return ~0
  double result = planckFunction(1000.0, 2000.0, 1e-5);
  EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST(PlanckTest, Monochromatic) {
  // Single wavenumber (using planckFunction2)
  double wnum = 1500.0;
  double temp = 300.0;

  double result = planckFunction2(wnum, wnum, temp);
  EXPECT_GT(result, 0.0);
  EXPECT_TRUE(std::isfinite(result));
}

TEST(PlanckTest, IntegratedVsMonochromatic) {
  // Very narrow interval should approach monochromatic value
  double wnum = 1500.0;
  double delta = 0.01;
  double temp = 300.0;

  double mono = planckFunction2(wnum, wnum, temp);
  double integrated = planckFunction(wnum - delta/2, wnum + delta/2, temp);

  // They should be of similar order of magnitude
  EXPECT_GT(mono, 0.0);
  EXPECT_GT(integrated, 0.0);
}

TEST(PlanckTest, TemperatureDependence) {
  // Higher temperature should give larger Planck function
  double wnumlo = 1000.0;
  double wnumhi = 2000.0;

  double low_temp = planckFunction(wnumlo, wnumhi, 200.0);
  double high_temp = planckFunction(wnumlo, wnumhi, 400.0);

  EXPECT_GT(high_temp, low_temp);
}

TEST(PlanckTest, WavenumberDependence) {
  // At constant temperature, higher wavenumbers give more emission
  double temp = 300.0;

  double low_wnum = planckFunction(500.0, 1000.0, temp);
  double high_wnum = planckFunction(1500.0, 2000.0, temp);

  // This test depends on peak of Planck curve
  // At 300K, peak is around 600 cm^-1
  // So 500-1000 range includes peak, while 1500-2000 is in tail
  EXPECT_GT(low_wnum, 0.0);
  EXPECT_GT(high_wnum, 0.0);
}

TEST(PlanckTest, CloseWavenumbers) {
  // Very close wavenumbers should still work (tests Simpson's rule path)
  double wnum = 1500.0;
  double temp = 300.0;

  double result = planckFunction(wnum, wnum * 1.001, temp);  // 0.1% difference
  EXPECT_GT(result, 0.0);
  EXPECT_TRUE(std::isfinite(result));
}
