#include "testing.hpp"
#include <Eigen/Dense>
#include <optional>

// Basic sanity tests for the build system

TEST(BuildTest, EigenWorks) {
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 1.0;
  m(0, 1) = 2.0;
  m(1, 0) = 3.0;
  m(1, 1) = 4.0;

  EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 4.0);

  double det = m.determinant();
  EXPECT_NEAR(det, -2.0, 1e-10);
}

TEST(BuildTest, StdVectorWorks) {
  std::vector<double> v = {1.0, 2.0, 3.0};
  EXPECT_EQ(v.size(), 3);
  EXPECT_DOUBLE_EQ(v[0], 1.0);
}

TEST(BuildTest, CppSeventeenFeatures) {
  // Test structured bindings (C++17)
  auto [a, b] = std::make_pair(1, 2);
  EXPECT_EQ(a, 1);
  EXPECT_EQ(b, 2);

  // Test std::optional (C++17)
  std::optional<int> opt = 42;
  ASSERT_TRUE(opt.has_value());
  EXPECT_EQ(opt.value(), 42);
}
