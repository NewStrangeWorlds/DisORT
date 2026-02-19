/**
 * @file testing.hpp
 * @brief Minimal self-contained test framework — no external dependencies.
 *
 * Drop-in replacement for <gtest/gtest.h>.  Supported macros:
 *   TEST(Suite, Name)            — define a test case
 *   EXPECT_TRUE/FALSE(expr)      — non-fatal boolean checks
 *   EXPECT_EQ/NE(a, b)           — non-fatal equality / inequality
 *   EXPECT_LT/LE/GT/GE(a, b)    — non-fatal relational checks
 *   EXPECT_NEAR(a, b, tol)       — non-fatal |a-b| <= tol
 *   EXPECT_DOUBLE_EQ(a, b)       — non-fatal, tolerance ~4 * DBL_EPSILON
 *   EXPECT_THROW(expr, Type)     — non-fatal: must throw Type
 *   EXPECT_NO_THROW(expr)        — non-fatal: must not throw
 *   ASSERT_TRUE/FALSE(expr)      — fatal boolean (aborts current test)
 *   ASSERT_EQ/NE(a, b)           — fatal equality / inequality
 *   ASSERT_NEAR(a, b, tol)       — fatal near check
 *   ASSERT_THROW(expr, Type)     — fatal: must throw Type
 *
 * Usage:
 *   - Include this header in every test .cpp file (replaces gtest/gtest.h).
 *   - Provide exactly one main.cpp that includes this header and calls:
 *       int main() { return testing::run_all(); }
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace testing {

/// Thrown by ASSERT_* macros to abort the current test immediately.
struct TestFailure : std::exception {};

struct TestCase {
  std::string suite;
  std::string name;
  std::function<void()> fn;
};

/// Singleton registry of all registered test cases.
inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> reg;
  return reg;
}

/// Per-test failure flag (reset to false before each test run).
inline bool& current_test_failed() {
  static bool flag = false;
  return flag;
}

/// Record a test failure with source location and message.
inline void record_failure(const char* file, int line, const std::string& msg) {
  std::cerr << file << ":" << line << ":  " << msg << "\n";
  current_test_failed() = true;
}

/// Sign-safe equality helper: promotes both args to common type, avoids -Wsign-compare.
template<typename A, typename B>
inline bool equal_cmp(const A& a, const B& b) {
  return static_cast<std::common_type_t<A, B>>(a) ==
         static_cast<std::common_type_t<A, B>>(b);
}

/// GTest compatibility: Test::HasFailure() reports whether the current test has failed.
struct Test {
  static bool HasFailure() { return current_test_failed(); }
};

/// Used by TEST() macro to register a test at static-init time.
struct Registrar {
  Registrar(const char* suite, const char* name, std::function<void()> fn) {
    registry().push_back({suite, name, fn});
  }
};

/// Run all registered tests; return number of failures (0 = success).
inline int run_all() {
  int passed = 0, failed = 0;
  for (auto& tc : registry()) {
    current_test_failed() = false;
    try {
      tc.fn();
    } catch (const TestFailure&) {
      // ASSERT_* already printed the failure message; test is aborted
    } catch (const std::exception& e) {
      std::cerr << "EXCEPTION in " << tc.suite << "." << tc.name
                << ": " << e.what() << "\n";
      current_test_failed() = true;
    } catch (...) {
      std::cerr << "UNKNOWN EXCEPTION in " << tc.suite << "." << tc.name << "\n";
      current_test_failed() = true;
    }
    if (current_test_failed()) {
      std::cerr << "[ FAILED ] " << tc.suite << "." << tc.name << "\n";
      ++failed;
    } else {
      ++passed;
    }
  }
  const int total = passed + failed;
  std::cout << "\n" << passed << "/" << total << " tests passed";
  if (failed > 0) std::cout << ", " << failed << " FAILED";
  std::cout << "\n";
  return failed;
}

} // namespace testing

// ---------------------------------------------------------------------------
// Miscellaneous GTest compatibility macros
// ---------------------------------------------------------------------------
/// SUCCEED() — explicitly marks the test as passing (no-op in our framework).
#define SUCCEED() do {} while(0)

// ---------------------------------------------------------------------------
// TEST() — define and register a test case
// ---------------------------------------------------------------------------
#define TEST(Suite, Name)                                                        \
  static void _test_##Suite##_##Name();                                          \
  static ::testing::Registrar _reg_##Suite##_##Name(                             \
      #Suite, #Name, _test_##Suite##_##Name);                                    \
  static void _test_##Suite##_##Name()

// ---------------------------------------------------------------------------
// Non-fatal EXPECT_* macros
// ---------------------------------------------------------------------------
#define EXPECT_TRUE(expr)                                                        \
  do { if (!(expr)) ::testing::record_failure(                                   \
      __FILE__, __LINE__, "EXPECT_TRUE(" #expr ") failed"); } while(0)

#define EXPECT_FALSE(expr)                                                       \
  do { if (!!(expr)) ::testing::record_failure(                                  \
      __FILE__, __LINE__, "EXPECT_FALSE(" #expr ") failed"); } while(0)

#define EXPECT_EQ(a, b)                                                          \
  do {                                                                           \
    auto _a = (a); auto _b = (b);                                               \
    if (!::testing::equal_cmp(_a, _b)) {                                         \
      std::ostringstream _os;                                                    \
      _os << "EXPECT_EQ(" #a ", " #b "): " << _a << " != " << _b;              \
      ::testing::record_failure(__FILE__, __LINE__, _os.str());                  \
    }                                                                            \
  } while(0)

#define EXPECT_NE(a, b)                                                          \
  do {                                                                           \
    auto _a = (a); auto _b = (b);                                               \
    if (::testing::equal_cmp(_a, _b)) {                                          \
      std::ostringstream _os;                                                    \
      _os << "EXPECT_NE(" #a ", " #b "): both equal " << _a;                    \
      ::testing::record_failure(__FILE__, __LINE__, _os.str());                  \
    }                                                                            \
  } while(0)

#define EXPECT_NEAR(a, b, tol)                                                   \
  do {                                                                           \
    auto _a = (double)(a); auto _b = (double)(b); auto _t = (double)(tol);      \
    if (!(std::abs(_a - _b) <= _t)) {                                            \
      std::ostringstream _os;                                                    \
      _os << "EXPECT_NEAR(" #a ", " #b ", " #tol "): |"                         \
          << _a << " - " << _b << "| = " << std::abs(_a - _b)                   \
          << " > " << _t;                                                        \
      ::testing::record_failure(__FILE__, __LINE__, _os.str());                  \
    }                                                                            \
  } while(0)

// Matches GTest's 4-ULP tolerance: |a-b| <= 4 * eps * max(|a|, |b|, 1.0)
#define EXPECT_DOUBLE_EQ(a, b)                                                   \
  EXPECT_NEAR(a, b, 4e-15 * std::max({std::abs((double)(a)),                    \
                                       std::abs((double)(b)), 1.0}))

#define EXPECT_LT(a, b)                                                          \
  do {                                                                           \
    auto _a = (a); auto _b = (b);                                               \
    if (!(_a < _b)) {                                                            \
      std::ostringstream _os;                                                    \
      _os << "EXPECT_LT(" #a ", " #b "): " << _a << " >= " << _b;              \
      ::testing::record_failure(__FILE__, __LINE__, _os.str());                  \
    }                                                                            \
  } while(0)

#define EXPECT_LE(a, b)                                                          \
  do {                                                                           \
    auto _a = (a); auto _b = (b);                                               \
    if (!(_a <= _b)) {                                                           \
      std::ostringstream _os;                                                    \
      _os << "EXPECT_LE(" #a ", " #b "): " << _a << " > " << _b;               \
      ::testing::record_failure(__FILE__, __LINE__, _os.str());                  \
    }                                                                            \
  } while(0)

#define EXPECT_GT(a, b)                                                          \
  do {                                                                           \
    auto _a = (a); auto _b = (b);                                               \
    if (!(_a > _b)) {                                                            \
      std::ostringstream _os;                                                    \
      _os << "EXPECT_GT(" #a ", " #b "): " << _a << " <= " << _b;              \
      ::testing::record_failure(__FILE__, __LINE__, _os.str());                  \
    }                                                                            \
  } while(0)

#define EXPECT_GE(a, b)                                                          \
  do {                                                                           \
    auto _a = (a); auto _b = (b);                                               \
    if (!(_a >= _b)) {                                                           \
      std::ostringstream _os;                                                    \
      _os << "EXPECT_GE(" #a ", " #b "): " << _a << " < " << _b;               \
      ::testing::record_failure(__FILE__, __LINE__, _os.str());                  \
    }                                                                            \
  } while(0)

#define EXPECT_THROW(expr, ExcType)                                              \
  do {                                                                           \
    bool _threw = false;                                                         \
    try { (void)(expr); }                                                        \
    catch (const ExcType&) { _threw = true; }                                    \
    catch (...) {}                                                               \
    if (!_threw) ::testing::record_failure(                                       \
        __FILE__, __LINE__,                                                      \
        "EXPECT_THROW(" #expr ", " #ExcType "): did not throw");                 \
  } while(0)

#define EXPECT_NO_THROW(expr)                                                    \
  do {                                                                           \
    try { (void)(expr); }                                                        \
    catch (const std::exception& _e) {                                           \
      std::ostringstream _os;                                                    \
      _os << "EXPECT_NO_THROW(" #expr "): threw: " << _e.what();                \
      ::testing::record_failure(__FILE__, __LINE__, _os.str());                  \
    } catch (...) {                                                              \
      ::testing::record_failure(__FILE__, __LINE__,                              \
          "EXPECT_NO_THROW(" #expr "): threw unknown exception");                \
    }                                                                            \
  } while(0)

// ---------------------------------------------------------------------------
// Fatal ASSERT_* macros — throw TestFailure to abort the current test
// ---------------------------------------------------------------------------
#define ASSERT_TRUE(expr)                                                        \
  do { if (!(expr)) {                                                            \
    ::testing::record_failure(                                                   \
        __FILE__, __LINE__, "ASSERT_TRUE(" #expr ") failed");                    \
    throw ::testing::TestFailure{};                                              \
  } } while(0)

#define ASSERT_FALSE(expr)                                                       \
  do { if (!!(expr)) {                                                           \
    ::testing::record_failure(                                                   \
        __FILE__, __LINE__, "ASSERT_FALSE(" #expr ") failed");                   \
    throw ::testing::TestFailure{};                                              \
  } } while(0)

#define ASSERT_EQ(a, b)                                                          \
  do {                                                                           \
    auto _a = (a); auto _b = (b);                                               \
    if (!::testing::equal_cmp(_a, _b)) {                                         \
      std::ostringstream _os;                                                    \
      _os << "ASSERT_EQ(" #a ", " #b "): " << _a << " != " << _b;              \
      ::testing::record_failure(__FILE__, __LINE__, _os.str());                  \
      throw ::testing::TestFailure{};                                            \
    }                                                                            \
  } while(0)

#define ASSERT_NE(a, b)                                                          \
  do {                                                                           \
    auto _a = (a); auto _b = (b);                                               \
    if (::testing::equal_cmp(_a, _b)) {                                          \
      std::ostringstream _os;                                                    \
      _os << "ASSERT_NE(" #a ", " #b "): both equal " << _a;                    \
      ::testing::record_failure(__FILE__, __LINE__, _os.str());                  \
      throw ::testing::TestFailure{};                                            \
    }                                                                            \
  } while(0)

#define ASSERT_NEAR(a, b, tol)                                                   \
  do {                                                                           \
    auto _a = (double)(a); auto _b = (double)(b); auto _t = (double)(tol);      \
    if (!(std::abs(_a - _b) <= _t)) {                                            \
      std::ostringstream _os;                                                    \
      _os << "ASSERT_NEAR(" #a ", " #b ", " #tol "): |"                         \
          << _a << " - " << _b << "| = " << std::abs(_a - _b)                   \
          << " > " << _t;                                                        \
      ::testing::record_failure(__FILE__, __LINE__, _os.str());                  \
      throw ::testing::TestFailure{};                                            \
    }                                                                            \
  } while(0)

#define ASSERT_THROW(expr, ExcType)                                              \
  do {                                                                           \
    bool _threw = false;                                                         \
    try { (void)(expr); }                                                        \
    catch (const ExcType&) { _threw = true; }                                    \
    catch (...) {}                                                               \
    if (!_threw) {                                                               \
      ::testing::record_failure(__FILE__, __LINE__,                              \
          "ASSERT_THROW(" #expr ", " #ExcType "): did not throw");               \
      throw ::testing::TestFailure{};                                            \
    }                                                                            \
  } while(0)
