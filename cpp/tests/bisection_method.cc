#include <catch2/catch.hpp>
#include <random>

#include <Eigen/Dense>

#include "psi/bisection_method.h"
#include "psi/types.h"

typedef psi::t_real Scalar;
typedef psi::Vector<Scalar> t_Vector;
typedef psi::Matrix<Scalar> t_Matrix;

const Scalar a = -0.5;
const Scalar b = 1.;
const Scalar tol = 1e-4;
TEST_CASE("Bisection x^3") {
  using namespace psi;
  psi::logging::set_level("debug");
  std::function<Scalar(Scalar)> const func = [](const Scalar &x) -> Scalar { return x * x * x; };
  const Scalar x0 = 0;
  const Scalar x0_est = bisection_method(func(x0), func, a, b, tol);
  CHECK(std::abs(x0_est - x0) <= tol);
}
TEST_CASE("Bisection f(x) = x") {
  using namespace psi;
  std::function<Scalar(Scalar)> const func = [](const Scalar &x) -> Scalar { return x; };
  const Scalar x0 = 0.23235104239409;
  const Scalar x0_est = bisection_method(func(x0), func, a, b, tol);
  CHECK(std::abs(x0_est - x0) <= tol);
}
TEST_CASE("Bisection exp()") {
  using namespace psi;
  std::function<Scalar(Scalar)> const func = [](const Scalar &x) -> Scalar { return std::exp(-x); };
  const Scalar x0 = 1;
  const Scalar x0_est = bisection_method(func(x0), func, a, b, tol);
  CHECK(std::abs(x0_est - x0) <= tol);
}
