#include <catch2/catch.hpp>
#include <numeric>
#include <random>
#include <utility>

#include "psi/maths.h"
#include "psi/relative_variation.h"
#include "psi/sampling.h"
#include "psi/types.h"

TEST_CASE("Projector on positive quadrant", "[utility][project]") {
  using namespace psi;

  SECTION("Real matrix") {
    Image<> input = Image<>::Ones(5, 5) + Image<>::Random(5, 5) * 0.55;
    input(1, 1) *= -1;
    input(3, 2) *= -1;

    auto const expr = positive_quadrant(input);
    CAPTURE(input);
    CAPTURE(expr);
    CHECK(expr(1, 1) == Approx(0));
    CHECK(expr(3, 2) == Approx(0));

    auto value = expr.eval();
    CHECK(value(1, 1) == Approx(0));
    CHECK(value(3, 2) == Approx(0));
    value(1, 1) = input(1, 1);
    value(3, 2) = input(3, 2);
    CHECK(value.isApprox(input));
  }

  SECTION("Complex matrix") {
    Image<t_complex> input = Image<t_complex>::Ones(5, 5) + Image<t_complex>::Random(5, 5) * 0.55;
    input.real()(1, 1) *= -1;
    input.real()(3, 2) *= -1;

    auto const expr = positive_quadrant(input);
    CAPTURE(input);
    CAPTURE(expr);
    CHECK(expr.imag().isApprox(Image<>::Zero(5, 5)));

    auto value = expr.eval();
    CHECK(value.real()(1, 1) == Approx(0));
    CHECK(value.real()(3, 2) == Approx(0));
    value(1, 1) = input.real()(1, 1);
    value(3, 2) = input.real()(3, 2);
    CHECK(value.real().isApprox(input.real()));
    CHECK(value.imag().isApprox(0e0 * input.real()));
  }
}

TEST_CASE("Weighted l1 norm", "[utility][l1]") {
  psi::Array<> weight(4);
  weight << 1, 2, 3, 4;

  SECTION("Real valued") {
    psi::Array<> input(4);
    input << 5, -6, 7, -8;
    CHECK(psi::l1_norm(input, weight) == Approx(5 + 12 + 21 + 32));
  }
  SECTION("Complex valued") {
    psi::t_complex const i(0, 1);
    psi::Array<psi::t_complex> input(4);
    input << 5. + 5. * i, 6. + 6. * i, 7. + 7. * i, 8. + 8. * i;
    CHECK(psi::l1_norm(input, weight) == Approx(std::sqrt(2) * (5 + 12 + 21 + 32)));
  }
}

TEST_CASE("Soft threshhold", "[utility][threshhold]") {
  psi::Array<> input(6);
  input << 1e1, 2e1, 3e1, 4e1, 1e4, 2e4;

  SECTION("Single-valued threshhold") {
    // check thresshold
    CHECK(psi::soft_threshhold(input, 1.1e1)(0) == Approx(0));
    CHECK(not(psi::soft_threshhold(input, 1.1e1)(1) == Approx(0)));

    // check linearity
    auto a = psi::soft_threshhold(input, 9e0)(0);
    auto b = psi::soft_threshhold(input, 4.5e0)(0);
    auto c = psi::soft_threshhold(input, 2.25e0)(0);
    CAPTURE(b - a);
    CAPTURE(c - b);
    CHECK((b - a) == Approx(2 * (c - b)));
  }

  SECTION("Multi-values threshhold") {
    using namespace psi;
    Array<> threshhold(6);
    input[2] *= -1;
    threshhold << 1.1e1, 1.1e1, 1e0, 4.5, 2.25, 2.26;

    SECTION("Real input") {
      Array<> const actual = soft_threshhold(input, threshhold);
      CHECK(actual(0) == 0e0);
      CHECK(actual(1) == input(1) - threshhold(1));
      CHECK(actual(2) == input(2) + threshhold(2));
      CHECK(actual(3) == input(3) - threshhold(3));

#ifdef CATCH_HAS_THROWS_AS
      CHECK_THROWS_AS(soft_threshhold(input, threshhold.head(2)), psi::Exception);
#endif
    }
    SECTION("Complex input") {
      Array<t_complex> const actual = soft_threshhold(input.cast<t_complex>(), threshhold);
      CHECK(actual(0) == 0e0);
      CHECK(actual(1) == input(1) - threshhold(1));
      CHECK(actual(2) == input(2) + threshhold(2));
      CHECK(actual(3) == input(3) - threshhold(3));

#ifdef CATCH_HAS_THROWS_AS
      CHECK_THROWS_AS(soft_threshhold(input, threshhold.head(2)), psi::Exception);
#endif
    }
  }
}


TEST_CASE("l21 norm", "[utility][l21]") {
  using cd = std::complex<double>;
  psi::Matrix<double> X = psi::Matrix<double>::Ones(3, 3);
  X << 1., 2., 3.,
       1., 2., 3.,
       4., 5., 6.;

  psi::Matrix<psi::t_complex> Y = psi::Matrix<psi::t_complex>::Zero(3, 3);
  Y << cd(1., 0.), cd(2., .0), cd(3., 0.),
       cd(1., 0.), cd(2., 0.), cd(3., 0.),
       cd(4., 0.), cd(5., 0.), cd(6., 0.);

  SECTION("Standard version") {
    CHECK(psi::l21_norm(X) == Approx(2*std::sqrt(14) + std::sqrt(16 + 25 + 36)));
  }
  SECTION("Weighted version") {
    psi::Vector<double> w = 2*psi::Vector<double>::Ones(3);
    CHECK(psi::l21_norm(X, w) == Approx(2*(2*std::sqrt(14) + std::sqrt(16 + 25 + 36))));
  }
  SECTION("Weighted version (complex input)") {
    psi::Vector<double> w = 2*psi::Vector<double>::Ones(3);
    CHECK(psi::l21_norm(Y, w) == Approx(2*(2*std::sqrt(14) + std::sqrt(16 + 25 + 36))));
  }
}

TEST_CASE("Nuclear norm", "[utility][nuclear]") {
  psi::Matrix<double> X = psi::Matrix<double>::Ones(10, 10);
  psi::Matrix<psi::t_complex> Y = psi::Matrix<psi::t_complex>::Ones(10, 10);
  psi::Vector<double> w = 2*psi::Vector<double>::Ones(10);
  Eigen::BDCSVD<psi::Matrix<double>> svd(X);
  Eigen::BDCSVD<psi::Matrix<psi::t_complex>> svdY(Y);

  SECTION("Standard version") {
    CHECK(psi::nuclear_norm(X) == Approx(10.));
  }
  SECTION("Standard version (singular values)") {
    CHECK(psi::nuclear_norm(X, w) == Approx(20.));
  }
  SECTION("Standard version (singular values, complex input)") {
    CHECK(psi::nuclear_norm(Y, w) == Approx(20.));
  }
  SECTION("Weighted version") {
    CHECK(psi::nuclear_norm(svd.singularValues()) == Approx(10.));   
  }
  SECTION("Weighted version (complex input)") {
    CHECK(psi::nuclear_norm(svdY.singularValues()) == Approx(10.));   
  }
  SECTION("Weighted version (singular values)") {
    CHECK(psi::nuclear_norm(svd.singularValues(), w) == Approx(20.));  
  }
}

TEST_CASE("Sampling", "[utility][sampling]") {
  typedef psi::Vector<int> t_Vector;
  t_Vector const input = t_Vector::Random(12);

  psi::Sampling const sampling(12, {1, 3, 6, 5});

  t_Vector down = t_Vector::Zero(4);
  sampling(down, input);
  CHECK(down.size() == 4);
  CHECK(down(0) == input(1));
  CHECK(down(1) == input(3));
  CHECK(down(2) == input(6));
  CHECK(down(3) == input(5));

  t_Vector up = t_Vector::Zero(input.size());
  sampling.adjoint(up, down);
  CHECK(up(1) == input(1));
  CHECK(up(3) == input(3));
  CHECK(up(6) == input(6));
  CHECK(up(5) == input(5));
  up(1) = 0;
  up(3) = 0;
  up(6) = 0;
  up(5) = 0;
  CHECK(up == t_Vector::Zero(up.size()));
}

TEST_CASE("Relative variation", "[utility][convergence]") {
  psi::RelativeVariation<double> relvar(1e-8);

  psi::Array<> input = psi::Array<>::Random(6);
  CHECK(not relvar(input));
  CHECK(relvar(input));
  CHECK(relvar(input + relvar.tolerance() * 0.5 / 6. * psi::Array<>::Random(6)));
  CHECK(not relvar(input + relvar.tolerance() * 1.1 * psi::Array<>::Ones(6)));
}

TEST_CASE("Scalar elative variation", "[utility][convergence]") {
  psi::ScalarRelativeVariation<double> relvar(1e-8, 1e-8, "Yo");
  psi::t_real input = psi::Array<>::Random(1)(0);
  CHECK(not relvar(input));
  CHECK(relvar(input));
  CHECK(not relvar(input + 0.1));
  CHECK(relvar(input + 0.1 + 0.1 * relvar.relative_tolerance()));
}

TEST_CASE("Standard deviation", "[utility]") {
  psi::Array<psi::t_complex> input = psi::Array<psi::t_complex>::Random(6) + 1e0;
  psi::t_complex mean = input.mean();
  psi::t_real stddev = 0e0;
  for(psi::Vector<>::Index i(0); i < input.size(); ++i)
    stddev += std::real(std::conj(input(i) - mean) * (input(i) - mean));
  stddev = std::sqrt(stddev) / std::sqrt(psi::t_real(input.size()));

  CHECK(std::abs(psi::standard_deviation(input) - stddev) < 1e-8);
  CHECK(std::abs(psi::standard_deviation(input.matrix()) - stddev) < 1e-8);
}

// Checks type traits work
static_assert(not psi::details::HasValueType<double>::value, "");
static_assert(not psi::details::HasValueType<std::pair<double, int>>::value, "");
static_assert(psi::details::HasValueType<std::complex<double>>::value, "");
static_assert(psi::details::HasValueType<psi::Image<psi::t_complex>::Scalar>::value, "");

static_assert(std::is_same<psi::real_type<psi::t_real>::type, psi::t_real>::value, "");
static_assert(std::is_same<psi::real_type<psi::t_complex>::type, psi::t_real>::value, "");

static_assert(psi::is_complex<std::complex<double>>::value, "Testing is_complex");
static_assert(psi::is_complex<std::complex<int>>::value, "Testing is_complex");
static_assert(not psi::is_complex<double>::value, "Testing is_complex");
static_assert(not psi::is_complex<psi::Vector<double>>::value, "Testing is_complex");
static_assert(not psi::is_complex<psi::Vector<std::complex<int>>>::value, "Testing is_complex");
