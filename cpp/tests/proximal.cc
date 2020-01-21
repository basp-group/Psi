#include <catch.hpp>
#include <numeric>
#include <random>
#include <utility>

#include "psi/l1_proximal.h"
#include "psi/proximal.h"
#include "psi/types.h"

template <class T> psi::Matrix<T> concatenated_permutations(psi::t_uint i, psi::t_uint j) {
  extern std::unique_ptr<std::mt19937_64> mersenne;
  std::vector<size_t> cols(j);
  std::iota(cols.begin(), cols.end(), 0);
  std::shuffle(cols.begin(), cols.end(), *mersenne);

  assert(j % i == 0);
  auto const N = j / i;
  auto const elem = 1e0 / std::sqrt(static_cast<typename psi::real_type<T>::type>(N));
  psi::Matrix<T> result = psi::Matrix<T>::Zero(i, cols.size());
  for(typename psi::Matrix<T>::Index k(0); k < result.cols(); ++k)
    result(cols[k] / N, k) = elem;
  return result;
}

TEST_CASE("L2Ball", "[proximal]") {
  using namespace psi;
  proximal::L2Ball<t_real> ball(0.5);
  Vector<t_real> out;
  Vector<t_real> x(5);
  x << 1, 2, 3, 4, 5;

  out = ball(0, x);
  CHECK(x.isApprox(out / 0.5 * x.stableNorm()));
  ball.epsilon(x.stableNorm() * 1.001);
  out = ball(0, x);
  CHECK(x.isApprox(out));
}

TEST_CASE("WeightedL2Ball", "[proximal]") { // wrong: no closed-form proximal operator in this case
  using namespace psi;
  Vector<t_real> const weights = 0.01 * Vector<t_real>::Random(5).array() + 1e0;
  Vector<t_real> x(5);
  x << 1, 2, 3, 4, 5;
  proximal::WeightedL2Ball<t_real> wball(0.5, weights);
  proximal::L2Ball<t_real> ball(0.5);

  Vector<t_real> const expected
      = ball((x.array() * weights.array()).matrix()).array() / weights.array();
  Vector<t_real> const actual = wball(x);
  CHECK(actual.isApprox(expected));

  wball.epsilon((x.array() * weights.array()).matrix().stableNorm() * 1.001);
  CHECK(x.isApprox(wball(x)));
}

TEST_CASE("Euclidian norm", "[proximal]") {
  using namespace psi;
  proximal::EuclidianNorm eucl;

  Vector<t_real> out(5);
  Vector<t_real> x(5);
  x << 1, 2, 3, 4, 5;
  eucl(out, x.stableNorm() * 1.001, x);
  CHECK(out.isApprox(Vector<t_real>::Zero(x.size())));

  out = eucl(0.1, x);
  CHECK(out.isApprox(x * (1e0 - 0.1 / x.stableNorm())));
}

TEST_CASE("Translation", "[proximal]") {
  using namespace psi;
  Vector<t_real> out(5);
  Vector<t_real> x(5);
  x << 1, 2, 3, 4, 5;
  proximal::L2Ball<t_real> ball(5000);
  // Pass in a reference, so we can modify ball.epsilon later in the test.
  auto const translated = proximal::translate(std::ref(ball), -x * 0.5);
  translated(out, 0, x);
  CHECK(out.isApprox(x));

  ball.epsilon(0.125);
  out = translated(0, x);
  Vector<t_real> expected = ball(1, x * 0.5) + x * 0.5;
  CHECK(out.isApprox(expected));
}

TEST_CASE("Tight-Frame L1 proximal", "[l1][proximal]") {
  using namespace psi;
  auto l1 = proximal::L1TightFrame<t_complex>();
  auto check_is_minimum = [&l1](Vector<t_complex> const &x, t_real gamma = 1e0) {
    typedef t_complex Scalar;
    Vector<t_complex> const p = l1(gamma, x);
    auto const mini = l1.objective(x, p, gamma);
    auto const eps = 1e-4;
    for(Vector<t_complex>::Index i(0); i < p.size(); ++i) {
      for(auto const dir : {Scalar(eps, 0), Scalar(0, eps), Scalar(-eps, 0), Scalar(0, -eps)}) {
        Vector<t_complex> p_plus = p;
        p_plus[i] += dir;
        CHECK(l1.objective(x, p_plus, gamma) >= mini);
      }
    }
  };

  Vector<t_complex> const input = Vector<t_complex>::Random(8);

  // no weights
  SECTION("Scalar weights") {
    CHECK(l1(1, input).isApprox(proximal::l1_norm(1, input)));
    CHECK(l1(0.3, input).isApprox(proximal::l1_norm(0.3, input)));
    check_is_minimum(input, 0.664);
  }

  // with weights == 1
  SECTION("vector weights") {
    l1.weights(Vector<t_real>::Ones(input.size()));
    CHECK(l1(1, input).isApprox(proximal::l1_norm(1, input)));
    CHECK(l1(0.2, input).isApprox(proximal::l1_norm(0.2, input)));
    check_is_minimum(input, 0.664);
  }

  SECTION("vector weights with random values") {
    l1.weights(Vector<t_real>::Random(input.size()).array().abs().matrix());
    check_is_minimum(input, 0.235);
  }

  SECTION("Psi is a concatenation of permutations") {
    auto const psi = concatenated_permutations<t_complex>(input.size(), input.size() * 10);
    l1.Psi(psi).weights(1e0);
    check_is_minimum(input, 0.235);
  }

#ifdef CATCH_HAS_THROWS_AS
  SECTION("Weights cannot be negative") {
    CHECK_THROWS_AS(l1.weights(-1e0), Exception);
    Vector<t_real> weights = Vector<t_real>::Random(5).array().abs().matrix();
    weights[2] = -1;
    CHECK_THROWS_AS(l1.weights(weights), Exception);
  }
#endif
}

TEST_CASE("L1 proximal utilities", "[l1][utilities]") {
  using namespace psi;
  typedef t_complex Scalar;

  SECTION("Mixing") {
    auto const input = Vector<Scalar>::Random(10).eval();
    Vector<Scalar> output;

    SECTION("No Mixing") {
      proximal::L1<Scalar>::NoMixing()(output, 2.1 * input, 0);
      CHECK(output.isApprox(2.1 * input));
      proximal::L1<Scalar>::NoMixing()(output, 4.1 * input, 10);
      CHECK(output.isApprox(4.1 * input));
    }

    SECTION("Fista Mixing") {
      proximal::L1<Scalar>::FistaMixing fista;
      // step zero: no mixing yet
      fista(output, 2.1 * input, 0);
      CHECK(output.isApprox(2.1 * input));
      // step one: first mixing
      fista(output, 3.1 * input, 1);
      auto const alpha = (fista.next(1) - 1) / fista.next(fista.next(1));
      Vector<Scalar> const first = (1e0 + alpha) * 3.1 * input - alpha * 2.1 * input;
      CHECK(output.isApprox(first));
      // step two: second mixing
      fista(output, 4.1 * input, 1);
      auto const beta = (fista.next(fista.next(1)) - 1) / fista.next(fista.next(fista.next(1)));
      Vector<Scalar> const second = (1e0 + alpha) * 4.1 * input - alpha * first;
      CHECK(output.isApprox(second));
    }
  }

  SECTION("Breaker") {
    proximal::L1<Scalar>::Breaker breaker(2e0);
    SECTION("Finds convergence") {
      std::vector<t_real> objectives
          = {1.0, 0.9, 0.5, 0.6, 0.4, 0.4 + 0.41 * 1e-8, 0.3, 0.3 + 0.29 * 1e-8};
      for(size_t i(0); i < objectives.size() - 1; ++i) {
        CHECK(not breaker(objectives[i]));
        CHECK(breaker.current() == Approx(objectives[i]).epsilon(1e-12));
      }
      CHECK(breaker(objectives.back()));
      CHECK(not breaker.two_cycle());
      CHECK(breaker.converged());
    }

    SECTION("Find cycle") {
      std::vector<t_real> objectives = {1.0, 0.9, 0.5, 0.6, 0.4, 0.3, 0.4, 0.3};
      for(size_t i(0); i < objectives.size() - 1; ++i) {
        CHECK(not breaker(objectives[i]));
        CHECK(breaker.current() == Approx(objectives[i]).epsilon(1e-12));
      }
      CHECK(breaker(objectives.back()));
      CHECK(breaker.two_cycle());
      CHECK(not breaker.converged());
    }
  }
}

TEST_CASE("L1 proximal", "[l1][proximal]") {
  using namespace psi;
  typedef t_complex Scalar;
  auto l1 = proximal::L1<Scalar>().tolerance(1e-10);

  Vector<Scalar> const input = Vector<Scalar>::Random(4);

  SECTION("Check against tight-frame") {
    l1.fista_mixing(false);
    SECTION("Scalar weights") {
      auto const result = l1(1, input);
      CHECK(result.good);
      CHECK(result.niters > 0);
      CHECK(l1.itermax() == 0);
      CHECK(result.proximal.isApprox(proximal::L1TightFrame<Scalar>()(1, input)));
    }
    SECTION("Vector weights and more complex Psi") {
      auto const Psi = concatenated_permutations<Scalar>(input.size(), input.size() * 10);
      auto const weights = Vector<t_real>::Random(Psi.cols()).array().abs().matrix().eval();
      auto const gamma = 1e-1 / Psi.array().abs().sum();
      l1.Psi(Psi).weights(weights).tolerance(1e-12);
      auto const result = l1(gamma, input);
      CHECK(result.good);
      CHECK(result.niters > 0);
      auto const expected = l1.tight_frame(gamma, input).eval();
      CHECK(result.objective == Approx(l1.objective(input, expected, gamma)));
      CAPTURE((result.proximal - expected).array().abs().transpose());
      CHECK(result.proximal.isApprox(expected));
    }
  }

  SECTION("General case") {
    auto check_is_minimum = [&l1, &input](t_real gamma, Vector<Scalar> const &proximal) {
      // returns false if did not converge.
      // Looks like computing the proximal does not always work...
      auto const mini = l1.objective(input, proximal, gamma);
      auto const eps = 1e-4;
      // check alongst specific directions
      for(Vector<Scalar>::Index i(0); i < proximal.size(); ++i) {
        for(auto const dir : {Scalar(eps, 0), Scalar(0, eps), Scalar(-eps, 0), Scalar(0, -eps)}) {
          Vector<Scalar> p_plus = proximal;
          p_plus[i] += dir;
          if(l1.positivity_constraint())
            p_plus = psi::positive_quadrant(p_plus);
          else if(l1.real_constraint())
            p_plus = p_plus.real().cast<Scalar>();
          auto const rel_var = std::abs((l1.objective(input, p_plus, gamma) - mini) / mini);
          CHECK((l1.objective(input, p_plus, gamma) > mini or rel_var < l1.tolerance() * 10));
        }
      }
      // check alongst non-specific directions
      for(size_t i(0); i < 10; ++i) {
        Vector<Scalar> p_plus = proximal + proximal.Random(proximal.size()) * eps;
        if(l1.positivity_constraint())
          p_plus = psi::positive_quadrant(p_plus);
        else if(l1.real_constraint())
          p_plus = p_plus.real().cast<Scalar>();
        auto const rel_var = std::abs((l1.objective(input, p_plus, gamma) - mini) / mini);
        CHECK((l1.objective(input, p_plus, gamma) > mini or rel_var < l1.tolerance() * 10));
      }
    };

    auto const Psi = Matrix<Scalar>::Random(input.size(), input.size() * 10).eval();
    auto const weights = Vector<t_real>::Random(Psi.cols()).array().abs().matrix().eval();
    auto const gamma = 1e-1 / Psi.array().abs().sum();

    l1.Psi(Psi).weights(weights).fista_mixing(true).tolerance(1e-10).itermax(5000);

    SECTION("No constraints") {
      CHECK(not l1.positivity_constraint());
      CHECK(not l1.real_constraint());
      auto const result = l1(gamma, input);
      CHECK(result.good);
      check_is_minimum(gamma, result.proximal);
    }
    SECTION("Positivity constraints") {
      l1.positivity_constraint(true);
      CHECK(l1.positivity_constraint());
      CHECK(not l1.real_constraint());
      auto const result = l1(gamma, input);
      CHECK(result.good);
      check_is_minimum(gamma, result.proximal);
    }
    SECTION("Real constraints") {
      l1.real_constraint(true);
      CHECK(l1.real_constraint());
      CHECK(not l1.positivity_constraint());
      auto const result = l1(gamma, input);
      CHECK(result.good);
      check_is_minimum(gamma, result.proximal);
    }
  }
}
TEST_CASE("Nuclear norm proximal", "[Nuclear norm][proximal]"){
	typedef psi::t_real Scalar;
	typedef psi::Matrix<Scalar> Matrix;
  typedef psi::Matrix<psi::t_complex> cMatrix;
	typedef psi::Vector<Scalar> Vector;

	int Nx = 5, Ny = 3;
  Matrix w = Matrix::Ones(Ny,1);

  SECTION("Real case") {
    Matrix out = Matrix(Nx,Ny);
	  Vector Diag = Vector::LinSpaced(3, 3, 0);
	  Matrix x = Matrix(Diag.asDiagonal());
	  psi::proximal::nuclear_norm(out, x, w);
	  CHECK(out == Matrix((Diag-w).cwiseMax(0.).asDiagonal()));
  }
  SECTION("Complex case") {
    cMatrix out = cMatrix(Nx,Ny);
	  Vector Diag = Vector::LinSpaced(3, 3, 0);
	  cMatrix x = cMatrix(Diag.asDiagonal());
	  psi::proximal::nuclear_norm(out, x, w);
	  CHECK(out == cMatrix((Diag-w).cwiseMax(0.).asDiagonal()));
  }
}

TEST_CASE("l21 proximal", "[l21][proximal]") {
  const int N = 10;
  psi::Matrix<psi::t_real> X = psi::Matrix<psi::t_real>::Ones(N, N);
  psi::Vector<psi::t_real> w = psi::Vector<psi::t_real>::Ones(N);
  psi::Matrix<psi::t_real> out(N, N);
  psi::Matrix<psi::t_complex> Y = psi::Matrix<psi::t_complex>::Ones(N, N);
  psi::Matrix<psi::t_complex> out2;

  SECTION("Weighted version") {
    psi::proximal::l21_norm(out, X, w);
    // std::cout << "l21 norm (weigthed)" << out << "\n";
    CHECK(out(0, 0) == Approx((std::sqrt(N) - 1.)/std::sqrt(N)));
    CHECK((out.array() - (std::sqrt(N) - 1.)/std::sqrt(N)).matrix().norm() <= Approx(1e-8));
  }
  SECTION("Scalar weight") {
    psi::proximal::l21_norm(out, X, 2.);
    // std::cout << "l21 norm (scalar)" << out << "\n";
    CHECK(out(0, 0) == Approx((std::sqrt(N) - 2.)/std::sqrt(N)));
    CHECK((out.array() - (std::sqrt(N) - 2.)/std::sqrt(N)).matrix().norm() <= Approx(1e-8));
  }
  SECTION("Weighted version (complex input)") {
    psi::proximal::l21_norm(out2, Y, w);
    // std::cout << "l21 norm (weigthed complex)" << out2 << "\n";
    CHECK(abs(out2(0, 0)) == Approx((std::sqrt(N) - 1.)/std::sqrt(N)));
    CHECK((out2.array() - (std::sqrt(N) - 1.)/std::sqrt(N)).matrix().norm() <= Approx(1e-8));
  }
  SECTION("Scalar weight (complex input)") {
    psi::proximal::l21_norm(out2, Y, 2.);
    // std::cout << "l21 norm (complex scalar)" << out2 << "\n";
    CHECK(abs(out2(0, 0)) == Approx((std::sqrt(N) - 2.)/std::sqrt(N)));
    CHECK((out2.array() - (std::sqrt(N) - 2.)/std::sqrt(N)).matrix().norm() <= Approx(1e-8));
  }
}


