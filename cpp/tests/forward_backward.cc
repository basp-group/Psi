#include <catch2/catch.hpp>
#include <random>
#include <ctime>
#include <assert.h>

#include <Eigen/Dense>

#include "psi/forward_backward.h"
#include "psi/proximal.h"
#include "psi/types.h"

psi::t_int random_integer(psi::t_int min, psi::t_int max) {
  extern std::unique_ptr<std::mt19937_64> mersenne;
  std::uniform_int_distribution<psi::t_int> uniform_dist(min, max);
  return uniform_dist(*mersenne);
};

typedef psi::t_real Scalar;
typedef psi::Vector<Scalar> t_Vector;
typedef psi::Matrix<Scalar> t_Matrix;

// to be adapted from this point

auto const N = 5;

TEST_CASE("Forward-Backward, testing norm(output - data()) < l2ball_epsilon()", "[forwardbackward][integration]") {
  using namespace psi;

  t_Vector const ui = t_Vector::Ones(N);

  t_Vector const data = t_Vector::Zero(N);

  t_Vector target = t_Vector::Random(N);

  target = psi::positive_quadrant(target);

  t_Vector weights = t_Vector::Zero(1);
  weights(0) = 1.0;

  auto const epsilon = psi::l2_norm(target, weights)/2;

  auto const forwardbackward = algorithm::ForwardBackward<Scalar>(target, data)
                         .Ui(ui)
                         .itermax(5000)
                         .l2ball_epsilon(epsilon)
                         .relative_variation(1e-6);

  auto const result = forwardbackward();

  CHECK((result.x - data).stableNorm() <= epsilon);
}

TEST_CASE("Forward Backward, testing norm(output - data()) < l2ball_epsilon() where data vector is mainly zero", "[forwardbackward][integration]") {
  using namespace psi;

  auto const seed = std::time(0);
  std::srand((unsigned int)seed);
  std::mt19937 mersenne(std::time(0));

  t_Matrix const ui = t_Vector::Ones(N);

  t_Vector const data = t_Vector::Zero(N);

  t_Vector target = t_Vector::Zero(N);

  t_uint non_random_element = 1;

  // Check that the element chosen to be the non-random element in the array is within the array bounds
  assert(non_random_element < N && non_random_element >= 0);

  target(1) = std::rand();

  t_Vector weights = t_Vector::Zero(1);
  t_Vector out;

  weights(0) = 1.0;

  auto const epsilon = psi::l2_norm(target, (weights.array()).sqrt().matrix())/2;

  auto const forwardbackward = algorithm::ForwardBackward<Scalar>(target, data)
                               .Ui(ui)
                               .itermax(5000)
                               .l2ball_epsilon(epsilon)
                               .relative_variation(1e-6);

  auto const result = forwardbackward();

  proximal::L2Ball<Scalar> l2ball_proximal = proximal::L2Ball<Scalar>(epsilon);
  out = l2ball_proximal(0, target - data) + data;
  CHECK((out - data).stableNorm() <= epsilon); // ok...

  CHECK((result.x - data).stableNorm() <= epsilon);
  // Check decrease of the objective function
  CHECK(psi::l2_norm(result.x - target, (ui.array()).sqrt().matrix()) <= psi::l2_norm(target, (ui.array()).sqrt().matrix()));
  // CHECK(psi::l2_norm(result.x - data, ui.sqrt()) <= epsilon);
  // Check that the elements of the solution are zero where the elements of the target vector are zero.
  for(t_uint i=0; i < N; ++i){
    if(target(i) == 0){
      CHECK(result.x(i) == Approx(0));
    }else{
      CHECK(result.x(i) != Approx(0));
    }
  }
}

template <class T>
struct is_forward_backward_ref
    : public std::is_same<psi::algorithm::ForwardBackward<double> &, T> {};
TEST_CASE("Check type returned on setting variables") {
  // Yeah, could be static asserts [-> to be modified]
  using namespace psi;
  using namespace psi::algorithm;
  ForwardBackward<double> fb(Vector<double>::Zero(0), Vector<double>::Zero(0));
  CHECK(is_forward_backward_ref<decltype(fb.itermax(500))>::value);
  CHECK(is_forward_backward_ref<decltype(fb.relative_variation(5e-4))>::value);
  CHECK(is_forward_backward_ref<decltype(fb.l2ball_epsilon(1e-4))>::value);
  // CHECK(is_forward_backward_ref<decltype(fb.residual_convergence(1.001))>::value);


  typedef ConvergenceFunction<double> ConvFunc;
  CHECK(is_forward_backward_ref<decltype(fb.is_converged(std::declval<ConvFunc>()))>::value);
  CHECK(is_forward_backward_ref<decltype(fb.is_converged(std::declval<ConvFunc &>()))>::value);
  CHECK(is_forward_backward_ref<decltype(fb.is_converged(std::declval<ConvFunc &&>()))>::value);
  CHECK(is_forward_backward_ref<decltype(fb.is_converged(std::declval<ConvFunc const &>()))>::value);

  // CHECK(is_forward_backward_ref<decltype(fb.Ui(std::declval<t_Vector>()))>::value);
  // CHECK(is_forward_backward_ref<decltype(fb.Ui(std::declval<t_Vector &&>()))>::value);
  // CHECK(is_forward_backward_ref<decltype(fb.Ui(std::declval<t_Vector &>()))>::value);
  // CHECK(is_forward_backward_ref<decltype(fb.Ui(std::declval<t_Vector const &>()))>::value);
}
