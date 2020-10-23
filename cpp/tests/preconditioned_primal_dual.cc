#include <catch2/catch.hpp>
#include <random>
#include <ctime>
#include <assert.h>

#include <Eigen/Dense>

#include "psi/preconditioned_primal_dual.h"
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

auto const N = 5;


TEST_CASE("Preconditioned Primal Dual, testing norm(output - target()) < l2ball_epsilon()", "[preconditionedprimaldual][integration]") {
  using namespace psi;

  t_Matrix const mId = t_Matrix::Identity(N, N);
  t_Vector const Ui = t_Vector::Ones(N);

  auto const sigma1 = 1.0;
  auto const sigma2 = 1.0;

  t_Vector target = t_Vector::Random(N);

  target = psi::positive_quadrant(target);

  t_Vector weights = t_Vector::Zero(1);
  weights(0) = 1.0;

  auto const epsilon = psi::l2_norm(target, weights)/2;

  auto preconditionedprimaldual = algorithm::PreconditionedPrimalDual<Scalar>(target)
                         .Phi(mId)
                         .Psi(mId)
                         .itermax(5000)
                         .kappa(0.1)
                         .tau(0.49)
                         .l2ball_epsilon(epsilon)
                         .relative_variation(1e-4)
                         .residual_convergence(epsilon);


  auto result = preconditionedprimaldual();

  CHECK((result.x - target).stableNorm() <= epsilon);
}


TEST_CASE("Preconditioned Primal Dual, testing norm(output - target()) < l2ball_epsilon() no positive quadrant", "[preconditionedprimaldual][integration]") {
  using namespace psi;

  t_Matrix const mId = t_Matrix::Identity(N, N);
  t_Vector const Ui = t_Vector::Ones(N);

  auto const sigma1 = 1.0;
  auto const sigma2 = 1.0;

  t_Vector target = t_Vector::Random(N);

  t_Vector weights = t_Vector::Zero(1);
  weights(0) = 1.0;

  auto const epsilon = psi::l2_norm(target, weights)/2;

  auto preconditionedprimaldual = algorithm::PreconditionedPrimalDual<Scalar>(target)
                         .Ui(Ui)
                         .Phi(mId)
                         .Psi(mId)
                         .itermax(5000)
                         .kappa(0.1)
                         .tau(0.49)
                         .l2ball_epsilon(epsilon)
                         .positivity_constraint(false)
                         .relative_variation(1e-4)
                         .residual_convergence(epsilon);


  auto result = preconditionedprimaldual();

  CHECK((result.x - target).stableNorm() <= epsilon);
}


TEST_CASE("Preconditioned Primal Dual, testing norm(output - target()) < l2ball_epsilon() where target vector is mainly zero", "[preconditionedprimaldual][integration]") {
  using namespace psi;

  auto const seed = std::time(0);
  std::srand((unsigned int)seed);
  std::mt19937 mersenne(std::time(0));

  t_Matrix const mId = t_Matrix::Identity(N, N);
  t_Vector const Ui = t_Vector::Ones(N);

  auto const sigma1 = 1.0;
  auto const sigma2 = 1.0;

  t_Vector target = t_Vector::Zero(N);

  t_uint non_random_element = 1;

  // Check that the element chosen to be the non-random element in the array is within the array bounds
  assert(non_random_element < N && non_random_element >= 0);

  target(1) = std::rand();

  t_Vector weights = t_Vector::Zero(1);

  weights(0) = 1.0;

  auto const epsilon = psi::l2_norm(target, weights)/2;

  auto preconditionedprimaldual = algorithm::PreconditionedPrimalDual<Scalar>(target)
                         .Ui(Ui)
                         .Phi(mId)
                         .Psi(mId)
                         .itermax(5000)
                         .kappa(0.1)
                         .tau(0.49)
                         .l2ball_epsilon(epsilon)
                         .relative_variation(1e-4)
                         .residual_convergence(epsilon);

  auto result = preconditionedprimaldual();


  CHECK((result.x - target).stableNorm() <= epsilon);
  // Check that the elements of the solution are zero where the elements of the target vector are zero.
  for(t_uint i=0; i < N; ++i){
    if(target(i) == 0){
      CHECK(result.x(i) == Approx(0));
    }else{
      CHECK(result.x(i) != Approx(0));
    }
  }
}


// template <class T> // problem with default values inherited from PrimalDual
// struct is_preconditioned_primal_dual_ref
//     : public std::is_same<psi::algorithm::PreconditionedPrimalDual<double> &, T> {};
// TEST_CASE("Check type returned on setting variables") {
//   // Yeah, could be static asserts
//   using namespace psi;
//   using namespace psi::algorithm;
//   PreconditionedPrimalDual<double> ppd(Vector<double>::Zero(0));
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.itermax(500))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.kappa(1))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.relative_variation(5e-4))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.l2ball_epsilon(1e-4))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.residual_convergence(1.001))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.nu(1e0))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.sigma1(1e0))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.sigma2(1e0))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.levels(1))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.positivity_constraint(true))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.l1_proximal_weights(Vector<double>::Zero(0)))>::value);
//
//
//   typedef ConvergenceFunction<double> ConvFunc;
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.is_converged(std::declval<ConvFunc>()))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.is_converged(std::declval<ConvFunc &>()))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.is_converged(std::declval<ConvFunc &&>()))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.is_converged(std::declval<ConvFunc const &>()))>::value);
//
//   typedef LinearTransform<Vector<double>> LinTrans;
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.Phi(linear_transform_identity<double>()))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.Phi(std::declval<LinTrans>()))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.Phi(std::declval<LinTrans &&>()))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.Phi(std::declval<LinTrans &>()))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.Phi(std::declval<LinTrans const &>()))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.Psi(linear_transform_identity<double>()))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.Psi(std::declval<LinTrans>()))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.Psi(std::declval<LinTrans &&>()))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.Psi(std::declval<LinTrans &>()))>::value);
//   CHECK(is_preconditioned_primal_dual_ref<decltype(ppd.Psi(std::declval<LinTrans const &>()))>::value);
// }
