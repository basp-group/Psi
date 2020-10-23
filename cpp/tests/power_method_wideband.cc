#include <numeric>
#include <random>
#include <vector>
#include <Eigen/Eigenvalues>
#include "catch.hpp"

#include "psi/power_method_wideband.h"

TEST_CASE("Power Method Wideband") {

  using namespace psi;
  typedef t_real Scalar;
  typedef psi::LinearTransform<Vector<t_complex>> t_LinearTransform;
  auto const N = 10;
  Eigen::EigenSolver<Matrix<Scalar>> es;

  // Create linear operator
  Matrix<Scalar> B(N, N);
  std::iota(B.data(), B.data() + B.size(), 0);
  es.compute(B.adjoint() * B, true);

  auto const eigenvalues = es.eigenvalues();
  auto const eigenvectors = es.eigenvectors();
  Eigen::DenseIndex index;
  (eigenvalues.transpose() * eigenvalues).real().maxCoeff(&index);
  auto const eigenvalue = eigenvalues(index);
  Vector<t_complex> const eigenvector = eigenvectors.col(index);
  // Create input vector close to solution
  Matrix<t_complex> const input = eigenvector * 1e-4 + Matrix<t_complex>::Random(N, 1); // see if no problem with the type...
  auto const pm = algorithm::PowerMethodWideband<t_complex>().tolerance(1e-12);

  SECTION("AtA") {
    std::vector<psi::LinearTransform<psi::Vector<psi::t_complex>>> A(1, linear_transform(B.cast<t_complex>()));   
    // auto const result = psi::algorithm::power_method_wideband<Vector<t_complex>, Matrix<t_complex>>(A, 100, 1e-12, input); // problem

    auto const result = pm.AtA(A, input); // 
    CHECK(result.good);
    CAPTURE(eigenvalue);
    CAPTURE(result.magnitude);
    CAPTURE(result.eigenvector.transpose() * eigenvector);
    CHECK(std::abs(result.magnitude - std::abs(eigenvalue)) < 1e-8);
  }
}

TEST_CASE("Power Method Wideband 2") {
  using namespace psi;
  typedef t_real Scalar;
  typedef Vector<Scalar> t_Vector;
  typedef LinearTransform<t_Vector> t_LinearTransform;
  auto const N = 10;
  const t_uint power_iters = 1000;
  const t_real power_tol = 1e-4;
  Eigen::EigenSolver<Matrix<Scalar>> es;
  Matrix<Scalar> A(N, N);
  std::iota(A.data(), A.data() + A.size(), 0);
  es.compute(A.adjoint() * A, true);

  auto const eigenvalues = es.eigenvalues();
  auto const eigenvectors = es.eigenvectors();
  Eigen::DenseIndex index;
  (eigenvalues.transpose() * eigenvalues).real().maxCoeff(&index);
  auto const eigenvalue = eigenvalues(index);
  Vector<t_complex> const eigenvector = eigenvectors.col(index);
  Matrix<t_complex> const input = eigenvector * 1e-4 + Matrix<t_complex>::Random(N, 1);

  const auto forward = [=](Vector<t_complex> &out, const Vector<t_complex> &in) { out = A * in; };
  const auto backward
      = [=](Vector<t_complex> &out, const Vector<t_complex> &in) { out = A.adjoint() * in; };

  auto op = std::vector<psi::LinearTransform<Vector<t_complex>>>(1,linear_transform_identity<t_complex>());
  op[0] = psi::linear_transform<Vector<t_complex>>(forward, backward);

 
  
  SECTION("Power Method") {
     auto op_norm = algorithm::power_method_wideband<Vector<t_complex>,Matrix<t_complex>>(op, power_iters, power_tol, input);
     CAPTURE(eigenvalue);
     CAPTURE(op_norm * op_norm);
     CHECK(std::abs(op_norm * op_norm - eigenvalue) < power_tol * power_tol);
  }
}
