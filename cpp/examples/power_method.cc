#include <numeric>
#include <Eigen/Eigenvalues>
#include <psi/power_method.h>

int main(int, char const **) {

  typedef psi::t_real Scalar;
  auto const N = 10;

  // Create some kind of matrix
  psi::Matrix<Scalar> const A
      = psi::Matrix<uint8_t>::Identity(N, N).cast<Scalar>() * 100 + psi::Matrix<Scalar>(N, N);

  // the linear transform wraps the matrix into something the power-method understands
  auto const lt = psi::linear_transform(A.cast<psi::t_complex>());
  // instanciate the power method
  auto const pm = psi::algorithm::PowerMethod<psi::t_complex>().tolerance(1e-12);
  // call it
  auto const result = pm.AtA(lt, psi::Vector<psi::t_complex>::Ones(N));

  // Compute the eigen values explictly to figure out the result of the power method
  Eigen::EigenSolver<psi::Matrix<Scalar>> es;
  es.compute(A.adjoint() * A, true);
  Eigen::DenseIndex index;
  (es.eigenvalues().transpose() * es.eigenvalues()).real().maxCoeff(&index);
  auto const eigenvalue = es.eigenvalues()(index);

  // This should pass if the power method is correct
  if(std::abs(result.magnitude - std::abs(eigenvalue)) > 1e-8 * std::abs(eigenvalue))
    throw std::runtime_error("Power method did not converge to the expected value");

  return 0;
}
