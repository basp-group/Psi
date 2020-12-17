#include <numeric>
#include <Eigen/Eigenvalues>
#include <psi/power_method_wideband.h>

int main(int, char const **) {

  typedef psi::t_real Scalar;
  auto const N = 10;
  auto const band_number = 3;

  // Create some kind of linear operator (for the wideband linear transform)
  psi::Matrix<Scalar> const A
      = psi::Matrix<uint8_t>::Identity(N, N).cast<Scalar>() * 100 + psi::Matrix<Scalar>(N, N);

  // the linear transform wraps the matrix into something the power-method understands
  std::vector<std::vector<std::shared_ptr<psi::LinearTransform<psi::Vector<psi::t_complex>>>>> lt(band_number);
  for(int f=0; f<band_number; f++){
	  lt[f] =  std::vector<std::shared_ptr<psi::LinearTransform<psi::Vector<psi::t_complex>>>>(1);
  }

  for(psi::t_uint f=0; f<band_number; ++f){
    lt[f].emplace_back(std::make_shared<psi::LinearTransform<psi::Vector<psi::t_complex>>>(psi::linear_transform(A.cast<psi::t_complex>())));
  }

  // instanciate the power method
  auto const pm = psi::algorithm::PowerMethodWideband<psi::t_complex>().tolerance(1e-12);
  // call it
  auto const result = pm.AtA(lt, psi::Matrix<psi::t_complex>::Ones(N,N));

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
