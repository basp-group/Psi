#include <psi/l1_proximal.h>
#include <psi/logging.h>
#include <psi/types.h>

int main(int, char const **) {
  psi::logging::initialize();

  typedef psi::t_complex Scalar;
  typedef psi::real_type<Scalar>::type Real;
  auto const input = psi::Vector<Scalar>::Random(10).eval();
  auto const Psi = psi::Matrix<Scalar>::Random(input.size(), input.size() * 10).eval();
  psi::Vector<Real> const weights
      = psi::Vector<Scalar>::Random(Psi.cols()).normalized().array().abs();

  auto const l1 = psi::proximal::L1<Scalar>()
                      .tolerance(1e-12)
                      .itermax(100)
                      .fista_mixing(true)
                      .positivity_constraint(true)
                      .nu(1)
                      .Psi(Psi)
                      .weights(weights);

  // gamma should be sufficiently small. Or is it nu should not be 1?
  // In any case, this seems to work.
  Real const gamma = 1e-2 / Psi.array().abs().sum();
  auto const result = l1(gamma, input);

  if(not result.good)
    PSI_THROW("Did not converge");

  // Check the proximal is a minimum in any allowed direction (positivity constraint)
  Real const eps = 1e-4;
  for(size_t i(0); i < 10; ++i) {
    psi::Vector<Scalar> const dir = psi::Vector<Scalar>::Random(input.size()).normalized() * eps;
    psi::Vector<Scalar> const position = psi::positive_quadrant(result.proximal + dir);
    Real const dobj = l1.objective(input, position, gamma);
    // Fuzzy logic
    if(dobj < result.objective - 1e-8)
      PSI_THROW("This is not the minimum we are looking for: ") << dobj << " <~ "
                                                                 << result.objective;
  }

  return 0;
}
