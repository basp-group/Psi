#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include <Eigen/Eigenvalues>

#include <psi/preconditioned_primal_dual.h>
#include <psi/logging.h>
#include <psi/maths.h>
#include <psi/positive_quadrant.h>
#include <psi/relative_variation.h>
#include <psi/reweighted.h>
#include <psi/sampling.h>
#include <psi/types.h>
#include <psi/utilities.h>
#include <psi/wavelets.h>
#include <psi/wavelets/sara.h>
#include <psi/power_method.h>

// This header is not part of the installed psi interface
// It is only present in tests
#include <tools_for_tests/directories.h>
#include <tools_for_tests/tiffwrappers.h>

// \min_{x} ||\Psi^Tx||_1 \quad \mbox{s.t.} \quad ||y - Ax||_2 < \epsilon and x \geq 0
int main(int argc, char const **argv) {
  // Some typedefs for simplicity
  typedef double Scalar;
  // Column vector - linear algebra - A * x is a matrix-vector multiplication
  // type expected by ProximalADMM
  typedef psi::Vector<Scalar> Vector;
  // Matrix - linear algebra - A * x is a matrix-vector multiplication
  // type expected by ProximalADMM
  typedef psi::Matrix<Scalar> Matrix;
  // Image - 2D array - A * x is a coefficient-wise multiplication
  // Type expected by wavelets and image write/read functions
  typedef psi::Image<Scalar> Image;

  std::string const input = argc >= 2 ? argv[1] : "cameraman256";
  std::string const output = argc == 3 ? argv[2] : "none";
  if(argc > 3) {
    std::cout << "Usage:\n"
                 "$ "
              << argv[0] << " [input [output]]\n\n"
                            "- input: path to the image to clean (or name of standard PSI image)\n"
                            "- output: filename pattern for output image\n";
    exit(0);
  }
  // Set up random numbers for C and C++
  auto const seed = std::time(0);
  std::srand((unsigned int)seed);
  std::mt19937 mersenne(std::time(0));

  // Initializes and sets logger (if compiled with logging)
  // See set_level function for levels.
  psi::logging::initialize();

  PSI_MEDIUM_LOG("Read input file {}", input);
  Image const image = psi::notinstalled::read_standard_tiff(input);

  PSI_MEDIUM_LOG("Initializing sensing operator");
  psi::t_uint nmeasure = 0.33 * image.size();
  auto sampling
      = psi::linear_transform<Scalar>(psi::Sampling(image.size(), nmeasure, mersenne));

  PSI_MEDIUM_LOG("Initializing wavelets");
  psi::wavelets::SARA const sara{std::make_tuple(std::string{"DB3"}, 1u),
                                  std::make_tuple(std::string{"DB1"}, 2u),
                                  std::make_tuple(std::string{"DB1"}, 3u),
                                  std::make_tuple(std::string{"DB1"}, 4u)};

  auto const nlevels = sara.size();

  auto psi = psi::linear_transform<Scalar>(sara, image.rows(), image.cols());

  PSI_MEDIUM_LOG("Computing preconditioned preconditioned primal-dual parameters");
  Vector const y0 = sampling * Vector::Map(image.data(), image.size());
  Vector const Ui = Vector::Ones(y0.size());
  auto const snr = 30.0;
  auto const sigma = y0.stableNorm() / std::sqrt(y0.size()) * std::pow(10.0, -(snr / 20.0));
  auto const epsilon = std::sqrt(nmeasure + 2 * std::sqrt(y0.size())) * sigma;

  PSI_MEDIUM_LOG("Create dirty vector");
  std::normal_distribution<> gaussian_dist(0, sigma);
  Vector y(y0.size());
  for(psi::t_int i = 0; i < y0.size(); i++)
    y(i) = y0(i) + gaussian_dist(mersenne);
  // Write dirty imagte to file
  if(output != "none") {
    Vector const dirty = sampling.adjoint() * y;
    psi::utilities::write_tiff(Matrix::Map(dirty.data(), image.rows(), image.cols()),
                                "dirty_" + output + ".tiff");
  }

  //  Vector rand = Vector::Random(image.size());
  PSI_HIGH_LOG("Setting up power method to calculate sigma values");
  Eigen::EigenSolver<Matrix> es;
  PSI_HIGH_LOG("Setting up matrix A");

  Vector rand = Vector::Random(image.size()*nlevels);

  auto const pm = psi::algorithm::PowerMethod<psi::t_real>().tolerance(1e-12);

  auto const tau = 0.49;
  auto const kappa = 0.1;

  // sigma1 should be 1 (or number of wavelet operators being used)
  PSI_HIGH_LOG("Calculating sigma1");
  auto const nu1data = pm.AtA(psi, rand);
  auto const nu1 = nu1data.magnitude;
  auto sigma1 = 1e0 / nu1;

  rand = Vector::Random(image.size());

  // sigma2 should something like 1x10-10
  PSI_HIGH_LOG("Calculating sigma2");
  auto const nu2data = pm.AtA(sampling, rand);
  auto const nu2 = nu2data.magnitude;
  auto sigma2 = 1e0 / nu2;


    PSI_HIGH_LOG("Creating preconditioned primal-dual Functor");
  auto ppd = psi::algorithm::PreconditionedPrimalDual<Scalar>(y)
                         .Ui(Ui)
                         .itermax(500)
                         .tau(tau)
                         .kappa(kappa)
                         .sigma1(sigma1)
                         .sigma2(sigma2)
                         .l2ball_epsilon(epsilon)
                         .levels(nlevels)
                         .nu(nu2)
                         .Psi(psi)
                         .Phi(sampling)
                         .relative_variation(5e-4)
                         .residual_convergence(epsilon * 1.001)
                         .positivity_constraint(true);


  PSI_MEDIUM_LOG("Creating the reweighted algorithm");
  // This follows the reweighted algorithm for SDMM
  auto const min_delta = sigma * std::sqrt(y.size()) / std::sqrt(8 * image.size());
  auto reweighted
      = psi::algorithm::reweighted(ppd).itermax(5).min_delta(min_delta).is_converged(
          psi::RelativeVariation<Scalar>(1e-3));

  PSI_MEDIUM_LOG("Starting preconditioned primal dual");
  // Here, we default to (Φ^Ty/ν, ΦΦ^Ty/ν - y)
  auto const diagnostic = reweighted();
  PSI_MEDIUM_LOG("preconditioned primal-dual returned {}", diagnostic.good);

  PSI_MEDIUM_LOG("PSI-preconditioned primal-dual converged in {} iterations", diagnostic.niters);
  if(output != "none")
    psi::utilities::write_tiff(Matrix::Map(diagnostic.algo.x.data(), image.rows(), image.cols()),
                                output + ".tiff");

  return 0;
}
