#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include <Eigen/Eigenvalues>
#include <string> // for std::to_string

#include <psi/primal_dual_time_blocking.h>
#include <psi/logging.h>
#include <psi/maths.h>
#include <psi/positive_quadrant.h>
#include <psi/relative_variation.h>
#include <psi/reweighted_time_blocking.h>
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
	typedef std::complex<double> Scalar;
	// Column vector - linear algebra - A * x is a matrix-vector multiplication
	typedef psi::Vector<Scalar> Vector;
	// Matrix - linear algebra - A * x is a matrix-vector multiplication
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
	//psi::logging::set_level("debug");

	PSI_MEDIUM_LOG("Read input file {}", input);
	Image const image = psi::notinstalled::read_standard_tiff(input);

	const psi::t_int block_number = 3;

	PSI_MEDIUM_LOG("Initializing sensing operator");
	psi::t_uint nmeasure = 0.33 * image.size();
	std::vector<std::shared_ptr<psi::LinearTransform<psi::Vector<psi::t_complex>>>> Phi(block_number);
	std::vector<psi::LinearTransform<psi::Vector<psi::t_complex>>> Phitemp(block_number, psi::linear_transform_identity<psi::t_complex>());
	for(int l = 0; l < block_number; ++l){
		//! TODO This is likely wrong as tempphi will go out of scope but the shared ptr will still be used. Fix this.
		psi::LinearTransform<psi::Vector<psi::t_complex>> tempphi = psi::linear_transform<psi::t_complex>(psi::Sampling(image.size(), nmeasure, mersenne));
		Phi[l] = std::make_shared<psi::LinearTransform<psi::Vector<psi::t_complex>>>(tempphi);
		Phitemp[l] = psi::linear_transform<psi::t_complex>(psi::Sampling(image.size(), nmeasure, mersenne));
	}

	PSI_MEDIUM_LOG("Computing primal-dual parameters");
	auto const snr = 30.0;
	std::vector<psi::Vector<psi::t_complex>> target(block_number);
	psi::Vector<psi::t_real> l2ball_epsilon(block_number);
	for(int l = 0; l < block_number; ++l){
		target[l] = Phitemp[l] * Vector::Map(image.data(), image.size());
		auto const sigma = target[l].stableNorm() / std::sqrt(target[l].size()) * std::pow(10.0, -(snr / 20.0));
		l2ball_epsilon[l] = std::sqrt(nmeasure + 2 * std::sqrt(target[l].size())) * sigma;

		std::normal_distribution<> gaussian_dist(0, sigma);
		for(psi::t_int i = 0; i < target[l].size(); i++)
			target[l](i) = target[l](i) + gaussian_dist(mersenne);
	}

	PSI_HIGH_LOG("Setting up power method to calculate sigma values");
	Eigen::EigenSolver<Matrix> es;
	PSI_HIGH_LOG("Setting up matrix A");
	PSI_HIGH_LOG("Calculating nu");
	auto const pm = psi::algorithm::PowerMethod<psi::t_complex>().tolerance(1e-12);
	Vector nu(block_number);
	for(int l = 0; l < block_number; ++l){
		Vector rand = Vector::Random(image.size());
		auto const nudata = pm.AtA(Phi[l], rand);
		nu(l) = nudata.magnitude;
	}

	// Write dirty image to file
	if(output != "none") {
		for(int l = 0; l < block_number; ++l){
			Vector const dirty = Phi[l]->adjoint() * target[l];
			Image dirty_image = Image::Map(dirty.data(), image.rows(), image.cols());
			psi::utilities::write_tiff(dirty_image.real(), output + "_dirty_" + std::to_string(l) + ".tiff");
		}
	}

	PSI_MEDIUM_LOG("Initializing wavelets");
	psi::wavelets::SARA const sara{ std::make_tuple(std::string{"DB3"}, 1u),
		std::make_tuple(std::string{"DB1"}, 2u),
		std::make_tuple(std::string{"DB1"}, 3u),
		std::make_tuple(std::string{"DB1"}, 4u)};
	auto nlevels = sara.size();
	auto const Psi
	= psi::linear_transform<psi::t_complex>(sara, image.rows(), image.cols());
	auto const min_delta = 1e-5;

	// Algorithm parameters
	auto const tau = 0.33;   // 3 terms involved, 0.99/3.
	PSI_MEDIUM_LOG("tau is {} ", tau);
	auto const kappa = 1.0;
	PSI_MEDIUM_LOG("kappa is {} ", kappa);

	auto sigma1 = 1.0;
	PSI_HIGH_LOG("sigma1 is {} ", sigma1);

	auto sigma2 =  1.0;
	PSI_HIGH_LOG("sigma2 is {} ", sigma2);

	auto Decomp = psi::mpi::Decomposition(false);

	psi::t_int frequencies = 1;

	std::vector<psi::t_int> time_blocks = std::vector<psi::t_int>(frequencies);
	time_blocks[0] = block_number;

	std::vector<psi::t_int> wavelet_levels = std::vector<psi::t_int>(frequencies);
	wavelet_levels[0] = 4;

	std::vector<std::vector<psi::t_int>> sub_blocks = std::vector<std::vector<psi::t_int>>(frequencies);
	sub_blocks[0] = std::vector<psi::t_int>(frequencies);
	sub_blocks[0][0] = 0;

	Decomp.decompose_primal_dual(false, false, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);

	nlevels = Decomp.my_frequencies()[0].number_of_wavelets;

	std::vector<psi::Vector<psi::t_real>> Ui(Decomp.my_frequencies()[0].number_of_time_blocks);


	// Instantiate algorithm
	PSI_HIGH_LOG("Creating time blocking primal-dual functor");
	auto timeblocking_pd
	= psi::algorithm::PrimalDualTimeBlocking<psi::t_complex>(target, image.size(), l2ball_epsilon, Phi, Ui)
	.itermax(10)
	.tau(tau)
	.sigma1(sigma1)
	.sigma2(sigma2)
	.kappa(kappa)
	.levels(nlevels)
	.l1_proximal_weights(psi::Vector<psi::t_real>::Ones(image.size()*nlevels))
	.positivity_constraint(true)
	.relative_variation(5e-4)
	.residual_convergence(1.001)
	.update_epsilon(false)
	.Psi(Psi)
	.preconditioning(false)
	.decomp(Decomp);

	PSI_HIGH_LOG("Creating time blocking re-weighting-scheme functor");
	auto reweighted = psi::algorithm::reweighted(timeblocking_pd)
	.itermax(3)
	.min_delta(min_delta)
	.is_converged(psi::RelativeVariation<psi::t_complex>(1e-5));

	PSI_HIGH_LOG("Starting re-weighted time blocking primal dual from psi library");
	auto diagnostic = reweighted();

	PSI_MEDIUM_LOG("Time blocking primal-dual returned {}", diagnostic.good);

	PSI_MEDIUM_LOG("Time blocking primal-dual converged in {} iterations", diagnostic.niters);

	if(output != "none"){
		for(int l = 0; l < block_number; ++l){
			Matrix main_image = Matrix::Map(diagnostic.algo.x.data(), image.rows(), image.cols());
			psi::utilities::write_tiff(main_image.real(), output + std::to_string(l) + ".tiff");
		}
	}

	return 0;
}
