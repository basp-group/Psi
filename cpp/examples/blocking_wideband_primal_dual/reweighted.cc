#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include <Eigen/Eigenvalues>
#include <string> // for std::to_string

#include <psi/primal_dual_wideband_blocking.h>
#include <psi/logging.h>
#include <psi/maths.h>
#include <psi/positive_quadrant.h>
#include <psi/relative_variation.h>
#include <psi/reweighted_wideband.h>
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

	const psi::t_int frequencies = 3;
	const psi::t_int block_number = 1;

	PSI_MEDIUM_LOG("Initializing sensing operator");
	psi::t_uint nmeasure = 0.33 * image.size();
	std::vector<std::vector<std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>>>> Phi(frequencies);
	for(int f=0; f<frequencies; f++){
		Phi[f] = std::vector<std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>>>(block_number);
	}
	std::vector<std::vector<psi::LinearTransform<psi::Vector<psi::t_complex>>>> Phitemp(frequencies);
	for(int f=0; f<frequencies; f++){
		Phitemp[f] = std::vector<psi::LinearTransform<psi::Vector<psi::t_complex>>>(block_number, psi::linear_transform_identity<psi::t_complex>());
	}
	for(int f=0; f<frequencies; f++){
		//! TODO This is likely wrong as tempphi will go out of scope but the shared ptr will still be used. Fix this.
		psi::LinearTransform<psi::Vector<psi::t_complex>> tempphi = psi::linear_transform<psi::t_complex>(psi::Sampling(image.size(), nmeasure, mersenne));
		Phi[f][0] = std::make_shared<psi::LinearTransform<psi::Vector<psi::t_complex>>>(tempphi);
		Phitemp[f][0] = psi::linear_transform<psi::t_complex>(psi::Sampling(image.size(), nmeasure, mersenne));
	}

	PSI_MEDIUM_LOG("Computing primal-dual parameters");
	auto const snr = 30.0;
	std::vector<std::vector<psi::Vector<psi::t_complex>>> target(frequencies);
	for(int f=0; f<frequencies; f++){
		target[f] = std::vector<psi::Vector<psi::t_complex>>(1);
	}
	psi::Vector<psi::Vector<psi::t_real>> l2ball_epsilon(frequencies);
	for(int f=0; f<frequencies; f++){
		l2ball_epsilon[f] = psi::Vector<psi::t_real>(1);
	}
	for(int f=0; f<frequencies; f++){
		target[f][0] = Phitemp[f][0] * Vector::Map(image.data(), image.size());
		auto const sigma = target[f][0].stableNorm() / std::sqrt(target[f][0].size()) * std::pow(10.0, -(snr / 20.0));
		l2ball_epsilon[f][0] = std::sqrt(nmeasure + 2 * std::sqrt(target[f][0].size())) * sigma;

		std::normal_distribution<> gaussian_dist(0, sigma);
		for(psi::t_int i = 0; i < target[f].size(); i++)
			target[f][0](i) = target[f][0](i) + gaussian_dist(mersenne);
	}

	// Write dirty image to file
	if(output != "none") {
		for(int f=0; f<frequencies; f++){
			Vector const dirty = Phi[f][0]->adjoint() * target[f][0];
			Image dirty_image = Image::Map(dirty.data(), image.rows(), image.cols());
			psi::utilities::write_tiff(dirty_image.real(), output + "_dirty_" + std::to_string(f) + ".tiff");
		}
	}

	PSI_MEDIUM_LOG("Initializing wavelets");
	psi::wavelets::SARA const sara{ std::make_tuple(std::string{"DB3"}, 1u),
		std::make_tuple(std::string{"DB1"}, 2u),
		std::make_tuple(std::string{"DB1"}, 3u),
		std::make_tuple(std::string{"DB1"}, 4u)};
	auto nlevels = sara.size();
	auto const min_delta = 1e-5;

	// Algorithm parameters
	auto const tau = 0.33;   // 3 terms involved, 0.99/3.
	PSI_MEDIUM_LOG("tau is {} ", tau);
	auto const kappa1 = 1.0; // identity operator within the nuclear norm
	PSI_MEDIUM_LOG("kappa1 is {} ", kappa1);
	auto kappa2 = 1.0;       // Daubechies wavelets are orthogonal, and only the identity is considered in addition to these dictionaries -> operator norm equal to 1
	PSI_MEDIUM_LOG("kappa2 is {} ", kappa2);
	auto kappa3 = 1.;    // inverse of the norm of the full measurement operator Psi (single value), to be loaded
	auto const mu = 1e-2; // hyperparameter related to the l21 norm

	auto Decomp = psi::mpi::Decomposition(false);

	std::vector<psi::t_int> time_blocks = std::vector<psi::t_int>();
	time_blocks[0] = frequencies;

	std::vector<psi::t_int> wavelet_levels = std::vector<psi::t_int>(frequencies);
	wavelet_levels[0] = nlevels;

	std::vector<std::vector<psi::t_int>> sub_blocks = std::vector<std::vector<psi::t_int>>(frequencies);
	sub_blocks[0] = std::vector<psi::t_int>(frequencies);
	sub_blocks[0][0] = 0;

	Decomp.decompose_primal_dual(false, false, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);

	std::vector<psi::LinearTransform<Vector>> Psi;
	for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
		Psi.push_back(psi::linear_transform<psi::t_complex>(sara, image.rows(), image.cols()));
	}

	psi::LinearTransform<psi::Vector<psi::t_complex>> Psi_Root(psi::linear_transform_identity<Scalar>());
	if(Decomp.my_number_of_root_wavelets()>0){
		Psi_Root = psi::linear_transform<psi::t_complex>(psi::wavelets::distribute_sara(sara, Decomp.my_lower_root_wavelet(), Decomp.my_number_of_root_wavelets(), Decomp.global_number_of_root_wavelets()), image.rows(), image.cols());
	}

	std::vector<std::vector<psi::Vector<psi::t_real>>> Ui(Decomp.my_number_of_frequencies());
	for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
		Ui[f] = std::vector<psi::Vector<psi::t_real>>(Decomp.my_frequencies()[f].number_of_time_blocks);
	}

	std::vector<psi::t_uint> levels(Decomp.my_number_of_frequencies());
	for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
		levels[f] = Decomp.my_frequencies()[f].number_of_wavelets;
	}

	// Instantiate algorithm
	PSI_HIGH_LOG("Creating wideband blocking primal-dual functor");
	auto widebandblocking_pd
	= psi::algorithm::PrimalDualWidebandBlocking<psi::t_complex>(target, image.size(), l2ball_epsilon, Phi, Ui)
	.itermax(10)
	.tau(tau)
	.kappa1(kappa1)
	.kappa2(kappa2)
	.kappa3(kappa3)
	.mu(mu)
	.levels(levels)
	.global_levels(nlevels)
	.l21_proximal_weights(psi::Vector<psi::t_real>::Ones(image.size()*nlevels))
	.nuclear_proximal_weights(psi::Vector<psi::t_real>::Ones(target.size()))
	.positivity_constraint(true)
	.relative_variation(5e-4)
	.residual_convergence(1.001)
	.update_epsilon(false)
	.Psi(Psi)
	.Psi_Root(Psi_Root)
	.preconditioning(false)
	.decomp(Decomp);

	PSI_HIGH_LOG("Creating wideband blocking re-weighting-scheme functor");
	auto reweighted = psi::algorithm::reweighted(widebandblocking_pd)
	.itermax(3)
	.min_delta(min_delta)
	.is_converged(psi::RelativeVariation<psi::t_complex>(1e-5));

	PSI_HIGH_LOG("Starting re-weighted wideband blocking primal dual from psi library");
	auto diagnostic = reweighted();

	PSI_MEDIUM_LOG("Wideband blocking primal-dual returned {}", diagnostic.good);

	PSI_MEDIUM_LOG("Wideband blocking primal-dual converged in {} iterations", diagnostic.niters);

	if(output != "none"){
		for(int f=0; f<frequencies; f++){
			Matrix main_image = Matrix::Map(diagnostic.algo.x.col(f).data(), image.rows(), image.cols());
			psi::utilities::write_tiff(main_image.real(), output + std::to_string(f) + ".tiff");
		}
	}

	return 0;
}
