#include <iostream>
#include <numeric>
#include <cmath>
#include <complex>
#include <catch2/catch.hpp>

#include "psi/config.h"
#include "psi/io.h"

using namespace psi;

typedef psi::t_real Scalar;
typedef psi::Vector<Scalar> t_Vector;
typedef psi::Matrix<Scalar> t_Matrix;

auto const out_size = 100;
auto const other_size = 5;
auto const epsilons = 3.75;
auto const weights = 7;
auto const l21weights = 8.5;
auto const nucweights = 90.3;

#ifdef PSI_MPI

TEST_CASE("Checkpointing serial") {

	int rank, size;

	auto const world = mpi::Communicator::World();
	bool parallel = true;
	auto Decomp = mpi::Decomposition(parallel, world);


	if(world.rank() == 0){
		auto check = psi::io::IO<Scalar>();
		std::string filename = "restart.dat";

		t_Vector out(100);
		Vector<t_real> total_epsilons(other_size);
		Vector<t_real> l1_weights(other_size);

		t_Vector out2(100);
		Vector<t_real> total_epsilons2(other_size);
		Vector<t_real> l1_weights2(other_size);

		t_Vector out3;
		Vector<t_real> total_epsilons3;
		Vector<t_real> l1_weights3;

		t_real kappa = 5.4;
		t_real kappa_2;
		t_real kappa_3;

		t_real sigma2 = 94.26;
		t_real sigma2_2;
		t_real sigma2_3;

		t_real delta = 0.44;
		t_real delta_2;
		t_real delta_3;

		int current_reweighting_iter = 3;
		int current_reweighting_iter_2;
		int current_reweighting_iter_3;

		for(int i=0; i<out.size(); i++){
			out[i]  = i;
		}

		for(int i=0; i<total_epsilons.size(); i++){
			total_epsilons[i]  = epsilons;
		}

		for(int i=0; i<l1_weights.size(); i++){
			l1_weights[i]  = weights;
		}


		SECTION("Checkpoint time blocking") {
			psi::io::IOStatus check_status = check.checkpoint_time_blocking(filename, out, total_epsilons, l1_weights, kappa, sigma2, delta, current_reweighting_iter);
			CHECK(check_status == psi::io::IOStatus::Success);
		}

		SECTION("Checkpoint and restore time blocking") {
			psi::io::IOStatus check_status = check.checkpoint_time_blocking(filename, out, total_epsilons, l1_weights, kappa, sigma2, delta, current_reweighting_iter);
			CHECK(check_status == psi::io::IOStatus::Success);
			psi::io::IOStatus restore_status = check.restore_time_blocking(filename, out2, total_epsilons2, l1_weights2, kappa_2, sigma2_2, out.size(), delta_2, current_reweighting_iter_2);
			CHECK(restore_status == psi::io::IOStatus::Success);
			CHECK(kappa == kappa_2);
			CHECK(sigma2 == sigma2_2);
			CHECK(delta == delta_2);
			CHECK(current_reweighting_iter == current_reweighting_iter_2);
			for(int i=0; i<out.size(); i++){
				CHECK(out[i] == out2[i]);
			}
			for(int i=0; i<total_epsilons.size(); i++){
				CHECK(total_epsilons[i] == total_epsilons2[i]);
			}
			for(int i=0; i<l1_weights.size(); i++){
				CHECK(l1_weights[i] == l1_weights2[i]);
			}
		}

		SECTION("Checkpoint and restore with array creation") {
			psi::io::IOStatus check_status = check.checkpoint_time_blocking(filename, out, total_epsilons, l1_weights, kappa, sigma2, delta, current_reweighting_iter);
			CHECK(check_status == psi::io::IOStatus::Success);
			psi::io::IOStatus restore_status = check.restore_time_blocking(filename, out3, total_epsilons3, l1_weights3, kappa_3, sigma2_3, out.size(), delta_3, current_reweighting_iter_3);
			CHECK(restore_status == psi::io::IOStatus::Success);
			CHECK(kappa == kappa_3);
			CHECK(sigma2 == sigma2_3);
			CHECK(delta == delta_3);
			CHECK(current_reweighting_iter == current_reweighting_iter_3);
			CHECK(out3.size() == out.size());
			CHECK(total_epsilons3.size() == total_epsilons.size());
			CHECK(l1_weights3.size() == l1_weights.size());

			for(int i=0; i<out.size(); i++){
				CHECK(out[i] == out3[i]);
			}
			for(int i=0; i<total_epsilons.size(); i++){
				CHECK(total_epsilons[i] == total_epsilons3[i]);
			}
			for(int i=0; i<l1_weights.size(); i++){
				CHECK(l1_weights[i] == l1_weights3[i]);
			}
		}
	}

}

TEST_CASE("Checkpointing serial wideband") {

	int rank, size;

	auto const world = mpi::Communicator::World();
	bool parallel = true;
	auto Decomp = mpi::Decomposition(parallel, world);


	if(world.rank() == 0){
		auto check = psi::io::IO<Scalar>();
		std::string filename = "restart.dat";

		int frequencies = 4;

		t_Matrix out(100,frequencies);
		Vector<Vector<t_real>> total_epsilons(frequencies);
		for(int i=0; i<frequencies; i++){
			total_epsilons(i) = Vector<t_real>(other_size);
		}
		Vector<t_real> l21_weights(other_size);
		Vector<t_real> nuclear_weights(other_size);

		t_Matrix out2(100,frequencies);
		Vector<Vector<t_real>> total_epsilons2(frequencies);
		for(int i=0; i<frequencies; i++){
			total_epsilons2(i) = Vector<t_real>(other_size);
		}
		Vector<t_real> l21_weights2(other_size);
		Vector<t_real> nuclear_weights2(other_size);

		t_Matrix out3;
		Vector<Vector<t_real>> total_epsilons3;
		Vector<t_real> l21_weights3;
		Vector<t_real> nuclear_weights3;

		t_real kappa1 = 5.4;
		t_real kappa1_2;
		t_real kappa1_3;

		t_real kappa2 = 195.3;
		t_real kappa2_2;
		t_real kappa2_3;

		t_real kappa3 = 6.7;
		t_real kappa3_2;
		t_real kappa3_3;

		t_real delta = 0.44;
		t_real delta_2;
		t_real delta_3;

		int current_reweighting_iter = 3;
		int current_reweighting_iter_2;
		int current_reweighting_iter_3;

		for(int i=0; i<out.rows(); i++){
			for(int j=0; j<out.cols(); j++){
				out(i,j)  = j;
			}
		}

		for(int i=0; i<total_epsilons.size(); i++){
			for(int j=0; j<total_epsilons(i).size(); j++){
				total_epsilons[i][j]  = epsilons;
			}
		}

		for(int i=0; i<l21_weights.size(); i++){
			l21_weights[i]  = l21weights;
			nuclear_weights[i] = nucweights;
		}


		SECTION("Checkpoint time blocking") {
			psi::io::IOStatus check_status = check.checkpoint_wideband(filename, out, total_epsilons, l21_weights, nuclear_weights, kappa1, kappa2, kappa3, delta, current_reweighting_iter);
			CHECK(check_status == psi::io::IOStatus::Success);
		}

		SECTION("Checkpoint and restore time blocking") {
			psi::io::IOStatus check_status = check.checkpoint_wideband(filename, out, total_epsilons, l21_weights, nuclear_weights, kappa1, kappa2, kappa3, delta, current_reweighting_iter);
			CHECK(check_status == psi::io::IOStatus::Success);
			psi::io::IOStatus restore_status = check.restore_wideband(filename, out2, total_epsilons2, l21_weights2, nuclear_weights2, kappa1_2, kappa2_2, kappa3_2, out.cols(), out.rows(), delta_2, current_reweighting_iter_2);
			CHECK(restore_status == psi::io::IOStatus::Success);
			CHECK(kappa1 == kappa1_2);
			CHECK(kappa2 == kappa2_2);
			CHECK(kappa3 == kappa3_2);
			CHECK(delta == delta_2);
			CHECK(current_reweighting_iter == current_reweighting_iter_2);
			for(int i=0; i<out.rows(); i++){
				for(int j=0; j<out.cols(); j++){
					CHECK(out(i,j) == out2(i,j));
				}
			}
			for(int i=0; i<total_epsilons.size(); i++){
				for(int j=0; j<total_epsilons(i).size(); j++){
					CHECK(total_epsilons[i][j] == total_epsilons2[i][j]);
				}
			}
			for(int i=0; i<l21_weights.size(); i++){
				CHECK(l21_weights[i] == l21_weights2[i]);
			}
			for(int i=0; i<nuclear_weights.size(); i++){
				CHECK(nuclear_weights[i] == nuclear_weights2[i]);
			}
		}

		SECTION("Checkpoint and restore with array creation") {
			psi::io::IOStatus check_status = check.checkpoint_wideband(filename, out, total_epsilons, l21_weights, nuclear_weights, kappa1, kappa2, kappa3, delta, current_reweighting_iter);
			CHECK(check_status == psi::io::IOStatus::Success);
			psi::io::IOStatus restore_status = check.restore_wideband(filename, out3, total_epsilons3, l21_weights3, nuclear_weights3, kappa1_3, kappa2_3, kappa3_3, out.cols(), out.rows(), delta_3, current_reweighting_iter_3);
			CHECK(restore_status == psi::io::IOStatus::Success);
			CHECK(kappa1 == kappa1_3);
			CHECK(kappa2 == kappa2_3);
			CHECK(kappa3 == kappa3_3);
			CHECK(delta == delta_3);
			CHECK(current_reweighting_iter == current_reweighting_iter_3);
			CHECK(out3.size() == out.size());
			CHECK(out3.rows() == out.rows());
			CHECK(out3.cols() == out3.cols());
			CHECK(total_epsilons3.size() == total_epsilons.size());
			for(int i=0; i<total_epsilons.size(); i++){
				CHECK(total_epsilons3(i).size() == total_epsilons(i).size());
			}
			CHECK(l21_weights3.size() == l21_weights.size());
			CHECK(nuclear_weights3.size() == nuclear_weights.size());
			for(int i=0; i<out.rows(); i++){
				for(int j=0; j<out.cols(); j++){
					CHECK(out(i,j) == out3(i,j));
				}
			}
			for(int i=0; i<total_epsilons.size(); i++){
				for(int j=0; j<total_epsilons(i).size(); j++){
					CHECK(total_epsilons[i][j] == total_epsilons3[i][j]);
				}
			}
			for(int i=0; i<l21_weights.size(); i++){
				CHECK(l21_weights[i] == l21_weights3[i]);
			}
			for(int i=0; i<nuclear_weights.size(); i++){
				CHECK(nuclear_weights[i] == nuclear_weights3[i]);
			}

		}
	}

}

TEST_CASE("Checkpointing in parallel") {

	int rank, size;

	auto const world = mpi::Communicator::World();
	bool parallel = true;
	auto Decomp = mpi::Decomposition(parallel, world);

	t_int nlevels = 1;
	t_int blocks_per_process = 2;
	t_int n_blocks = world.size()*blocks_per_process;
	t_int frequencies = 1;


	std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
	wavelet_levels[0] = nlevels;

	std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
	time_blocks[0] = n_blocks;

	std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
	sub_blocks[0] = std::vector<t_int>(1);
	sub_blocks[0][0] = 0;

	Decomp.decompose_primal_dual(false, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks, true);


	auto check = psi::io::IO<Scalar>();
	std::string filename = "restart.dat";

	t_int image_size = 100;

	t_Vector out(image_size);
	Vector<t_real> local_epsilons(Decomp.my_frequencies()[0].number_of_time_blocks);
	Vector<t_real> l1_weights(Decomp.my_frequencies()[0].number_of_wavelets*image_size);

	t_Vector out2(image_size);
	Vector<t_real> local_epsilons2(Decomp.my_frequencies()[0].number_of_time_blocks);
	Vector<t_real> l1_weights2(Decomp.my_frequencies()[0].number_of_wavelets*image_size);

	t_real kappa = 5.4;
	t_real kappa_2;

	t_real sigma2 = 853.3;
	t_real sigma2_2;

	t_real delta = 0.4;
	t_real delta_2;

	int current_reweighting_iter = 6;
	int current_reweighting_iter_2;

	for(int i=0; i<out.size(); i++){
		out[i]  = i;
	}

	for(int i=0; i<local_epsilons.size(); i++){
		local_epsilons[i]  = epsilons;
	}

	for(int i=0; i<l1_weights.size(); i++){
		l1_weights[i]  = weights;
	}

	SECTION("Checkpoint with collect") {
		psi::io::IOStatus check_status = check.checkpoint_time_blocking_with_collect(Decomp, filename, out, local_epsilons, l1_weights, kappa, sigma2, image_size, delta, current_reweighting_iter);
		CHECK(check_status == psi::io::IOStatus::Success);
	}

	SECTION("Checkpoint and restore") {
		psi::io::IOStatus check_status = check.checkpoint_time_blocking_with_collect(Decomp, filename, out, local_epsilons, l1_weights, kappa, sigma2, image_size, delta, current_reweighting_iter);
		CHECK(check_status == psi::io::IOStatus::Success);
		psi::io::IOStatus restore_status = check.restore_time_blocking_with_distribute(Decomp, filename, out2, local_epsilons2, l1_weights2, kappa_2, sigma2_2, image_size, delta_2, current_reweighting_iter_2);
		CHECK(restore_status == psi::io::IOStatus::Success);
		CHECK(kappa == kappa_2);
		CHECK(sigma2 == sigma2_2);
		CHECK(delta == delta_2);
		CHECK(current_reweighting_iter == current_reweighting_iter_2);
		if(Decomp.global_comm().rank() == 0){
			for(int i=0; i<out.size(); i++){
				CHECK(out[i] == out2[i]);
			}
		}
		CHECK(local_epsilons.size() == local_epsilons2.size());
		for(int i=0; i<local_epsilons.size(); i++){
			CHECK(local_epsilons[i] == local_epsilons2[i]);
		}
		CHECK(l1_weights.size() == l1_weights2.size());
		for(int i=0; i<l1_weights.size(); i++){
			CHECK(l1_weights[i] == l1_weights2[i]);
		}
	}

}

TEST_CASE("Checkpointing wideband in parallel") {

	int rank, size;

	auto const world = mpi::Communicator::World();
	bool parallel = true;
	auto Decomp = mpi::Decomposition(parallel, world);

	t_int nlevels = 3;
	t_int n_blocks = 2;
	t_int frequencies = world.size()*2;

	std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
	for(int i=0; i<frequencies; i++){
		wavelet_levels[i] = nlevels;
	}

	std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
	for(int i=0; i<frequencies; i++){
		time_blocks[i] = n_blocks;
	}

	std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(frequencies);
	for(int i=0; i<frequencies; i++){
		sub_blocks[i] = std::vector<t_int>(n_blocks);
		for(int j=0; j<n_blocks; j++){
			sub_blocks[i][j] = 1;
		}
	}

	Decomp.decompose_primal_dual(true, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);


	auto check = psi::io::IO<Scalar>();
	std::string filename = "restart.dat";

	t_int image_size = 100;

	t_Matrix out(image_size,frequencies);
	Vector<Vector<t_real>> local_epsilons(Decomp.my_number_of_frequencies());
	for(int i=0; i<Decomp.my_number_of_frequencies();i++){
		local_epsilons(i) = Vector<t_real>(Decomp.my_frequencies()[i].number_of_time_blocks);
	}
	Vector<t_real> l21_weights(Decomp.my_frequencies()[0].number_of_wavelets*image_size);
	Vector<t_real> nuclear_weights(image_size);

	t_Matrix out2(image_size,frequencies);
	Vector<Vector<t_real>> local_epsilons2(Decomp.my_number_of_frequencies());
	for(int i=0; i<Decomp.my_number_of_frequencies();i++){
		local_epsilons2(i) = Vector<t_real>(Decomp.my_frequencies()[i].number_of_time_blocks);
	}
	Vector<t_real> l21_weights2(Decomp.my_frequencies()[0].number_of_wavelets*image_size);
	Vector<t_real> nuclear_weights2(image_size);

	t_real kappa1 = 5.4;
	t_real kappa1_2;

	t_real kappa2 = 42;
	t_real kappa2_2;

	t_real kappa3 = 60.4;
	t_real kappa3_2;

	t_real delta = 0.4;
	t_real delta_2;

	int current_reweighting_iter = 6;
	int current_reweighting_iter_2;

	for(int i=0; i<out.rows(); i++){
		for(int j=0; j<out.cols(); j++){
			out(i,j)  = i;
		}
	}

	for(int i=0; i<local_epsilons.size(); i++){
		for(int j=0; j<local_epsilons(i).size(); j++){
			local_epsilons[i][j]  = epsilons;
		}
	}

	for(int i=0; i<l21_weights.size(); i++){
		l21_weights[i]  = l21weights;
	}

	for(int i=0; i<nuclear_weights.size(); i++){
		nuclear_weights[i]  = nucweights;
	}

	SECTION("Checkpoint with collect") {
		psi::io::IOStatus check_status = check.checkpoint_wideband_with_collect(Decomp, filename, out, local_epsilons, l21_weights, nuclear_weights, kappa1, kappa2, kappa3, image_size, delta, current_reweighting_iter);
		CHECK(check_status == psi::io::IOStatus::Success);
	}

	SECTION("Checkpoint and restore") {
		psi::io::IOStatus check_status = check.checkpoint_wideband_with_collect(Decomp, filename, out, local_epsilons, l21_weights, nuclear_weights, kappa1, kappa2, kappa3, image_size, delta, current_reweighting_iter);
		CHECK(check_status == psi::io::IOStatus::Success);
		psi::io::IOStatus restore_status = check.restore_wideband_with_distribute(Decomp, filename, out2, local_epsilons2, l21_weights2, nuclear_weights2, kappa1_2, kappa2_2, kappa3_2, out.cols(), image_size, delta_2, current_reweighting_iter_2);
		CHECK(restore_status == psi::io::IOStatus::Success);
		CHECK(kappa1 == kappa1_2);
		CHECK(kappa2 == kappa2_2);
		CHECK(kappa3 == kappa3_2);
		CHECK(delta == delta_2);
		CHECK(current_reweighting_iter == current_reweighting_iter_2);
		if(Decomp.global_comm().rank() == 0){
			for(int i=0; i<out.rows(); i++){
				for(int j=0; j<out.cols(); j++){
					CHECK(out(i,j) == out2(i,j));
				}
			}
			CHECK(l21_weights.size() == l21_weights2.size());
			for(int i=0; i<l21_weights.size(); i++){
				CHECK(l21_weights[i] == l21_weights2[i]);
			}
			CHECK(nuclear_weights.size() == nuclear_weights2.size());
			for(int i=0; i<nuclear_weights.size(); i++){
				CHECK(nuclear_weights[i] == nuclear_weights2[i]);
			}
		}
		CHECK(local_epsilons.size() == local_epsilons2.size());
		for(int i=0; i<local_epsilons.size(); i++){
			CHECK(local_epsilons(i).size() == local_epsilons2(i).size());
			for(int j=0; j<local_epsilons(i).size(); j++){
				CHECK(local_epsilons(i)[j] == local_epsilons2(i)[j]);
			}
		}
	}

}

#endif


