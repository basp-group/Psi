#include <iostream>
#include <numeric>
#include <cmath>
#include <complex>
#include <catch2/catch.hpp>
#include <mpi.h>

#include "psi/config.h"
#include "psi/mpi/decomposition.h"

using namespace psi;

#ifdef PSI_MPI
TEST_CASE("Creates a decomposition") {
	int rank, world_size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	auto const world = mpi::Communicator::World();
	bool parallel = true;

	SECTION("Initialisation") {
		auto Decomp = mpi::Decomposition(parallel, world);
		REQUIRE(*(Decomp.global_comm()) == MPI_COMM_WORLD);
	}

	SECTION("Build decomposition") {
		t_int nlevels = 7;
		t_int frequencies = 4;
		t_int blocks_per_frequency = 3;


		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		for(int i=0; i<frequencies; i++){
			wavelet_levels[i] = nlevels;
		}

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		for(int i=0; i<frequencies; i++){
			time_blocks[i] = blocks_per_frequency;
		}

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(frequencies);
		for(int i=0; i<frequencies; i++){
			sub_blocks[i] = std::vector<t_int>(blocks_per_frequency);
			for(int j=0; j<blocks_per_frequency; j++){
				sub_blocks[i][j] = 1;
			}
		}

		t_int upper_process = 4;
		auto Decomp = mpi::Decomposition(parallel, world);
		Decomp.decompose_frequencies(frequencies, 0, upper_process, true, false, true);
		CHECK(Decomp.frequencies().size() == frequencies);
		if(frequencies == 4){
			CHECK(Decomp.frequencies()[0].lower_process == 0);
			CHECK(Decomp.frequencies()[0].upper_process == 1);
			CHECK(Decomp.frequencies()[1].lower_process == 2);
			CHECK(Decomp.frequencies()[1].upper_process == 2);
			CHECK(Decomp.frequencies()[2].lower_process == 3);
			CHECK(Decomp.frequencies()[2].upper_process == 3);
			CHECK(Decomp.frequencies()[3].lower_process == 4);
			CHECK(Decomp.frequencies()[3].upper_process == 4);
		}
		for(int i=0; i<frequencies; i++){
			Decomp.decompose_time_blocks(i, time_blocks[i], true);
			CHECK(Decomp.time_blocks(i).size() == blocks_per_frequency);
		}
		if(blocks_per_frequency == 3 and frequencies == 4){
			CHECK(Decomp.time_blocks(0)[0].global_owner == 0);
			CHECK(Decomp.time_blocks(0)[1].global_owner == 0);
			CHECK(Decomp.time_blocks(0)[2].global_owner == 1);
			CHECK(Decomp.time_blocks(1)[0].global_owner == 2);
			CHECK(Decomp.time_blocks(1)[1].global_owner == 2);
			CHECK(Decomp.time_blocks(1)[2].global_owner == 2);
			CHECK(Decomp.time_blocks(2)[0].global_owner == 3);
			CHECK(Decomp.time_blocks(2)[1].global_owner == 3);
			CHECK(Decomp.time_blocks(2)[2].global_owner == 3);
			CHECK(Decomp.time_blocks(3)[0].global_owner == 4);
			CHECK(Decomp.time_blocks(3)[1].global_owner == 4);
			CHECK(Decomp.time_blocks(3)[2].global_owner == 4);
		}

		upper_process = 3;
		Decomp = mpi::Decomposition(parallel, world);
		Decomp.decompose_frequencies(frequencies, 0, upper_process, false, false, true);
		CHECK(Decomp.frequencies().size() == frequencies);
		if(frequencies == 4){
			CHECK(Decomp.frequencies()[0].lower_process == 0);
			CHECK(Decomp.frequencies()[0].upper_process == 0);
			CHECK(Decomp.frequencies()[1].lower_process == 1);
			CHECK(Decomp.frequencies()[1].upper_process == 1);
			CHECK(Decomp.frequencies()[2].lower_process == 2);
			CHECK(Decomp.frequencies()[2].upper_process == 2);
			CHECK(Decomp.frequencies()[3].lower_process == 3);
			CHECK(Decomp.frequencies()[3].upper_process == 3);
		}
		for(int i=0; i<frequencies; i++){
			Decomp.decompose_time_blocks(i, time_blocks[i], true);
			CHECK(Decomp.time_blocks(i).size() == blocks_per_frequency);
		}
		if(blocks_per_frequency == 3 and frequencies == 4){
			CHECK(Decomp.time_blocks(0)[0].global_owner == 0);
			CHECK(Decomp.time_blocks(0)[1].global_owner == 0);
			CHECK(Decomp.time_blocks(0)[2].global_owner == 0);
			CHECK(Decomp.time_blocks(1)[0].global_owner == 1);
			CHECK(Decomp.time_blocks(1)[1].global_owner == 1);
			CHECK(Decomp.time_blocks(1)[2].global_owner == 1);
			CHECK(Decomp.time_blocks(2)[0].global_owner == 2);
			CHECK(Decomp.time_blocks(2)[1].global_owner == 2);
			CHECK(Decomp.time_blocks(2)[2].global_owner == 2);
			CHECK(Decomp.time_blocks(3)[0].global_owner == 3);
			CHECK(Decomp.time_blocks(3)[1].global_owner == 3);
			CHECK(Decomp.time_blocks(3)[2].global_owner == 3);
		}
		upper_process = 2;
		Decomp = mpi::Decomposition(parallel, world);
		Decomp.decompose_frequencies(frequencies, 0, upper_process, false, false, true);
		CHECK(Decomp.frequencies().size() == frequencies);
		if(frequencies == 4){
			CHECK(Decomp.frequencies()[0].lower_process == 0);
			CHECK(Decomp.frequencies()[0].upper_process == 0);
			CHECK(Decomp.frequencies()[1].lower_process == 0);
			CHECK(Decomp.frequencies()[1].upper_process == 0);
			CHECK(Decomp.frequencies()[2].lower_process == 1);
			CHECK(Decomp.frequencies()[2].upper_process == 1);
			CHECK(Decomp.frequencies()[3].lower_process == 2);
			CHECK(Decomp.frequencies()[3].upper_process == 2);
		}
		for(int i=0; i<frequencies; i++){
			Decomp.decompose_time_blocks(i, time_blocks[i], true);
			CHECK(Decomp.time_blocks(i).size() == blocks_per_frequency);
		}
		if(blocks_per_frequency == 3 and frequencies == 4){
			CHECK(Decomp.time_blocks(0)[0].global_owner == 0);
			CHECK(Decomp.time_blocks(0)[1].global_owner == 0);
			CHECK(Decomp.time_blocks(0)[2].global_owner == 0);
			CHECK(Decomp.time_blocks(1)[0].global_owner == 0);
			CHECK(Decomp.time_blocks(1)[1].global_owner == 0);
			CHECK(Decomp.time_blocks(1)[2].global_owner == 0);
			CHECK(Decomp.time_blocks(2)[0].global_owner == 1);
			CHECK(Decomp.time_blocks(2)[1].global_owner == 1);
			CHECK(Decomp.time_blocks(2)[2].global_owner == 1);
			CHECK(Decomp.time_blocks(3)[0].global_owner == 2);
			CHECK(Decomp.time_blocks(3)[1].global_owner == 2);
			CHECK(Decomp.time_blocks(3)[2].global_owner == 2);
		}
		upper_process = 1;
		Decomp = mpi::Decomposition(parallel, world);
		Decomp.decompose_frequencies(frequencies, 0, upper_process, false, false, true);
		CHECK(Decomp.frequencies().size() == frequencies);
		if(frequencies == 4){
			CHECK(Decomp.frequencies()[0].lower_process == 0);
			CHECK(Decomp.frequencies()[0].upper_process == 0);
			CHECK(Decomp.frequencies()[1].lower_process == 0);
			CHECK(Decomp.frequencies()[1].upper_process == 0);
			CHECK(Decomp.frequencies()[2].lower_process == 1);
			CHECK(Decomp.frequencies()[2].upper_process == 1);
			CHECK(Decomp.frequencies()[3].lower_process == 1);
			CHECK(Decomp.frequencies()[3].upper_process == 1);
		}
		for(int i=0; i<frequencies; i++){
			Decomp.decompose_time_blocks(i, time_blocks[i], true);
			CHECK(Decomp.time_blocks(i).size() == blocks_per_frequency);
		}
		if(blocks_per_frequency == 3 and frequencies == 4){
			CHECK(Decomp.time_blocks(0)[0].global_owner == 0);
			CHECK(Decomp.time_blocks(0)[1].global_owner == 0);
			CHECK(Decomp.time_blocks(0)[2].global_owner == 0);
			CHECK(Decomp.time_blocks(1)[0].global_owner == 0);
			CHECK(Decomp.time_blocks(1)[1].global_owner == 0);
			CHECK(Decomp.time_blocks(1)[2].global_owner == 0);
			CHECK(Decomp.time_blocks(2)[0].global_owner == 1);
			CHECK(Decomp.time_blocks(2)[1].global_owner == 1);
			CHECK(Decomp.time_blocks(2)[2].global_owner == 1);
			CHECK(Decomp.time_blocks(3)[0].global_owner == 1);
			CHECK(Decomp.time_blocks(3)[1].global_owner == 1);
			CHECK(Decomp.time_blocks(3)[2].global_owner == 1);
		}
		upper_process = 0;
		Decomp = mpi::Decomposition(parallel, world);
		Decomp.decompose_frequencies(frequencies, 0, upper_process, false, false, true);
		CHECK(Decomp.frequencies().size() == frequencies);
		if(frequencies == 4){
			CHECK(Decomp.frequencies()[0].lower_process == 0);
			CHECK(Decomp.frequencies()[0].upper_process == 0);
			CHECK(Decomp.frequencies()[1].lower_process == 0);
			CHECK(Decomp.frequencies()[1].upper_process == 0);
			CHECK(Decomp.frequencies()[2].lower_process == 0);
			CHECK(Decomp.frequencies()[2].upper_process == 0);
			CHECK(Decomp.frequencies()[3].lower_process == 0);
			CHECK(Decomp.frequencies()[3].upper_process == 0);
		}
		for(int i=0; i<frequencies; i++){
			Decomp.decompose_time_blocks(i, time_blocks[i], true);
			CHECK(Decomp.time_blocks(i).size() == blocks_per_frequency);
		}
		if(blocks_per_frequency == 3 and frequencies == 4){
			CHECK(Decomp.time_blocks(0)[0].global_owner == 0);
			CHECK(Decomp.time_blocks(0)[1].global_owner == 0);
			CHECK(Decomp.time_blocks(0)[2].global_owner == 0);
			CHECK(Decomp.time_blocks(1)[0].global_owner == 0);
			CHECK(Decomp.time_blocks(1)[1].global_owner == 0);
			CHECK(Decomp.time_blocks(1)[2].global_owner == 0);
			CHECK(Decomp.time_blocks(2)[0].global_owner == 0);
			CHECK(Decomp.time_blocks(2)[1].global_owner == 0);
			CHECK(Decomp.time_blocks(2)[2].global_owner == 0);
			CHECK(Decomp.time_blocks(3)[0].global_owner == 0);
			CHECK(Decomp.time_blocks(3)[1].global_owner == 0);
			CHECK(Decomp.time_blocks(3)[2].global_owner == 0);
		}

	}

	SECTION ("Build Decomposition Single Frequency"){

		auto Decomp = mpi::Decomposition(parallel, world);

		// This test case only has one frequency
		t_int frequencies = 1;

		t_int nlevels = 7;
		t_int blocks_per_frequency = 30;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = blocks_per_frequency;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Decomp.decompose_primal_dual(false, true, false, true, true, frequencies, wavelet_levels, time_blocks, sub_blocks);

	}



	SECTION("Collect indices - Only time block parallelisation"){
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

		int indices_per_block = 6;

		std::vector<std::vector<Vector<t_int>>> indices(frequencies);

		for(int f=0; f<frequencies; f++){
			indices[f] = std::vector<Vector<t_int>>(blocks_per_process);
			for(int t=0; t<blocks_per_process; t++){
				indices[f][t] = Vector<t_int>(indices_per_block);
				indices[f][t].fill(world.rank());
			}
		}

		std::vector<std::vector<Vector<t_int>>> global_indices(frequencies);
		for(int f=0; f<frequencies; f++){
			global_indices[f] = std::vector<Vector<t_int>>(n_blocks);
		}
		Decomp.decompose_primal_dual(false, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);
		Decomp.template collect_indices<Vector<t_int>>(indices, global_indices, true);
		for(int f=0; f<frequencies; f++){
			if(world.rank() == 0){
				CHECK(global_indices[f].size() == n_blocks);
				for(int i=0; i<n_blocks; i++){
					CHECK(global_indices[f][i].size() == indices_per_block);
					for(int j=0; j<global_indices[f][i].size(); j++){
						CHECK(global_indices[f][i][j] == i/blocks_per_process);
					}
				}
			}
		}
	}

	SECTION("Collect indices - multiple frequencies including time block parallelsation"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int nlevels = 1;
		t_int n_blocks = 2;
		t_int frequencies_per_process = 2;
		t_int frequencies = world.size()*frequencies_per_process;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		for(int f=0; f<frequencies; f++){
			wavelet_levels[f] = nlevels;
		}

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		for(int f=0; f<frequencies; f++){
			time_blocks[f] = n_blocks;
		}

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(frequencies);
		for(int f=0; f<frequencies; f++){
			sub_blocks[f] = std::vector<t_int>(n_blocks);
			for(int t=0; t<n_blocks; t++){
				sub_blocks[f][t] = 0;
			}
		}

		int indices_per_block = 6;

		std::vector<std::vector<Vector<t_int>>> indices(frequencies_per_process);

		for(int f=0; f<frequencies_per_process; f++){
			indices[f] = std::vector<Vector<t_int>>(n_blocks);
			for(int t=0; t<n_blocks; t++){
				indices[f][t] = Vector<t_int>(indices_per_block);
				indices[f][t].fill(world.rank());
			}
		}

		std::vector<std::vector<Vector<t_int>>> global_indices(frequencies);
		for(int f=0; f<frequencies; f++){
			global_indices[f] = std::vector<Vector<t_int>>(n_blocks);
		}
		Decomp.decompose_primal_dual(true, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);
		Decomp.template collect_indices<Vector<t_int>>(indices, global_indices, true);
		if(Decomp.global_comm().is_root()){
			for(int f=0; f<frequencies; f++){
				CHECK(global_indices[f].size() == n_blocks);
				for(int i=0; i<n_blocks; i++){
					CHECK(global_indices[f][i].size() == indices_per_block);
					for(int j=0; j<global_indices[f][i].size(); j++){
						CHECK(global_indices[f][i][j] == f/frequencies_per_process);
					}
				}
			}
		}
	}


	SECTION("Collect indices - multiple frequencies without time block parallelisation"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int nlevels = 1;
		t_int n_blocks = 2;
		t_int frequencies_per_process = 2;
		t_int frequencies = world.size()*frequencies_per_process;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		for(int f=0; f<frequencies; f++){
			wavelet_levels[f] = nlevels;
		}

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		for(int f=0; f<frequencies; f++){
			time_blocks[f] = n_blocks;
		}

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(frequencies);
		for(int f=0; f<frequencies; f++){
			sub_blocks[f] = std::vector<t_int>(n_blocks);
			for(int t=0; t<n_blocks; t++){
				sub_blocks[f][t] = 0;
			}
		}

		int indices_per_block = 6;

		std::vector<std::vector<Vector<t_int>>> indices(frequencies_per_process);

		for(int f=0; f<frequencies_per_process; f++){
			indices[f] = std::vector<Vector<t_int>>(n_blocks);
			for(int t=0; t<n_blocks; t++){
				indices[f][t] = Vector<t_int>(indices_per_block);
				indices[f][t].fill(world.rank());
			}
		}

		std::vector<std::vector<Vector<t_int>>> global_indices(frequencies);
		for(int f=0; f<frequencies; f++){
			global_indices[f] = std::vector<Vector<t_int>>(n_blocks);
		}
		Decomp.decompose_primal_dual(true, false, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);
		Decomp.template collect_indices<Vector<t_int>>(indices, global_indices, true);
		if(Decomp.global_comm().is_root()){
			for(int f=0; f<frequencies; f++){
				CHECK(global_indices[f].size() == n_blocks);
				for(int i=0; i<n_blocks; i++){
					CHECK(global_indices[f][i].size() == indices_per_block);
					for(int j=0; j<global_indices[f][i].size(); j++){
						CHECK(global_indices[f][i][j] == f/frequencies_per_process);
					}
				}
			}
		}
	}

	SECTION("Distribute sparse matrices"){
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

		int indices_per_block = 10;

		Decomp.decompose_primal_dual(true, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);

		std::vector<std::vector<Eigen::SparseMatrix<t_complex>>> x_hat_i;
		std::vector<std::vector<Eigen::SparseMatrix<t_complex>>> x_hat_sparse(Decomp.my_number_of_frequencies());
		for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
			x_hat_sparse[f] = std::vector<Eigen::SparseMatrix<t_complex>>(Decomp.my_frequencies()[f].number_of_time_blocks);
		}
		if(world.rank() == 0){

			x_hat_i = std::vector<std::vector<Eigen::SparseMatrix<t_complex>>>(frequencies);
			for(int f=0; f<frequencies; f++){
				x_hat_i[f] = std::vector<Eigen::SparseMatrix<t_complex>>(Decomp.frequencies()[f].number_of_time_blocks);
			}
			for(int f=0; f<frequencies; f++){
				for(int i=0;i<Decomp.frequencies()[f].number_of_time_blocks;i++){
					x_hat_i[f][i] = Eigen::SparseMatrix<t_complex>(indices_per_block, 1);
					x_hat_i[f][i].reserve(indices_per_block);
					for (int k=0; k<indices_per_block; k++){
						x_hat_i[f][i].insert(k,0) = i/(blocks_per_process)*10;
					}
				}
			}
		}

		Decomp.template distribute_fourier_data<t_complex>(x_hat_sparse, x_hat_i);

		for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
			CHECK(x_hat_sparse[f].size() == blocks_per_process);
			for(int i=0; i<blocks_per_process; i++){
				CHECK(x_hat_sparse[f][i].size() == indices_per_block);
				for(int j=0; j<indices_per_block; j++){
					CHECK(std::real(x_hat_sparse[f][i].coeff(j,0)) == world.rank()*10);
				}
			}
		}
	}


	SECTION("Distribute sparse matrices with split and nonblocking communications"){
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

		int indices_per_block = 1000;

		Decomp.decompose_primal_dual(true, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);

		int max_block_number = 0;
		for(int f=0; f<Decomp.global_number_of_frequencies(); f++){
			if(Decomp.frequencies()[f].number_of_time_blocks > max_block_number){
				max_block_number = Decomp.frequencies()[f].number_of_time_blocks;
			}
		}

		int shapes[Decomp.global_number_of_frequencies()][max_block_number][3];

		std::vector<std::vector<Eigen::SparseMatrix<t_complex>>> x_hat_i;
		if(Decomp.global_comm().is_root()){
			x_hat_i = std::vector<std::vector<Eigen::SparseMatrix<t_complex>>>(Decomp.global_number_of_frequencies());
		}
		std::vector<std::vector<Eigen::SparseMatrix<t_complex>>> x_hat_sparse(Decomp.my_number_of_frequencies());
		for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
			x_hat_sparse[f] = std::vector<Eigen::SparseMatrix<t_complex>>(Decomp.my_frequencies()[f].number_of_time_blocks);
		}

		int my_local_indices = 0;
		int my_global_indices = 0;

		for(int f=0; f<Decomp.global_number_of_frequencies(); f++){

			if(Decomp.global_comm().is_root()){
				Decomp.initialise_requests(f,Decomp.frequencies()[f].number_of_time_blocks*4);
			}


			if(Decomp.frequencies()[f].in_this_frequency or Decomp.global_comm().is_root()){

				// Start the receives first.
				if(not Decomp.global_comm().is_root()){
					Decomp.template receive_fourier_data<t_complex>(x_hat_sparse[my_local_indices], f, my_local_indices, true);
					my_local_indices++;
				}

				if(Decomp.global_comm().is_root()){
					int my_index = 0;
					bool used_this_freq = false;
					x_hat_i[my_global_indices] = std::vector<Eigen::SparseMatrix<t_complex>>(Decomp.frequencies()[f].number_of_time_blocks);
					for(int i=0;i<Decomp.frequencies()[f].number_of_time_blocks;i++){
						x_hat_i[my_global_indices][i] = Eigen::SparseMatrix<t_complex>(indices_per_block, 1);
						x_hat_i[my_global_indices][i].reserve(indices_per_block);
						x_hat_i[my_global_indices][i].setZero();
						for (int k=0; k<indices_per_block; k++){
							if(k%3==0){
								x_hat_i[my_global_indices][i].insert(k,0) = (i/blocks_per_process)*10;
							}
						}
						Decomp.template send_fourier_data<t_complex>(x_hat_sparse[my_local_indices], x_hat_i[my_global_indices], &shapes[my_local_indices][i][0], i, my_index, f, used_this_freq, true);
					}
					if(used_this_freq){
						my_local_indices++;
					}
					my_global_indices++;
				}

				if(Decomp.global_comm().is_root()){
					Decomp.wait_on_requests(f, Decomp.frequencies()[f].number_of_time_blocks*4);
					Decomp.cleanup_requests(f);
				}
			}

		}

		for(int f=0; f<Decomp.my_number_of_frequencies(); f++){

			CHECK(x_hat_sparse[f].size() == blocks_per_process);
			for(int i=0; i<blocks_per_process; i++){
				CHECK(x_hat_sparse[f][i].size() == indices_per_block);
				for(int j=0; j<indices_per_block; j++){
					if(j%3 == 0){
						CHECK(std::real(x_hat_sparse[f][i].coeff(j,0)) == world.rank()*10.0);
					}else{
						CHECK(std::real(x_hat_sparse[f][i].coeff(j,0)) == 0);
					}
				}
			}
		}
	}

	SECTION("Distribute sparse matrices with split and nonblocking communications with OpenMP"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int nlevels = 1;
		t_int blocks_per_process = 2;
		t_int n_blocks = world.size()*blocks_per_process;
		t_int frequencies = 1;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		for(int f=0; f<frequencies; f++){
			wavelet_levels[f] = nlevels;
		}

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		for(int f=0; f<frequencies; f++){
			time_blocks[f] = n_blocks;
		}

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(frequencies);
		for(int f=0; f<frequencies; f++){
			sub_blocks[f] = std::vector<t_int>(n_blocks);
			for(int t=0; t<n_blocks; t++){
				sub_blocks[f][t] = 0;
			}
		}

		int indices_per_block = 1000;

		Decomp.decompose_primal_dual(true, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);

		int max_block_number = 0;
		for(int f=0; f<Decomp.global_number_of_frequencies(); f++){
			if(Decomp.frequencies()[f].number_of_time_blocks > max_block_number){
				max_block_number = Decomp.frequencies()[f].number_of_time_blocks;
			}
		}

		int shapes[Decomp.global_number_of_frequencies()][max_block_number][3];

		std::vector<std::vector<Eigen::SparseMatrix<t_complex>>> x_hat_i;
		if(Decomp.global_comm().is_root()){
			x_hat_i = std::vector<std::vector<Eigen::SparseMatrix<t_complex>>>(Decomp.global_number_of_frequencies());
		}
		std::vector<std::vector<Eigen::SparseMatrix<t_complex>>> x_hat_sparse(Decomp.my_number_of_frequencies());
		for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
			x_hat_sparse[f] = std::vector<Eigen::SparseMatrix<t_complex>>(Decomp.my_frequencies()[f].number_of_time_blocks);
		}

		int my_local_indices = 0;
		int my_global_indices = 0;

		for(int f=0; f<Decomp.global_number_of_frequencies(); f++){

			if(Decomp.global_comm().is_root()){
				Decomp.initialise_requests(f,Decomp.frequencies()[f].number_of_time_blocks*4);
			}

			if(Decomp.frequencies()[f].in_this_frequency){

				// Start the receives first.
				if(not Decomp.global_comm().is_root()){
					Decomp.template receive_fourier_data<t_complex>(x_hat_sparse[my_local_indices], f, my_local_indices, true);
					my_local_indices++;
				}

				if(Decomp.global_comm().is_root()){
					x_hat_i[my_global_indices] = std::vector<Eigen::SparseMatrix<t_complex>>(Decomp.frequencies()[f].number_of_time_blocks);
#pragma omp parallel for default(shared)
					for(int t=0;t<Decomp.frequencies()[f].number_of_time_blocks;t++){
						x_hat_i[my_global_indices][t] = Eigen::SparseMatrix<t_complex>(indices_per_block, 1);
						x_hat_i[my_global_indices][t].reserve(indices_per_block);
						x_hat_i[my_global_indices][t].setZero();
						for (int k=0; k<indices_per_block; k++){
							if(k%3==0){
								x_hat_i[my_global_indices][t].insert(k,0) = (t/blocks_per_process)*10;
							}
						}
					}
					int my_index = 0;
					bool used_this_freq = false;
					for(int t=0;t<Decomp.frequencies()[f].number_of_time_blocks;t++){
						Decomp.template send_fourier_data<t_complex>(x_hat_sparse[my_local_indices], x_hat_i[my_global_indices], &shapes[my_local_indices][t][0], t, my_index, f, used_this_freq, true);
					}
					if(used_this_freq){
						my_local_indices++;
					}
					my_global_indices++;
				}

				if(Decomp.global_comm().is_root()){
					Decomp.wait_on_requests(f, Decomp.frequencies()[f].number_of_time_blocks*4);
					Decomp.cleanup_requests(f);
				}
			}

		}

		for(int f=0; f<Decomp.my_number_of_frequencies(); f++){

			CHECK(x_hat_sparse[f].size() == blocks_per_process);
			for(int i=0; i<blocks_per_process; i++){
				CHECK(x_hat_sparse[f][i].size() == indices_per_block);
				for(int j=0; j<indices_per_block; j++){
					if(j%3 == 0){
						CHECK(std::real(x_hat_sparse[f][i].coeff(j,0)) == world.rank()*10.0);
					}else{
						CHECK(std::real(x_hat_sparse[f][i].coeff(j,0)) == 0);
					}
				}
			}
		}
	}

	SECTION("Collect residual norms"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int nlevels = 1;
		t_int n_blocks = world.size()*2;
		t_int frequencies = 1;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = n_blocks;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Vector<Vector<t_real>> residual_norms(1);
		residual_norms(0) = Vector<t_real>(2);
		Vector<Vector<t_real>> total_residual_norms(1);
		total_residual_norms(0) = Vector<t_real>(n_blocks);
		residual_norms(0).fill(world.rank());
		Decomp.decompose_primal_dual(false, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);
		Decomp.template collect_residual_norms<t_real>(residual_norms, total_residual_norms);
		if(world.rank() == 0){
			CHECK(total_residual_norms(0).size() == world.size()*2);
			for(int i=0; i<world.size()*2; i++){
				CHECK(total_residual_norms[0][i] == i/2);
			}
		}
	}

	SECTION("Collect and distributed epsilons"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int nlevels = 1;
		t_int n_blocks = world.size()*2;
		t_int frequencies = 1;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = n_blocks;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Vector<t_real> epsilons(2);
		Vector<t_real> new_epsilons(2);
		Vector<t_real> total_epsilons(n_blocks);
		epsilons.fill(world.rank());
		Decomp.decompose_primal_dual(false, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);
		Decomp.template collect_epsilons<Vector<t_real>>(epsilons, total_epsilons);

		if(world.rank() == 0){
			CHECK(total_epsilons.size() == world.size()*2);
			for(int i=0; i<world.size()*2; i++){
				CHECK(total_epsilons[i] == i/2);
			}
		}
		Decomp.template distribute_epsilons<Vector<t_real>>(new_epsilons, total_epsilons);
		CHECK(epsilons.size() ==new_epsilons.size());
		for(int i=0; i<2; i++){
			CHECK(epsilons[i] == new_epsilons[i]);
		}
	}

	SECTION("Collect and distributed x_bar with wideband and wavelet parallelisaton - frequency per process"){
		if(world_size >= 2){
			auto Decomp = mpi::Decomposition(parallel, world);
			t_int nlevels = 4;
			t_int n_blocks = 2;
			t_int frequencies = world_size;
			t_int x_bar_size = 24;

			std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
			for(int f=0; f<frequencies; f++){
				wavelet_levels[f] = nlevels;
			}

			std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
			for(int f=0; f<frequencies; f++){
				time_blocks[f] = n_blocks;
			}

			std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(frequencies);
			for(int f=0; f<frequencies; f++){
				sub_blocks[f] = std::vector<t_int>(n_blocks);
				for(int t=0; t<n_blocks; t++){
					sub_blocks[f][t] = 0;
				}
			}

			Matrix<t_real> x_bar(x_bar_size, frequencies/world_size);
			Matrix<t_real> new_x_bar(x_bar_size, frequencies/world_size);
			Matrix<t_real> total_x_bar(x_bar_size, frequencies);
			for(int i=0; i<x_bar.rows(); i++){
				for(int j=0; j<x_bar.cols(); j++){
					x_bar(i, j) = world.rank()+1;
				}
			}
			Decomp.decompose_primal_dual(true, true, false, true, true, frequencies, wavelet_levels, time_blocks, sub_blocks);
			Decomp.template collect_frequency_root_data<t_real>(x_bar, total_x_bar);

			if(world.rank() == 0){
				CHECK(total_x_bar.cols() == frequencies);
				for(int j=0; j<x_bar_size; j++){
					CHECK(total_x_bar.rows() == x_bar_size);
					for(int i=0; i<frequencies; i++){
						CHECK(static_cast<t_int>(total_x_bar(j, i)) == static_cast<t_int>((i+1/world_size))+1);
					}
				}
			}
			Decomp.template distribute_frequency_data<Matrix<t_real>, t_real>(new_x_bar, total_x_bar, true);
			CHECK(x_bar.rows() == new_x_bar.rows());
			for(int i=0; i<x_bar.rows(); i++){
				CHECK(x_bar.cols() == new_x_bar.cols());
				for(int j=0; j<x_bar.cols(); j++){
					if(Decomp.my_frequencies()[j].global_owner == Decomp.global_comm().rank()){
						CHECK(x_bar(i, j) == new_x_bar(i, j));
					}
				}
			}
		}else{
			WARN("Test skipped because it requires at least 2 MPI processes");
		}
	}

	SECTION("Collect and distributed x_bar with wideband and wavelet parallelisaton - two processes per frequency - root only"){
		if(world_size >= 2){
			auto Decomp = mpi::Decomposition(parallel, world);
			t_int nlevels = 4;
			t_int n_blocks = 2;
			t_int frequencies = world_size/2;
			t_int x_bar_size = 24;

			std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
			for(int f=0; f<frequencies; f++){
				wavelet_levels[f] = nlevels;
			}

			std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
			for(int f=0; f<frequencies; f++){
				time_blocks[f] = n_blocks;
			}

			std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(frequencies);
			for(int f=0; f<frequencies; f++){
				sub_blocks[f] = std::vector<t_int>(n_blocks);
				for(int t=0; t<n_blocks; t++){
					sub_blocks[f][t] = 0;
				}
			}

			Decomp.decompose_primal_dual(true, true, false, true, true, frequencies, wavelet_levels, time_blocks, sub_blocks);

			Matrix<t_real> x_bar(x_bar_size, 1);
			Matrix<t_real> new_x_bar(x_bar_size, 1);
			Matrix<t_real> total_x_bar(x_bar_size, frequencies);
			if(Decomp.my_frequencies()[0].global_owner == Decomp.global_comm().rank()){
				for(int i=0; i<x_bar.rows(); i++){
					for(int j=0; j<x_bar.cols(); j++){
						x_bar(i, j) = world.rank()+1;
					}
				}
			}

			Decomp.template collect_frequency_root_data<t_real>(x_bar, total_x_bar);

			if(world.rank() == 0){
				CHECK(total_x_bar.cols() == frequencies);
				for(int j=0; j<x_bar_size; j++){
					CHECK(total_x_bar.rows() == x_bar_size);
					for(int i=0; i<frequencies; i++){
						CHECK(static_cast<t_int>(total_x_bar(j, i)) == static_cast<t_int>(Decomp.frequencies()[i].global_owner)+1);
					}
				}
			}
			Decomp.template distribute_frequency_data<Matrix<t_real>, t_real>(new_x_bar, total_x_bar, true);
			CHECK(x_bar.rows() == new_x_bar.rows());
			for(int i=0; i<x_bar.rows(); i++){
				CHECK(x_bar.cols() == new_x_bar.cols());
				for(int j=0; j<x_bar.cols(); j++){
					if(Decomp.my_frequencies()[j].global_owner == Decomp.global_comm().rank()){
						CHECK(x_bar(i, j) == new_x_bar(i, j));
					}
				}
			}
		}else{
			WARN("Test skipped because it requires at least 2 MPI processes");
		}
	}

	SECTION("Collect and distributed x_bar with wideband and wavelet parallelisaton - two processes per frequency - distribute to all processes"){
		if(world_size >= 2){
			auto Decomp = mpi::Decomposition(parallel, world);
			t_int nlevels = 4;
			t_int n_blocks = 2;
			t_int frequencies = world_size/2;
			t_int x_bar_size = 24;

			std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
			for(int f=0; f<frequencies; f++){
				wavelet_levels[f] = nlevels;
			}

			std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
			for(int f=0; f<frequencies; f++){
				time_blocks[f] = n_blocks;
			}

			std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(frequencies);
			for(int f=0; f<frequencies; f++){
				sub_blocks[f] = std::vector<t_int>(n_blocks);
				for(int t=0; t<n_blocks; t++){
					sub_blocks[f][t] = 0;
				}
			}

			Decomp.decompose_primal_dual(true, true, false, true, true, frequencies, wavelet_levels, time_blocks, sub_blocks);

			Matrix<t_real> x_bar(x_bar_size, 1);
			Matrix<t_real> new_x_bar(x_bar_size, 1);
			Matrix<t_real> total_x_bar(x_bar_size, frequencies);
			if(Decomp.my_frequencies()[0].global_owner == Decomp.global_comm().rank()){
				for(int i=0; i<x_bar.rows(); i++){
					for(int j=0; j<x_bar.cols(); j++){
						x_bar(i, j) = world.rank()+1;
					}
				}
			}

			Decomp.template collect_frequency_root_data<t_real>(x_bar, total_x_bar);

			if(world.rank() == 0){
				CHECK(total_x_bar.cols() == frequencies);
				for(int j=0; j<x_bar_size; j++){
					CHECK(total_x_bar.rows() == x_bar_size);
					for(int i=0; i<frequencies; i++){
						CHECK(static_cast<t_int>(total_x_bar(j, i)) == static_cast<t_int>(Decomp.frequencies()[i].global_owner)+1);
					}
				}
			}
			Decomp.template distribute_frequency_data<Matrix<t_real>, t_real>(new_x_bar, total_x_bar, false);
			CHECK(x_bar.rows() == new_x_bar.rows());
			for(int i=0; i<x_bar.rows(); i++){
				CHECK(x_bar.cols() == new_x_bar.cols());
				for(int j=0; j<x_bar.cols(); j++){
					if(Decomp.my_frequencies()[j].global_owner == Decomp.global_comm().rank()){
						CHECK(x_bar(i, j) == new_x_bar(i, j));
					}else{
						CHECK(static_cast<t_int>(new_x_bar(i, j)) == static_cast<t_int>(Decomp.my_frequencies()[j].global_owner)+1);
					}
				}
			}
		}else{
			WARN("Test skipped because it requires at least 2 MPI processes");
		}
	}


	SECTION("Collect and distributed x_bar with wavelet parallelisaton - blocks per process"){
		auto Decomp = mpi::Decomposition(parallel, world);
		if(world_size >= 2){
			t_int nlevels = 4;
			t_int n_blocks = 2;
			t_int frequencies = world_size/2;
			t_int x_bar_size = 24;

			std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
			for(int f=0; f<frequencies; f++){
				wavelet_levels[f] = nlevels;
			}

			std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
			for(int f=0; f<frequencies; f++){
				time_blocks[f] = n_blocks;
			}

			std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(frequencies);
			for(int f=0; f<frequencies; f++){
				sub_blocks[f] = std::vector<t_int>(n_blocks);
				for(int t=0; t<n_blocks; t++){
					sub_blocks[f][t] = 0;
				}
			}

			Decomp.decompose_primal_dual(true, true, false, true, true, frequencies, wavelet_levels, time_blocks, sub_blocks);

			Matrix<t_real> x_bar(x_bar_size, Decomp.my_number_of_frequencies());
			Matrix<t_real> new_x_bar(x_bar_size, Decomp.my_number_of_frequencies());
			Matrix<t_real> total_x_bar(x_bar_size, frequencies);
			for(int j=0; j<x_bar_size; j++){
				for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
					x_bar(j, f) = Decomp.my_frequencies()[f].global_wavelet_owner+1;
				}
			}
			Decomp.template collect_frequency_root_data<t_real>(x_bar, total_x_bar);

			if(world.rank() == 0){
				CHECK(total_x_bar.cols() == frequencies);
				CHECK(total_x_bar.rows() == x_bar_size);
				for(int j=0; j<x_bar_size; j++){
					for(int i=0; i<frequencies; i++){
						CHECK(static_cast<t_int>(total_x_bar(j, i)) == static_cast<t_int>(Decomp.frequencies()[i].global_owner)+1);
					}
				}
			}
			Decomp.template distribute_frequency_data<Matrix<t_real>, t_real>(new_x_bar, total_x_bar, true);
			CHECK(x_bar.rows() == new_x_bar.rows());
			CHECK(x_bar.cols() == new_x_bar.cols());
			for(int i=0; i<x_bar.rows(); i++){
				for(int j=0; j<x_bar.cols(); j++){
					if(Decomp.my_frequencies()[j].global_owner == Decomp.global_comm().rank()){
						CHECK(x_bar(i, j) == new_x_bar(i, j));
					}
				}
			}

			for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
				if(Decomp.my_frequencies()[f].wavelet_comm.size()>1){
					if(Decomp.my_frequencies()[f].number_of_wavelets != 0){
						Vector<t_real> local_x_bar(new_x_bar.rows());
						if(Decomp.my_frequencies()[f].wavelet_comm.is_root()){
							local_x_bar = new_x_bar.col(f);
						}
						local_x_bar = Decomp.my_frequencies()[f].wavelet_comm.broadcast(local_x_bar, Decomp.my_frequencies()[f].wavelet_comm.root_id());
						if(not Decomp.my_frequencies()[f].wavelet_comm.is_root()){
							new_x_bar.col(f) = local_x_bar;
						}
					}
					for(int i=0; i<x_bar.rows(); i++){
						if(Decomp.my_frequencies()[f].number_of_wavelets != 0){
							CHECK(x_bar(i, f) == new_x_bar(i, f));
						}
					}
				}else{
					WARN("Test skipped because it requires more than one process in the wavelet decomposition");
				}
			}



		}else{
			WARN("Test skipped because it requires at least 2 MPI processes");
		}
	}


	SECTION("Collect and distributed epsilons wideband blocking"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int nlevels = 1;
		t_int n_freqs = world.size()*2;
		t_int n_blocks = 2;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(n_freqs);
		for(int i=0;i<n_freqs;i++){
			wavelet_levels[i] = nlevels;
		}

		std::vector<t_int> time_blocks = std::vector<t_int>(n_freqs);
		for(int i=0;i<n_freqs;i++){
			time_blocks[i] = n_blocks;
		}

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Vector<Vector<t_real>> epsilons(2);
		for(int i=0; i<2; i++){
			epsilons(i) = Vector<t_real>(2);
			epsilons(i).fill(world.rank());
		}
		Vector<Vector<t_real>> new_epsilons(2);
		for(int i=0; i<2; i++){
			new_epsilons(i) = Vector<t_real>(2);
		}
		Vector<Vector<t_real>> total_epsilons(n_freqs);
		for(int i=0; i<n_freqs; i++){
			total_epsilons(i) = Vector<t_real>(2);
		}
		Decomp.decompose_primal_dual(true, true, false, false, false, n_freqs, wavelet_levels, time_blocks, sub_blocks);
		Decomp.template collect_epsilons_wideband_blocking<Vector<Vector<t_real>>>(epsilons, total_epsilons);

		if(world.rank() == 0){
			CHECK(total_epsilons.size() == world.size()*2);
			for(int i=0; i<world.size()*2; i++){
				CHECK(total_epsilons(i).size() == 2);
				for(int j=0; j<2; j++){
					CHECK(total_epsilons[i][j] == i/2);
				}
			}
		}
		Decomp.template distribute_epsilons_wideband_blocking<Vector<Vector<t_real>>>(new_epsilons, total_epsilons);
		CHECK(epsilons.size() == new_epsilons.size());
		for(int i=0; i<2; i++){
			CHECK(epsilons(i).size() == new_epsilons(i).size());
			for(int j=0; j<2; j++){
				CHECK(epsilons[i][j] == new_epsilons[i][j]);
			}
		}
	}

	SECTION("Distribute l1 weight no wavelet parallelisation"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int weights_per_process = 2;
		t_uint nlevels = 5;
		t_int n_blocks = world.size()*2;
		t_int frequencies = 1;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = n_blocks;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Vector<t_real> l1_weights(nlevels*weights_per_process);
		Vector<t_real> total_l1_weights(nlevels*weights_per_process);
		Decomp.decompose_primal_dual(false, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);
		if(Decomp.global_comm().is_root()){
			for(int i=0; i<Decomp.frequencies()[0].number_of_wavelets; i++){
				for(int j=0; j<weights_per_process; j++){
					total_l1_weights[(i*weights_per_process)+j] = 13;
				}
			}
		}
		Decomp.template distribute_l1_weights<Vector<t_real>>(l1_weights, total_l1_weights, weights_per_process);
		CHECK(total_l1_weights.size() == nlevels*weights_per_process);
		CHECK(l1_weights.size() == nlevels*weights_per_process);
		if(Decomp.global_comm().is_root()){
			for(int i=0; i<l1_weights.size(); i++){
				CHECK(l1_weights[i] == 13);
			}
		}
	}

	SECTION("Distribute l1 weight with wavelet parallelisation"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int weights_per_process = 2;
		t_uint nlevels = 5;
		t_int n_blocks = world.size()*2;
		t_int frequencies = 1;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = n_blocks;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Vector<t_real> total_l1_weights(nlevels*weights_per_process);
		Decomp.decompose_primal_dual(false, true, false, true, true, frequencies, wavelet_levels, time_blocks, sub_blocks);
		Vector<t_real> l1_weights(Decomp.my_frequencies()[0].number_of_wavelets*weights_per_process);
		if(Decomp.global_comm().is_root()){
			for(int i=0; i<Decomp.frequencies()[0].number_of_wavelets; i++){
				int fill_value = Decomp.frequencies()[0].wavelets[i].global_owner;
				for(int j=0; j<weights_per_process; j++){
					total_l1_weights[(i*weights_per_process)+j] = fill_value;
				}
			}
		}
		Decomp.template distribute_l1_weights<Vector<t_real>>(l1_weights, total_l1_weights, weights_per_process);
		CHECK(total_l1_weights.size() == nlevels*weights_per_process);
		CHECK(l1_weights.size() == Decomp.my_frequencies()[0].number_of_wavelets*weights_per_process);
		for(int i=0; i<Decomp.my_frequencies()[0].number_of_wavelets*weights_per_process; i++){
			CHECK(l1_weights[i] == Decomp.global_comm().rank());
		}
	}

	SECTION("Collect l1 weight no wavelet parallelisation"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int weights_per_process = 2;
		t_uint nlevels = 5;
		t_int n_blocks = world.size()*2;
		t_int frequencies = 1;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = n_blocks;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Vector<t_real> l1_weights(nlevels*weights_per_process);
		Vector<t_real> total_l1_weights(nlevels*weights_per_process);
		Decomp.decompose_primal_dual(false, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);
		for(int i=0; i<Decomp.my_frequencies()[0].number_of_wavelets; i++){
			for(int j=0; j<weights_per_process; j++){
				l1_weights[(i*weights_per_process)+j] =13;
			}
		}

		Decomp.template collect_l1_weights<Vector<t_real>>(l1_weights, total_l1_weights, weights_per_process);
		CHECK(total_l1_weights.size() == nlevels*weights_per_process);
		CHECK(l1_weights.size() == nlevels*weights_per_process);
		if(Decomp.global_comm().is_root()){
			for(int i=0; i<total_l1_weights.size(); i++){
				CHECK(total_l1_weights[i] == 13);
			}
		}
	}

	SECTION("Collect l1 weight with wavelet parallelisation"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int weights_per_process = 2;
		t_uint nlevels = 5;
		t_int n_blocks = world.size()*2;
		t_int frequencies = 1;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = n_blocks;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Vector<t_real> total_l1_weights(nlevels*weights_per_process);
		Decomp.decompose_primal_dual(false, true, false, true, true, frequencies, wavelet_levels, time_blocks, sub_blocks);
		Vector<t_real> l1_weights(Decomp.my_frequencies()[0].number_of_wavelets*weights_per_process);
		for(int i=0; i<Decomp.my_frequencies()[0].number_of_wavelets; i++){
			for(int j=0; j<weights_per_process; j++){
				l1_weights[(i*weights_per_process)+j] = world.rank()*10;
			}
		}

		Decomp.template collect_l1_weights<Vector<t_real>>(l1_weights, total_l1_weights, weights_per_process);
		CHECK(l1_weights.size() == Decomp.my_frequencies()[0].number_of_wavelets*weights_per_process);
		if(Decomp.global_comm().is_root()){
			for(int i=0; i<Decomp.frequencies()[0].number_of_wavelets*weights_per_process; i++){
				CHECK(total_l1_weights[i] == (Decomp.frequencies()[0].wavelets[i/weights_per_process].global_owner)*10);
			}
		}
	}

	SECTION("Distribute l21 weight no wavelet parallelisation"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int weights_per_process = 2;
		t_uint nlevels = 5;
		t_int n_blocks = world.size()*2;
		t_int frequencies = 1;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = n_blocks;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Vector<t_real> total_l21_weights(nlevels*weights_per_process);
		Decomp.decompose_primal_dual(false, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);
		Vector<t_real> l21_weights(Decomp.my_number_of_root_wavelets()*weights_per_process);
		if(Decomp.global_comm().is_root()){
			for(int i=0; i<Decomp.global_number_of_root_wavelets(); i++){
				for(int j=0; j<weights_per_process; j++){
					total_l21_weights[(i*weights_per_process)+j] = 13;
				}
			}
		}
		Decomp.template distribute_l21_weights<Vector<t_real>>(l21_weights, total_l21_weights, weights_per_process);
		CHECK(total_l21_weights.size() == nlevels*weights_per_process);
		if(Decomp.global_comm().is_root()){
			CHECK(l21_weights.size() == nlevels*weights_per_process);
			for(int i=0; i<Decomp.my_number_of_root_wavelets(); i++){
				for(int j=0; j<weights_per_process; j++){
					CHECK(l21_weights[(i*weights_per_process)+j] == 13);
				}
			}
		}
	}

	SECTION("Distribute l21 weight with wavelet parallelisation"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int weights_per_process = 2;
		t_uint nlevels = 5;
		t_int n_blocks = world.size()*2;
		t_int frequencies = 1;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = n_blocks;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Vector<t_real> total_l21_weights(nlevels*weights_per_process);
		Decomp.decompose_primal_dual(false, true, false, true, true, frequencies, wavelet_levels, time_blocks, sub_blocks);
		Vector<t_real> l21_weights(Decomp.my_number_of_root_wavelets()*weights_per_process);
		if(Decomp.global_comm().is_root()){
			for(int i=0; i<Decomp.global_number_of_root_wavelets(); i++){
				int fill_value = Decomp.global_root_wavelets()[i].global_owner;
				for(int j=0; j<weights_per_process; j++){
					total_l21_weights[(i*weights_per_process)+j] = fill_value;
				}
			}
		}
		Decomp.template distribute_l21_weights<Vector<t_real>>(l21_weights, total_l21_weights, weights_per_process);
		CHECK(total_l21_weights.size() == nlevels*weights_per_process);
		CHECK(l21_weights.size() == Decomp.my_number_of_root_wavelets()*weights_per_process);
		for(int i=0; i<Decomp.my_number_of_root_wavelets()*weights_per_process; i++){
			CHECK(l21_weights[i] == Decomp.global_comm().rank());
		}
	}

	SECTION("Collect l21 weight no wavelet parallelisation"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int weights_per_process = 2;
		t_uint nlevels = 5;
		t_int n_blocks = world.size()*2;
		t_int frequencies = 1;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = n_blocks;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Vector<t_real> l21_weights(nlevels*weights_per_process);
		Vector<t_real> total_l21_weights(nlevels*weights_per_process);
		Decomp.decompose_primal_dual(false, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);
		for(int i=0; i<Decomp.global_number_of_root_wavelets(); i++){
			for(int j=0; j<weights_per_process; j++){
				l21_weights[(i*weights_per_process)+j] =13;
			}
		}

		Decomp.template collect_l21_weights<Vector<t_real>>(l21_weights, total_l21_weights, weights_per_process);
		CHECK(total_l21_weights.size() == nlevels*weights_per_process);
		CHECK(l21_weights.size() == nlevels*weights_per_process);
		if(Decomp.global_comm().is_root()){
			for(int i=0; i<total_l21_weights.size(); i++){
				CHECK(total_l21_weights[i] == 13);
			}
		}
	}

	SECTION("Collect l21 weight with wavelet parallelisation"){
		auto Decomp = mpi::Decomposition(parallel, world);
		t_int weights_per_process = 2;
		t_uint nlevels = 5;
		t_int n_blocks = world.size()*2;
		t_int frequencies = 1;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = n_blocks;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Vector<t_real> total_l21_weights(nlevels*weights_per_process);
		Decomp.decompose_primal_dual(false, true, false, true, true, frequencies, wavelet_levels, time_blocks, sub_blocks);
		Vector<t_real> l21_weights(Decomp.my_number_of_root_wavelets()*weights_per_process);
		for(int i=0; i<Decomp.my_number_of_root_wavelets(); i++){
			for(int j=0; j<weights_per_process; j++){
				l21_weights[(i*weights_per_process)+j] = world.rank()*10;
			}
		}

		Decomp.template collect_l21_weights<Vector<t_real>>(l21_weights, total_l21_weights, weights_per_process);
		CHECK(l21_weights.size() == Decomp.my_number_of_root_wavelets()*weights_per_process);
		if(Decomp.global_comm().is_root()){
			for(int i=0; i<Decomp.global_number_of_root_wavelets()*weights_per_process; i++){
				CHECK(total_l21_weights[i] == (Decomp.global_root_wavelets()[i/weights_per_process].global_owner)*10);
			}
		}
	}




}
#endif
