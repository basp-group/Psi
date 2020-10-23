#include <iostream>
#include <numeric>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <catch2/catch.hpp>

#include "psi/config.h"
#include "psi/mpi/communicator.h"
#include "psi/mpi/scalapack.h"
#include "psi/maths.h"
#include "psi/proximal.h"
#include <Eigen/SVD>
using namespace psi;

typedef psi::t_real Scalar;
typedef psi::Vector<Scalar> t_Vector;
typedef psi::Matrix<Scalar> t_Matrix;

#ifdef PSI_SCALAPACK
TEST_CASE("Setup Scalapack") {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	auto const world = mpi::Communicator::World();
	bool parallel = true;
	auto Decomp = mpi::Decomposition(parallel, world);

	srand (time(NULL));

	SECTION("General stuff") {

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		scalapack.setupBlacs(Decomp, size, size*2, size*2);

		int row, col;

		std::tie(row, col) = scalapack.process_grid_size(size);
		REQUIRE(row != 0);
		REQUIRE(col != 0);

		std::tie(row, col) = scalapack.process_grid_size(4);
		REQUIRE(row == 2);
		REQUIRE(col == 2);

		std::tie(row, col) = scalapack.process_grid_size(9);
		REQUIRE(row == 3);
		REQUIRE(col == 3);

		std::tie(row, col) = scalapack.process_grid_size(8);
		REQUIRE(row == 4);
		REQUIRE(col == 2);

		std::tie(row, col) = scalapack.process_grid_size(18);
		REQUIRE(row == 6);
		REQUIRE(col == 3);

		std::tie(row, col) = scalapack.process_grid_size(19);
		REQUIRE(row == 19);
		REQUIRE(col == 1);

		std::tie(row, col) = scalapack.process_grid_size(20);
		REQUIRE(row == 5);
		REQUIRE(col == 4);

	}

	SECTION("Test gather A: square matrix (vector)"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);


		int M = 256;
		int N = 256;

		scalapack.setupBlacs(Decomp, size, M, N);

		int npcol = scalapack.getnpcol();
		int nprow = scalapack.getnprow();
		int mb = scalapack.getmb();
		int nb = scalapack.getnb();
		int mp = scalapack.getmpa();
		int np = scalapack.getnpa();
		int mycol = scalapack.getmycol();
		int myrow = scalapack.getmyrow();
		int csrc = scalapack.getdesca()[7];
		int rsrc = scalapack.getdesca()[6];

		t_Vector A = t_Vector(mp*np);

		// Create the distributed data initially
		for (int j = 0; j < mp; ++j){
			for (int i = 0; i < np; ++i){
				A[np*j+i] = rank;
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(M*N);
		}
		scalapack.gather(Decomp, A, total_data, M, N, mp, np);

		t_Vector newA = t_Vector(mp*np);
		scalapack.scatter(Decomp, newA, total_data, M, N, mp, np);


		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(A[np*i+j]==newA[np*i+j]);
			}
		}

	}

	SECTION("Test gather A: square matrix (vector), reduced process count"){


		if(size > 3){

			mpi::Scalapack scalapack = mpi::Scalapack(true);

			int M = 256;
			int N = 256;

			scalapack.setupBlacs(Decomp, size-1, M, N);

			if(scalapack.involvedInSVD()){

				int npcol = scalapack.getnpcol();
				int nprow = scalapack.getnprow();
				int mb = scalapack.getmb();
				int nb = scalapack.getnb();
				int mp = scalapack.getmpa();
				int np = scalapack.getnpa();
				int mycol = scalapack.getmycol();
				int myrow = scalapack.getmyrow();
				int csrc = scalapack.getdesca()[7];
				int rsrc = scalapack.getdesca()[6];

				t_Vector A = t_Vector(mp*np);

				// Create the distributed data initially
				for (int j = 0; j < mp; ++j){
					for (int i = 0; i < np; ++i){
						A[np*j+i] = rank;
					}
				}

				Vector<t_real> total_data;
				if(Decomp.global_comm().is_root()){
					total_data = Vector<t_real>(M*N);
				}
				scalapack.gather(Decomp, A, total_data, M, N, mp, np);

				t_Vector newA = t_Vector(mp*np);
				scalapack.scatter(Decomp, newA, total_data, M, N, mp, np);


				for(int i=0; i<mp; i++){
					for(int j=0; j<np; j++){
						REQUIRE(A[np*i+j]==newA[np*i+j]);
					}
				}

			}

		}else{
			WARN("Test skipped because it requires at least 3 MPI processes");
		}

	}


	SECTION("Test gather scatter A: square matrix (vector)"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);


		int M = 256;
		int N = 256;

		scalapack.setupBlacs(Decomp, size, M, N);

		int npcol = scalapack.getnpcol();
		int nprow = scalapack.getnprow();
		int mb = scalapack.getmb();
		int nb = scalapack.getnb();
		int mp = scalapack.getmpa();
		int np = scalapack.getnpa();
		int mycol = scalapack.getmycol();
		int myrow = scalapack.getmyrow();
		int csrc = scalapack.getdesca()[7];
		int rsrc = scalapack.getdesca()[6];

		t_Vector A = t_Vector(mp*np);
		A.setZero();

		// Create the distributed data initially
		for (int j = 1; j <= mp; ++j){
			int gj = indxl2g_(&j,&nb,&mycol,&csrc,&npcol);
			for (int i = 1; i <= np; ++i){
				int gi = indxl2g_(&i,&mb,&myrow,&rsrc,&nprow);

				if(gi != gj) continue;
				A[(scalapack.getllda()*(j-1)+(i-1))] = gi;
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(M*N);
		}
		scalapack.gather(Decomp, A, total_data, M, N, mp, np);

		t_Vector newA = t_Vector(mp*np);
		scalapack.scatter(Decomp, newA, total_data, M, N, mp, np);

		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(newA[i*np+j] == A[i*np+j]);
			}
		}


	}

	SECTION("Test gather scatter A: square matrix"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);


		int M = 256;
		int N = 256;

		scalapack.setupBlacs(Decomp, size, M, N);

		int npcol = scalapack.getnpcol();
		int nprow = scalapack.getnprow();
		int mb = scalapack.getmb();
		int nb = scalapack.getnb();
		int mp = scalapack.getmpa();
		int np = scalapack.getnpa();
		int mycol = scalapack.getmycol();
		int myrow = scalapack.getmyrow();
		int csrc = scalapack.getdesca()[7];
		int rsrc = scalapack.getdesca()[6];

		t_Vector A = t_Vector(mp*np);
		A.setZero();

		// Create the distributed data initially
		for (int j = 1; j <= mp; ++j){
			int gj = indxl2g_(&j,&nb,&mycol,&csrc,&npcol);
			for (int i = 1; i <= np; ++i){
				int gi = indxl2g_(&i,&mb,&myrow,&rsrc,&nprow);

				if(gi != gj) continue;
				A[(scalapack.getllda()*(j-1)+(i-1))] = gi;
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(M*N);
		}
		scalapack.gather(Decomp, A, total_data, M, N, mp, np);

		Matrix<t_real> total_data_matrix(M,N);
		if(Decomp.global_comm().is_root()){
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					total_data_matrix(i,j) = total_data[i*N+j];
				}
			}
		}


		t_Vector newA = t_Vector(mp*np);
		scalapack.scatter(Decomp, newA, total_data_matrix, M, N, mp, np);

		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(newA[i*np+j] == A[i*np+j]);
			}
		}


	}

	SECTION("Test gather scatter A: non-square matrix (vector)"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 256;
		int N = 128;

		scalapack.setupBlacs(Decomp, size, M, N);

		int npcol = scalapack.getnpcol();
		int nprow = scalapack.getnprow();
		int mb = scalapack.getmb();
		int nb = scalapack.getnb();
		int mp = scalapack.getmpa();
		int np = scalapack.getnpa();
		int mycol = scalapack.getmycol();
		int myrow = scalapack.getmyrow();
		int csrc = scalapack.getdesca()[7];
		int rsrc = scalapack.getdesca()[6];

		t_Vector A = t_Vector(mp*np);
		A.setZero();

		// Create the distributed data initially
		for (int j = 1; j <= mp; ++j){
			int gj = indxl2g_(&j,&nb,&mycol,&csrc,&npcol);
			for (int i = 1; i <= np; ++i){
				int gi = indxl2g_(&i,&mb,&myrow,&rsrc,&nprow);

				if(gi != gj) continue;
				A[(scalapack.getllda()*(j-1)+(i-1))] = gi;
			}
		}

		Vector<t_real> total_data(M*N);
		scalapack.gather(Decomp, A, total_data, M, N, mp, np);

		t_Vector newA = t_Vector(mp*np);
		scalapack.scatter(Decomp, newA, total_data, M, N, mp, np);

		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(newA[i*np+j]-A[i*np+j] == Approx(0.0).margin(0.000001));
			}
		}

	}

	SECTION("Test gather scatter A: square matrix (vector), random array"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);


		int M = 256;
		int N = 256;

		scalapack.setupBlacs(Decomp, size, M, N);

		int mp = scalapack.getmpa();
		int np = scalapack.getnpa();

		t_Vector A = t_Vector(mp*np);
		A.setZero();

		// Create the distributed data initially
		for (int i = 0; i < mp; ++i){
			for (int j = 0; j < np; ++j){
				A[(np*i+j)] = rand();
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(M*N);
		}
		scalapack.gather(Decomp, A, total_data, M, N, mp, np);

		t_Vector newA = t_Vector(mp*np);
		scalapack.scatter(Decomp, newA, total_data, M, N, mp, np);

		for(int i=0; i<np; i++){
			for(int j=0; j<mp; j++){
				REQUIRE(newA[i*mp+j] == A[i*mp+j]);
			}
		}


	}

	SECTION("Test gather scatter A: non-square matrix (vector), random array"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 256;
		int N = 140;

		scalapack.setupBlacs(Decomp, size, M, N);

		int mp = scalapack.getmpa();
		int np = scalapack.getnpa();

		t_Vector A = t_Vector(mp*np);

		// Create the distributed data initially
		for (int i = 0; i < mp; ++i){
			for (int j = 0; j < np; ++j){
				A[(np*i+j)] = rand();
			}
		}

		Vector<t_real> total_data(M*N);
		scalapack.gather(Decomp, A, total_data, M, N, mp, np);

		t_Vector newA = t_Vector(mp*np);
		scalapack.scatter(Decomp, newA, total_data, M, N, mp, np);

		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(newA[i*np+j] == A[i*np+j]);
			}
		}

	}

	SECTION("Test gather scatter A: non-square matrix (vector), random array, reduced process count"){

		if(size > 3){

			mpi::Scalapack scalapack = mpi::Scalapack(true);

			int M = 256;
			int N = 140;

			scalapack.setupBlacs(Decomp, size-1, M, N);

			if(scalapack.involvedInSVD()){

				int mp = scalapack.getmpa();
				int np = scalapack.getnpa();

				t_Vector A = t_Vector(mp*np);

				// Create the distributed data initially
				for (int i = 0; i < mp; ++i){
					for (int j = 0; j < np; ++j){
						A[(np*i+j)] = rand();
					}
				}

				Vector<t_real> total_data;
				if(Decomp.global_comm().is_root()){
					total_data = Vector<t_real>(M*N);
				}
				scalapack.gather(Decomp, A, total_data, M, N, mp, np);

				t_Vector newA = t_Vector(mp*np);
				scalapack.scatter(Decomp, newA, total_data, M, N, mp, np);

				for(int i=0; i<mp; i++){
					for(int j=0; j<np; j++){
						REQUIRE(newA[i*np+j] == A[i*np+j]);
					}
				}

			}

		}else{
			WARN("Test skipped because it requires at least 3 MPI processes");
		}

	}

	SECTION("Test gather scatter U: square matrix (vector)"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);


		int M = 256;
		int N = 256;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int npcol = scalapack.getnpcol();
		int nprow = scalapack.getnprow();
		int mb = scalapack.getmb();
		int nb = scalapack.getnb();
		int mp = scalapack.getmpu();
		int np = scalapack.getnpu();
		int mycol = scalapack.getmycol();
		int myrow = scalapack.getmyrow();
		int csrc = scalapack.getdescu()[7];
		int rsrc = scalapack.getdescu()[6];

		t_Vector U = t_Vector(mp*np);
		U.setZero();

		// Create the distributed data initially
		for (int j = 1; j <= mp; ++j){
			int gj = indxl2g_(&j,&nb,&mycol,&csrc,&npcol);
			for (int i = 1; i <= np; ++i){
				int gi = indxl2g_(&i,&mb,&myrow,&rsrc,&nprow);

				if(gi != gj) continue;
				U[(scalapack.getlldu()*(j-1)+(i-1))] = gi;
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(M*minMN);
		}
		scalapack.gather(Decomp, U, total_data, M, minMN, mp, np);

		t_Vector newU = t_Vector(mp*np);
		scalapack.scatter(Decomp, newU, total_data, M, minMN, mp, np);

		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(newU[i*np+j] == U[i*np+j]);
			}
		}


	}

	SECTION("Test gather scatter U: square matrix"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 256;
		int N = 256;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int npcol = scalapack.getnpcol();
		int nprow = scalapack.getnprow();
		int mb = scalapack.getmb();
		int nb = scalapack.getnb();
		int mp = scalapack.getmpu();
		int np = scalapack.getnpu();
		int mycol = scalapack.getmycol();
		int myrow = scalapack.getmyrow();
		int csrc = scalapack.getdescu()[7];
		int rsrc = scalapack.getdescu()[6];

		t_Vector U = t_Vector(mp*np);
		U.setZero();

		// Create the distributed data initially
		for (int j = 1; j <= mp; ++j){
			int gj = indxl2g_(&j,&nb,&mycol,&csrc,&npcol);
			for (int i = 1; i <= np; ++i){
				int gi = indxl2g_(&i,&mb,&myrow,&rsrc,&nprow);

				if(gi != gj) continue;
				U[(scalapack.getlldu()*(j-1)+(i-1))] = gi;
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(M*minMN);
		}
		scalapack.gather(Decomp, U, total_data, M, minMN, mp, np);

		Matrix<t_real> total_data_matrix;
		if(Decomp.global_comm().is_root()){
			total_data_matrix = Matrix<t_real>(M, minMN);
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < minMN; ++j){
					total_data_matrix(i,j) = total_data[i*N+j];
				}
			}
		}


		t_Vector newU = t_Vector(mp*np);
		scalapack.scatter(Decomp, newU, total_data_matrix, M, minMN, mp, np);

		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(newU[i*np+j] == U[i*np+j]);
			}
		}


	}

	SECTION("Test gather scatter U: non-square matrix (vector)"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 256;
		int N = 128;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int npcol = scalapack.getnpcol();
		int nprow = scalapack.getnprow();
		int mb = scalapack.getmb();
		int nb = scalapack.getnb();
		int mp = scalapack.getmpu();
		int np = scalapack.getnpu();
		int mycol = scalapack.getmycol();
		int myrow = scalapack.getmyrow();
		int csrc = scalapack.getdescu()[7];
		int rsrc = scalapack.getdescu()[6];

		t_Vector U = t_Vector(mp*np);
		U.setZero();

		// Create the distributed data initially
		for (int j = 1; j <= mp; ++j){
			int gj = indxl2g_(&j,&nb,&mycol,&csrc,&npcol);
			for (int i = 1; i <= np; ++i){
				int gi = indxl2g_(&i,&mb,&myrow,&rsrc,&nprow);

				if(gi != gj) continue;
				U[(scalapack.getlldu()*(j-1)+(i-1))] = gi;
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(M*minMN);
		}
		scalapack.gather(Decomp, U, total_data, M, minMN, mp, np);

		t_Vector newU = t_Vector(mp*np);
		scalapack.scatter(Decomp, newU, total_data, M, minMN, mp, np);

		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(newU[i*np+j] == U[i*np+j]);
			}
		}

	}

	SECTION("Test gather scatter U: square matrix (vector), random array"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 256;
		int N = 256;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int mp = scalapack.getmpu();
		int np = scalapack.getnpu();

		t_Vector U = t_Vector(mp*np);
		U.setZero();

		// Create the distributed data initially
		for (int i = 0; i < mp; ++i){
			for (int j = 0; j < np; ++j){
				U[(np*i+j)] = rand();
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(M*minMN);
		}
		scalapack.gather(Decomp, U, total_data, M, minMN, mp, np);

		t_Vector newU = t_Vector(mp*np);
		scalapack.scatter(Decomp, newU, total_data, M, minMN, mp, np);

		for(int i=0; i<np; i++){
			for(int j=0; j<mp; j++){
				REQUIRE(newU[i*mp+j] == U[i*mp+j]);
			}
		}


	}

	SECTION("Test gather scatter U: non-square matrix (vector), random array"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 256;
		int N = 140;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int mp = scalapack.getmpu();
		int np = scalapack.getnpu();

		t_Vector U = t_Vector(mp*np);

		// Create the distributed data initially
		for (int i = 0; i < mp; ++i){
			for (int j = 0; j < np; ++j){
				U[(np*i+j)] = rand();
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(M*minMN);
		}
		scalapack.gather(Decomp, U, total_data, M, minMN, mp, np);

		t_Vector newU = t_Vector(mp*np);
		scalapack.scatter(Decomp, newU, total_data, M, minMN, mp, np);

		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(newU[i*np+j] == U[i*np+j]);
			}
		}


	}

	SECTION("Test gather scatter VT: square matrix (vector)"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);


		int M = 256;
		int N = 256;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int npcol = scalapack.getnpcol();
		int nprow = scalapack.getnprow();
		int mb = scalapack.getmb();
		int nb = scalapack.getnb();
		int mp = scalapack.getmpvt();
		int np = scalapack.getnpvt();
		int mycol = scalapack.getmycol();
		int myrow = scalapack.getmyrow();
		int csrc = scalapack.getdescvt()[7];
		int rsrc = scalapack.getdescvt()[6];

		t_Vector VT = t_Vector(mp*np);
		VT.setZero();

		// Create the distributed data initially
		for (int j = 1; j <= mp; ++j){
			int gj = indxl2g_(&j,&nb,&mycol,&csrc,&npcol);
			for (int i = 1; i <= np; ++i){
				int gi = indxl2g_(&i,&mb,&myrow,&rsrc,&nprow);

				if(gi != gj) continue;
				VT[(scalapack.getlldvt()*(j-1)+(i-1))] = gi;
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(minMN*N);
		}
		scalapack.gather(Decomp, VT, total_data, minMN, N, mp, np);

		t_Vector newVT = t_Vector(mp*np);
		scalapack.scatter(Decomp, newVT, total_data, minMN, N, mp, np);

		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(newVT[i*np+j] == VT[i*np+j]);
			}
		}


	}

	SECTION("Test gather scatter VT: square matrix"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 256;
		int N = 256;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int npcol = scalapack.getnpcol();
		int nprow = scalapack.getnprow();
		int mb = scalapack.getmb();
		int nb = scalapack.getnb();
		int mp = scalapack.getmpvt();
		int np = scalapack.getnpvt();
		int mycol = scalapack.getmycol();
		int myrow = scalapack.getmyrow();
		int csrc = scalapack.getdescvt()[7];
		int rsrc = scalapack.getdescvt()[6];

		t_Vector VT = t_Vector(mp*np);
		VT.setZero();

		// Create the distributed data initially
		for (int j = 1; j <= mp; ++j){
			int gj = indxl2g_(&j,&nb,&mycol,&csrc,&npcol);
			for (int i = 1; i <= np; ++i){
				int gi = indxl2g_(&i,&mb,&myrow,&rsrc,&nprow);

				if(gi != gj) continue;
				VT[(scalapack.getlldvt()*(j-1)+(i-1))] = gi;
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(minMN*N);
		}
		scalapack.gather(Decomp, VT, total_data, minMN, N, mp, np);

		Matrix<t_real> total_data_matrix;
		if(Decomp.global_comm().is_root()){
			total_data_matrix = Matrix<t_real>(minMN,N);
			for (int i = 0; i < minMN; ++i){
				for (int j = 0; j < N; ++j){
					total_data_matrix(i,j) = total_data[i*N+j];
				}
			}
		}


		t_Vector newVT = t_Vector(mp*np);
		scalapack.scatter(Decomp, newVT, total_data_matrix, minMN, N, mp, np);

		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(newVT[i*np+j] == VT[i*np+j]);
			}
		}


	}

	SECTION("Test gather scatter VT: non-square matrix (vector)"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 256;
		int N = 128;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int npcol = scalapack.getnpcol();
		int nprow = scalapack.getnprow();
		int mb = scalapack.getmb();
		int nb = scalapack.getnb();
		int mp = scalapack.getmpvt();
		int np = scalapack.getnpvt();
		int mycol = scalapack.getmycol();
		int myrow = scalapack.getmyrow();
		int csrc = scalapack.getdescvt()[7];
		int rsrc = scalapack.getdescvt()[6];

		t_Vector VT = t_Vector(mp*np);
		VT.setZero();

		// Create the distributed data initially
		for (int j = 1; j <= mp; ++j){
			int gj = indxl2g_(&j,&nb,&mycol,&csrc,&npcol);
			for (int i = 1; i <= np; ++i){
				int gi = indxl2g_(&i,&mb,&myrow,&rsrc,&nprow);

				if(gi != gj) continue;
				VT[(scalapack.getlldvt()*(j-1)+(i-1))] = gi;
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(minMN*N);
		}
		scalapack.gather(Decomp, VT, total_data, minMN, N, mp, np);

		t_Vector newVT = t_Vector(mp*np);
		scalapack.scatter(Decomp, newVT, total_data, minMN, N, mp, np);

		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(newVT[i*np+j] == VT[i*np+j]);
			}
		}

	}

	SECTION("Test gather scatter VT: square matrix (vector), random array"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 256;
		int N = 256;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int mp = scalapack.getmpvt();
		int np = scalapack.getnpvt();

		t_Vector VT = t_Vector(mp*np);
		VT.setZero();

		// Create the distributed data initially
		for (int i = 0; i < mp; ++i){
			for (int j = 0; j < np; ++j){
				VT[(np*i+j)] = rand();
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(minMN*N);
		}
		scalapack.gather(Decomp, VT, total_data, minMN, N, mp, np);

		t_Vector newVT = t_Vector(mp*np);
		scalapack.scatter(Decomp, newVT, total_data, minMN, N, mp, np);

		for(int i=0; i<np; i++){
			for(int j=0; j<mp; j++){
				REQUIRE(newVT[i*mp+j] == VT[i*mp+j]);
			}
		}


	}

	SECTION("Test gather scatter VT: non-square matrix (vector), random array"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 256;
		int N = 140;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int mp = scalapack.getmpvt();
		int np = scalapack.getnpvt();

		t_Vector VT = t_Vector(mp*np);

		// Create the distributed data initially
		for (int i = 0; i < mp; ++i){
			for (int j = 0; j < np; ++j){
				VT[(np*i+j)] = rand();
			}
		}

		Vector<t_real> total_data;
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(minMN*N);
		}
		scalapack.gather(Decomp, VT, total_data, minMN, N, mp, np);

		t_Vector newVT = t_Vector(mp*np);
		scalapack.scatter(Decomp, newVT, total_data, minMN, N, mp, np);

		for(int i=0; i<mp; i++){
			for(int j=0; j<np; j++){
				REQUIRE(newVT[i*np+j] == VT[i*np+j]);
			}
		}

	}

	SECTION("Test SVD compared to Eigen (vector)"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 256;
		int N = 256;

		scalapack.setupBlacs(Decomp, size, M, N);

		int mp = scalapack.getmpa();
		int np = scalapack.getnpa();

		Vector<t_real> total_data;
		Matrix<t_real> eigen_data;

		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(M*N);
			eigen_data = Matrix<t_real>(M,N);
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					total_data[N*i+j] = (M*j+i);
					eigen_data(i,j) = (M*j+i);
				}
			}
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					REQUIRE(total_data[N*i+j] == eigen_data(i,j));
				}
			}

		}

		Vector<t_real> A(mp*np);
		Vector<t_real> results(std::min(M,N));

		scalapack.scatter(Decomp, A, total_data, M, N, mp, np);

		scalapack.setupSVD(A, results);
		scalapack.runSVD(A, results);


		if(Decomp.global_comm().is_root()){

			typename Eigen::BDCSVD<Matrix<t_real>> svd(eigen_data, Eigen::ComputeThinU | Eigen::ComputeThinV);
			auto s = svd.singularValues();

			for(int i=0; i<std::min(M,N); i++){
				// This formulation is required to deal with 0.0 == 0.0 comparisons where
				// catch2 Approx does not work well.
				REQUIRE(results[i]-s[i] == Approx(0.0).margin(0.000001));
			}

		}

	}

	SECTION("Test SVD compared to Eigen with U and VT (vector)"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 8*size;
		int N = 8*size;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int mpa = scalapack.getmpa();
		int npa = scalapack.getnpa();
		int mpu = scalapack.getmpu();
		int npu = scalapack.getnpu();
		int mpvt = scalapack.getmpvt();
		int npvt = scalapack.getnpvt();

		Vector<t_real> total_data;
		Matrix<t_real> eigen_data;

		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(M*N);
			eigen_data = Matrix<t_real>(M,N);
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					total_data[N*i+j] = (M*j+i);
					eigen_data(i,j) = (M*j+i);
				}
			}
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					REQUIRE(total_data[N*i+j] == eigen_data(i,j));
				}
			}

		}

		Vector<t_real> A(mpa*npa);
		Vector<t_real> U(mpu*npu);
		Vector<t_real> VT(mpvt*npvt);
		Vector<t_real> results(minMN);

		scalapack.scatter(Decomp, A, total_data, M, N, mpa, npa);

		scalapack.setupSVD(A, results, U, VT);
		scalapack.runSVD(A, results, U, VT);

		Vector<t_real> total_U;
		Vector<t_real> total_VT;
		if(Decomp.global_comm().is_root()){
			total_U = Vector<t_real>(M*minMN);
			total_VT = Vector<t_real>(minMN*N);
		}

		scalapack.gather(Decomp, U, total_U, M, minMN, mpu, npu);
		scalapack.gather(Decomp, VT, total_VT, minMN, N, mpvt, npvt);


		if(Decomp.global_comm().is_root()){

			typename Eigen::BDCSVD<Matrix<t_real>> svd(eigen_data, Eigen::ComputeThinU | Eigen::ComputeThinV);
			auto s = svd.singularValues();
			auto u = svd.matrixU();
			auto vt = svd.matrixV();

			for(int i=0; i<std::min(M,N); i++){
				// This formulation is required to deal with 0.0 == 0.0 comparisons where
				// catch2 Approx does not work well.
				REQUIRE(results[i]-s[i] == Approx(0.0).margin(0.000001));
			}

			Matrix<t_real> matrix_U = Eigen::Map<Matrix<t_real>>(total_U.data(), M, minMN);
			Matrix<t_real> matrix_VT = Eigen::Map<Matrix<t_real>>(total_VT.data(), minMN, N);

			auto const scalapack_result = (matrix_U * results.asDiagonal() * matrix_VT).transpose();
			auto const eigen_result = u * s.asDiagonal() * vt.transpose();

			for(int i=0; i<M; i++){
				for(int j=0; j<N; j++){
					// This formulation is required to deal with 0.0 == 0.0 comparisons where
					// catch2 Approx does not work well.
					REQUIRE(scalapack_result(i,j) - eigen_result(i,j) == Approx(0.0).margin(0.00001));
				}
			}

		}

	}

	SECTION("Test SVD compared to Eigen with U and VT (vector) non-square"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 128;
		int N = 256;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int mpa = scalapack.getmpa();
		int npa = scalapack.getnpa();
		int mpu = scalapack.getmpu();
		int npu = scalapack.getnpu();
		int mpvt = scalapack.getmpvt();
		int npvt = scalapack.getnpvt();

		Vector<t_real> total_data;
		Matrix<t_real> eigen_data;


		// Here the total_data indexing is different (M*j+i instead of N*i+j) because the
		// underlying Scalapack routines are Fortran so order 2d array data different to C/C++.
		// For the previous tests the matrices were square so this was not an issue.
		if(Decomp.global_comm().is_root()){
			total_data = Vector<t_real>(M*N);
			eigen_data = Matrix<t_real>(M,N);
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					total_data[M*j+i] = (M*j+i);
					eigen_data(i,j) = (M*j+i);
				}
			}
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					REQUIRE(total_data[M*j+i] == eigen_data(i,j));
				}
			}

		}

		Vector<t_real> A(mpa*npa);
		Vector<t_real> U(mpu*npu);
		Vector<t_real> VT(mpvt*npvt);
		Vector<t_real> results(minMN);

		scalapack.scatter(Decomp, A, total_data, M, N, mpa, npa);

		scalapack.setupSVD(A, results, U, VT);
		scalapack.runSVD(A, results, U, VT);

		Vector<t_real> total_U;
		Vector<t_real> total_VT;
		if(Decomp.global_comm().is_root()){
			total_U = Vector<t_real>(M*minMN);
			total_VT = Vector<t_real>(minMN*N);
		}

		scalapack.gather(Decomp, U, total_U, M, minMN, mpu, npu);
		scalapack.gather(Decomp, VT, total_VT, minMN, N, mpvt, npvt);


		if(Decomp.global_comm().is_root()){

			typename Eigen::BDCSVD<Matrix<t_real>> svd(eigen_data, Eigen::ComputeThinU | Eigen::ComputeThinV);
			auto s = svd.singularValues();
			auto u = svd.matrixU();
			auto vt = svd.matrixV();

			for(int i=0; i<std::min(M,N); i++){
				// This formulation is required to deal with 0.0 == 0.0 comparisons where
				// catch2 Approx does not work well.
				REQUIRE(results[i]-s[i] == Approx(0.0).margin(0.000001));
			}

			Matrix<t_real> matrix_U = Eigen::Map<Matrix<t_real>>(total_U.data(), M, minMN);
			Matrix<t_real> matrix_VT = Eigen::Map<Matrix<t_real>>(total_VT.data(), minMN, N);

			auto const scalapack_result = matrix_U * results.asDiagonal() * matrix_VT;
			auto const eigen_result = u * s.asDiagonal() * vt.transpose();

			for(int i=0; i<M; i++){
				for(int j=0; j<N; j++){
					// This formulation is required to deal with 0.0 == 0.0 comparisons where
					// catch2 Approx does not work well.
					REQUIRE(scalapack_result(i,j) - eigen_result(i,j) == Approx(0.0).margin(0.00001));
				}
			}

		}

	}

	SECTION("Test proximal operator of the nuclear norm compared to Eigen (vector) non-square"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 128;
		int N = 256;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int mpa = scalapack.getmpa();
		int npa = scalapack.getnpa();
		int mpu = scalapack.getmpu();
		int npu = scalapack.getnpu();
		int mpvt = scalapack.getmpvt();
		int npvt = scalapack.getnpvt();

		Vector<t_real> total_data;
		Matrix<t_real> eigen_data;

		// Here the total_data indexing is different (M*j+i instead of N*i+j) because the
		// underlying Scalapack routines are Fortran so order 2d array data different to C/C++.
		// For the previous tests the matrices were square so this was not an issue.
		if(Decomp.global_comm().is_root()){
			eigen_data = Matrix<t_real>(M,N);
			total_data = Vector<t_real>(M*N);
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					total_data[M*j+i] = (M*j+i);
					eigen_data(i,j) = (M*j+i);
				}
			}
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					REQUIRE(total_data[M*j+i] == eigen_data(i,j));
				}
			}

		}

		Vector<t_real> A(mpa*npa);
		Vector<t_real> U(mpu*npu);
		Vector<t_real> VT(mpvt*npvt);
		Vector<t_real> results(minMN);

		scalapack.scatter(Decomp, A, total_data, M, N, mpa, npa);

		scalapack.setupSVD(A, results, U, VT);
		scalapack.runSVD(A, results, U, VT);

		Vector<t_real> total_U;
		Vector<t_real> total_VT;

		if(Decomp.global_comm().is_root()){
			total_U = Vector<t_real>(M*minMN);
			total_VT = Vector<t_real>(minMN*N);
		}

		scalapack.gather(Decomp, U, total_U, M, minMN, mpu, npu);
		scalapack.gather(Decomp, VT, total_VT, minMN, N, mpvt, npvt);

		if(Decomp.global_comm().is_root()){

			t_real threshold = 2.;
			typename Eigen::BDCSVD<Matrix<t_real>> svd(eigen_data, Eigen::ComputeThinU | Eigen::ComputeThinV);
			auto s = svd.singularValues();
			auto u = svd.matrixU();
			auto v = svd.matrixV();

			for(int i=0; i<std::min(M,N); i++){
				// This formulation is required to deal with 0.0 == 0.0 comparisons where
				// catch2 Approx does not work well.
				REQUIRE(results[i]-s[i] == Approx(0.0).margin(0.000001));
			}

			Matrix<t_real> matrix_U = Eigen::Map<Matrix<t_real>>(total_U.data(), M, minMN);
			Matrix<t_real> matrix_VT = Eigen::Map<Matrix<t_real>>(total_VT.data(), minMN, N);

			Vector<t_real> thresholded_s = psi::soft_threshhold(s, threshold);
			Vector<t_real> thresholded_results = psi::soft_threshhold(results, threshold);

			for(int i=0; i<std::min(M,N); i++){
				// This formulation is required to deal with 0.0 == 0.0 comparisons where
				// catch2 Approx does not work well.
				REQUIRE(thresholded_s[i]-thresholded_results[i] == Approx(0.0).margin(0.000001));
			}

			auto const scalapack_result = matrix_U * thresholded_results.asDiagonal() * matrix_VT;
			auto const eigen_result = u * thresholded_s.asDiagonal() * v.transpose();

			Matrix<t_real> output_eigen_prox(M, N);
			psi::proximal::nuclear_norm(output_eigen_prox, eigen_data, threshold);

			for(int i=0; i<M; i++){
				for(int j=0; j<N; j++){
					// This formulation is required to deal with 0.0 == 0.0 comparisons where
					// catch2 Approx does not work well.
					REQUIRE(scalapack_result(i,j) - eigen_result(i,j) == Approx(0.0).margin(0.00001));
					REQUIRE(output_eigen_prox(i,j) - eigen_result(i,j) == Approx(0.0).margin(0.00001));
				}
			}

		}

	}

	SECTION("Test proximal operator of the nuclear norm (comparing Eigen and Scalapack versions)"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 128;
		int N = 256;

		int minMN = std::min(M,N);

		scalapack.setupBlacs(Decomp, size, M, N);

		int mpa = scalapack.getmpa();
		int npa = scalapack.getnpa();
		int mpu = scalapack.getmpu();
		int npu = scalapack.getnpu();
		int mpvt = scalapack.getmpvt();
		int npvt = scalapack.getnpvt();

		Vector<t_real> total_data;
		Matrix<t_real> eigen_data;

		// Here the total_data indexing is different (M*j+i instead of N*i+j) because the
		// underlying Scalapack routines are Fortran so order 2d array data different to C/C++.
		// For the previous tests the matrices were square so this was not an issue.
		if(Decomp.global_comm().is_root()){
			eigen_data = Matrix<t_real>(M,N);
			total_data = Vector<t_real>(M*N);
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					total_data[M*j+i] = (M*j+i);
					eigen_data(i,j) = (M*j+i);
				}
			}
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					REQUIRE(total_data[M*j+i] == eigen_data(i,j));
				}
			}

		}

		Vector<t_real> A(mpa*npa);
		Vector<t_real> U(mpu*npu);
		Vector<t_real> VT(mpvt*npvt);
		Vector<t_real> results(minMN);

		scalapack.scatter(Decomp, A, eigen_data, M, N, mpa, npa);

		scalapack.setupSVD(A, results, U, VT);
		scalapack.runSVD(A, results, U, VT);

		Vector<t_real> total_U;
		Vector<t_real> total_VT;

		if(Decomp.global_comm().is_root()){
			total_U = Vector<t_real>(M*minMN);
			total_VT = Vector<t_real>(minMN*N);
		}

		scalapack.gather(Decomp, U, total_U, M, minMN, mpu, npu);
		scalapack.gather(Decomp, VT, total_VT, minMN, N, mpvt, npvt);

		if(Decomp.global_comm().is_root()){

			t_real threshold = 2.;
			typename Eigen::BDCSVD<Matrix<t_real>> svd(eigen_data, Eigen::ComputeThinU | Eigen::ComputeThinV);
			auto s = svd.singularValues();
			auto u = svd.matrixU();
			auto v = svd.matrixV();

			for(int i=0; i<std::min(M,N); i++){
				// This formulation is required to deal with 0.0 == 0.0 comparisons where
				// catch2 Approx does not work well.
				REQUIRE(results[i]-s[i] == Approx(0.0).margin(0.000001));
			}

			Matrix<t_real> matrix_U = Eigen::Map<Matrix<t_real>>(total_U.data(), M, minMN);
			Matrix<t_real> matrix_VT = Eigen::Map<Matrix<t_real>>(total_VT.data(), minMN, N);

			Vector<t_real> thresholded_s = psi::soft_threshhold(s, threshold);
			Vector<t_real> thresholded_results = psi::soft_threshhold(results, threshold);

			for(int i=0; i<std::min(M,N); i++){
				// This formulation is required to deal with 0.0 == 0.0 comparisons where
				// catch2 Approx does not work well.
				REQUIRE(thresholded_s[i]-thresholded_results[i] == Approx(0.0).margin(0.000001));
			}

			auto const scalapack_result = matrix_U * thresholded_results.asDiagonal() * matrix_VT;
			auto const eigen_result = u * thresholded_s.asDiagonal() * v.transpose();

			Matrix<t_real> output_eigen_prox(M, N);
			psi::proximal::nuclear_norm(output_eigen_prox, eigen_data, threshold);

			for(int i=0; i<M; i++){
				for(int j=0; j<N; j++){
					// This formulation is required to deal with 0.0 == 0.0 comparisons where
					// catch2 Approx does not work well.
					REQUIRE(scalapack_result(i,j) - eigen_result(i,j) == Approx(0.0).margin(0.00001));
					REQUIRE(output_eigen_prox(i,j) - eigen_result(i,j) == Approx(0.0).margin(0.00001));
				}
			}

		}

	}

	SECTION("Test non-square SVD compared to Eigen (vector)"){

		mpi::Scalapack scalapack = mpi::Scalapack(true);

		int M = 256;
		int N = 128;

		scalapack.setupBlacs(Decomp, size, M, N);

		int mp = scalapack.getmpa();
		int np = scalapack.getnpa();

		Vector<t_real> total_data;
		Matrix<t_real> eigen_data;

		// Here the total_data indexing is different (M*j+i instead of N*i+j) because the
		// underlying Scalapack routines are Fortran so order 2d array data different to C/C++.
		// For the previous tests the matrices were square so this was not an issue.
		if(Decomp.global_comm().is_root()){
			eigen_data = Matrix<t_real>(M,N);
			total_data = Vector<t_real>(M*N);
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					total_data[M*j+i] = (M*j+i);
					eigen_data(i,j) = (M*j+i);
				}
			}
			for (int i = 0; i < M; ++i){
				for (int j = 0; j < N; ++j){
					REQUIRE(total_data[M*j+i] == eigen_data(i,j));
				}
			}

		}

		Vector<t_real> A(mp*np);
		Vector<t_real> results(std::min(M,N));

		scalapack.scatter(Decomp, A, total_data, M, N, mp, np);

		scalapack.setupSVD(A, results);
		scalapack.runSVD(A, results);

		if(Decomp.global_comm().is_root()){

			typename Eigen::BDCSVD<Matrix<t_real>> svd(eigen_data, Eigen::ComputeThinU | Eigen::ComputeThinV);
			auto s = svd.singularValues();

			for(int i=0; i<std::min(M,N); i++){
				// This formulation is required to deal with 0.0 == 0.0 comparisons where
				// catch2 Approx does not work well.
				REQUIRE(results[i]-s[i] == Approx(0.0).margin(0.000001));
			}

		}

	}

	SECTION("Test non-square SVD compared to Eigen (vector), reduced process count"){

		if(size > 3){
			mpi::Scalapack scalapack = mpi::Scalapack(true);

			int M = 256;
			int N = 128;

			scalapack.setupBlacs(Decomp, size-1, M, N);

			if(scalapack.involvedInSVD()){

				int mp = scalapack.getmpa();
				int np = scalapack.getnpa();

				Vector<t_real> total_data;
				Matrix<t_real> eigen_data;

				// Here the total_data indexing is different (M*j+i instead of N*i+j) because the
				// underlying Scalapack routines are Fortran so order 2d array data different to C/C++.
				// For the previous tests the matrices were square so this was not an issue.
				if(Decomp.global_comm().is_root()){
					eigen_data = Matrix<t_real>(M,N);
					total_data = Vector<t_real>(M*N);
					for (int i = 0; i < M; ++i){
						for (int j = 0; j < N; ++j){
							total_data[M*j+i] = (M*j+i);
							eigen_data(i,j) = (M*j+i);
						}
					}
					for (int i = 0; i < M; ++i){
						for (int j = 0; j < N; ++j){
							REQUIRE(total_data[M*j+i] == eigen_data(i,j));
						}
					}

				}

				Vector<t_real> A(mp*np);
				Vector<t_real> results(std::min(M,N));

				scalapack.scatter(Decomp, A, total_data, M, N, mp, np);

				scalapack.setupSVD(A, results);
				scalapack.runSVD(A, results);

				if(Decomp.global_comm().is_root()){

					typename Eigen::BDCSVD<Matrix<t_real>> svd(eigen_data, Eigen::ComputeThinU | Eigen::ComputeThinV);
					auto s = svd.singularValues();

					for(int i=0; i<std::min(M,N); i++){
						// This formulation is required to deal with 0.0 == 0.0 comparisons where
						// catch2 Approx does not work well.
						REQUIRE(results[i]-s[i] == Approx(0.0).margin(0.000001));
					}

				}
			}

		}else{
			WARN("Test skipped because it requires at least 3 MPI processes");
		}
	}

	SECTION("Test sendToScalapackRoot and recvFromScalapackRoot"){

		if(size > 3){

			auto const world = mpi::Communicator::World(1);
			bool parallel = true;
			auto Decomp = mpi::Decomposition(parallel, world);

			mpi::Scalapack scalapack = mpi::Scalapack(true);

			int M = 128;
			int N = 256;

			scalapack.setupBlacs(Decomp, size, M, N);

			if(scalapack.involvedInSVD() or Decomp.global_comm().is_root()){

				int mp;

				if(scalapack.involvedInSVD()){
					mp = scalapack.getmpa();
				}
				int np;
				if(scalapack.involvedInSVD()){
					np = scalapack.getnpa();
				}

				Vector<t_real> total_data;

				if(Decomp.global_comm().is_root()){
					total_data = Vector<t_real>(M*N);
					for (int i = 0; i < M; ++i){
						for (int j = 0; j < N; ++j){
							total_data[N*i+j] = std::real(M*j+i);
						}
					}
				}

				bool am_global_root = Decomp.global_comm().is_root();
				bool am_scalapack_root = scalapack.scalapack_comm().is_root();

				REQUIRE(not (am_global_root and am_scalapack_root));

				if(not Decomp.global_comm().is_root() and (scalapack.involvedInSVD() and scalapack.scalapack_comm().is_root())){
					total_data = Vector<t_real>(M*N);
				}

				if(Decomp.global_comm().is_root() or (scalapack.involvedInSVD() and scalapack.scalapack_comm().is_root())){
					scalapack.sendToScalapackRoot(Decomp, total_data);
				}

				if(not Decomp.global_comm().is_root() and (scalapack.involvedInSVD() and scalapack.scalapack_comm().is_root())){
					for (int i = 0; i < M; ++i){
						for (int j = 0; j < N; ++j){
							CHECK(total_data[N*i+j] == Approx(std::real(M*j+i)));
						}
					}
				}

				Vector<t_real> new_total_data(M*N);

				if(not Decomp.global_comm().is_root() and (scalapack.involvedInSVD() and scalapack.scalapack_comm().is_root())){
					for (int i = 0; i < M; ++i){
						for (int j = 0; j < N; ++j){
							new_total_data[N*i+j] = total_data[N*i+j];
						}
					}
				}

				if(Decomp.global_comm().is_root() or (scalapack.involvedInSVD() and scalapack.scalapack_comm().is_root())){
					scalapack.recvFromScalapackRoot(Decomp, new_total_data);
				}

				if(Decomp.global_comm().is_root()){
					for (int i = 0; i < M; ++i){
						for (int j = 0; j < N; ++j){
							CHECK(std::real(new_total_data[N*i+j]) == Approx(total_data[N*i+j]));
						}
					}
				}

			}

		}else{
			WARN("Test skipped because it requires at least 3 MPI processes");
		}
	}




}
#endif
