#include <iostream>
#include <numeric>
#include <cmath>
#include <complex>
#include <catch2/catch.hpp>

#include "psi/config.h"
#include "psi/mpi/communicator.h"

using namespace psi;

#ifdef PSI_MPI
TEST_CASE("Creates a mpi communicator") {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	auto const world = mpi::Communicator::World();

	SECTION("General stuff") {
		REQUIRE(*world == MPI_COMM_WORLD);
		REQUIRE(static_cast<t_int>(world.rank()) == rank);
		REQUIRE(static_cast<t_int>(world.size()) == size);

		mpi::Communicator shallow = world;
		CHECK(*shallow == *world);
	}

	SECTION("Duplicate") {
		mpi::Communicator dup = world.duplicate();
		CHECK(*dup != *world);
	}

	SECTION("Split") {
		mpi::Communicator split_comm = mpi::Communicator(world.split(world.rank()/2));
		CHECK(*split_comm != *world);
	}

	SECTION("Scatter one") {
		if(world.rank() == world.root_id()) {
			std::vector<t_int> scattered(world.size());
			std::iota(scattered.begin(), scattered.end(), 2);
			auto const result = world.scatter_one(scattered);
			CHECK(result == world.rank() + 2);
		} else {
			auto const result = world.scatter_one<t_int>();
			CHECK(result == world.rank() + 2);
		}
	}


	SECTION("Scatter columns of matrix (int)") {
		int perproc = 2;
		if(world.rank() == world.root_id()) {
			Matrix<t_int> scattered(10, world.size()*perproc);
			for(int i=0; i<scattered.cols(); i++){
				for(int j=0; j<scattered.rows(); j++){
					scattered(j,i) = i/perproc;
				}
			}
			auto const result = world.scatter_eigen_simple_columns<t_int>(scattered,perproc);
			for(int j=0; j<result.cols(); j++){
				for(int i=0; i<result.rows(); i++){
					CHECK(result(i,j) == world.rank());
				}
			}
		} else {
			auto const result = world.scatter_eigen_simple_columns<t_int>(perproc);
			for(int j=0; j<result.cols(); j++){
				for(int i=0; i<result.rows(); i++){
					CHECK(result(i,j) == world.rank());
				}
			}
		}
	}

	SECTION("Scatter columns of matrix (real)") {
		int perproc = 2;
		if(world.rank() == world.root_id()) {
			Matrix<t_real> scattered(10, world.size()*perproc);
			for(int i=0; i<scattered.cols(); i++){
				for(int j=0; j<scattered.rows(); j++){
					scattered(j,i) = double(i/perproc);
				}
			}
			auto const result = world.scatter_eigen_simple_columns<t_real>(scattered,perproc);
			for(int j=0; j<result.cols(); j++){
				for(int i=0; i<result.rows(); i++){
					CHECK(result(i,j) == double(world.rank()));
				}
			}
		} else {
			auto const result = world.scatter_eigen_simple_columns<t_real>(perproc);
			for(int j=0; j<result.cols(); j++){
				for(int i=0; i<result.rows(); i++){
					CHECK(result(i,j) == double(world.rank()));
				}
			}
		}
	}



	SECTION("ScatterV") {
		std::vector<t_int> sizes(world.size()), displs(world.size());
		for(t_uint i(0); i < world.rank(); ++i)
			sizes[i] = world.rank() * 2 + i;
		for(t_uint i(1); i < world.rank(); ++i)
			displs[i] = displs[i - 1] + sizes[i - 1];
		Vector<t_int> const sendee
		= Vector<t_int>::Random(std::accumulate(sizes.begin(), sizes.end(), 0));
		auto const result = world.rank() == world.root_id() ?
				world.scatterv(sendee, sizes) :
				world.scatterv<t_int>(sizes[world.rank()]);
		CHECK(result.isApprox(sendee.segment(displs[world.rank()], sizes[world.rank()])));
	}

	SECTION("Gather a single item") {
		if(world.rank() == world.root_id()) {
			std::vector<t_int> scattered(world.size());
			std::iota(scattered.begin(), scattered.end(), 2);
			auto const result = world.scatter_one(scattered);
			REQUIRE(result == world.rank() + 2);
			auto const gathered = world.gather(result);
			for(decltype(gathered)::size_type i = 0; i < gathered.size(); i++)
				CHECK(gathered[i] == scattered[i]);
		} else {
			auto const result = world.scatter_one<t_int>();
			REQUIRE(result == world.rank() + 2);
			auto const gather = world.gather(result);
			CHECK(gather.size() == 0);
		}
	}

	SECTION("Gather an eigen vector") {
		auto const size = [](t_int n) { return n * 2 + 10; };
		auto const totsize = [](t_int n) { return std::max(0, n * (9 + n)); };
		Vector<t_int> const sendee = Vector<t_int>::Constant(size(world.rank()), world.rank());
		std::vector<t_int> sizes(world.size());
		int n(0);
		std::generate(sizes.begin(), sizes.end(), [&n, &size]() { return size(n++); });

		auto const result = world.is_root() ? world.gather(sendee, sizes) : world.gather(sendee);
		if(world.rank() == world.root_id()) {
			CHECK(result.size() == totsize(world.size()));
			for(decltype(world.size()) i(0); i < world.size(); ++i)
				CHECK(result.segment(totsize(i), size(i)) == Vector<t_int>::Constant(size(i), i));
		} else
			CHECK(result.size() == 0);
	}

	SECTION("Gather an std::set") {
		std::set<t_int> const input{static_cast<t_int>(world.size()), static_cast<t_int>(world.rank())};
		auto const result = world.gather(input, world.gather<t_int>(input.size()));
		if(world.is_root()) {
			CHECK(result.size() == world.size() + 1);
			for(decltype(world.size()) i(0); i <= world.size(); ++i)
				CHECK(result.count(i) == 1);
		} else
			CHECK(result.size() == 0);
	}

	SECTION("Gather an std::vector") {
		std::vector<t_int> const input{static_cast<t_int>(world.size()), static_cast<t_int>(world.rank())};
		auto const result = world.gather(input, world.gather<t_int>(input.size()));
		if(world.is_root()) {
			CHECK(result.size() == world.size() * 2);
			for(decltype(world.size()) i(0); i < world.size(); ++i) {
				CHECK(result[2 * i] == world.size());
				CHECK(result[2 *i + 1] == i);
			}
		} else
			CHECK(result.size() == 0);
	}

	SECTION("Gather columns of matrix (int)") {
		int perproc = 2;
		Matrix<t_int> scattered(10, perproc);
		for(int i=0; i<scattered.cols(); i++){
			for(int j=0; j<scattered.rows(); j++){
				scattered(j,i) = world.rank();
			}
		}
		auto const result = world.gather_eigen_simple_columns<t_int>(scattered,perproc);
		if(world.rank() == world.root_id()) {
			for(int i=0; i<scattered.cols(); i++){
				for(int j=0; j<scattered.rows(); j++){
					CHECK(scattered(j,i) == i/perproc);
				}
			}
		} else {
			CHECK(result.size() == 0);
		}
	}

	SECTION("Gather columns of matrix (real)") {
		int perproc = 7;
		Matrix<t_real> scattered(10, perproc);
		for(int i=0; i<scattered.cols(); i++){
			for(int j=0; j<scattered.rows(); j++){
				scattered(j,i) = double(world.rank());
			}
		}
		auto const result = world.gather_eigen_simple_columns<t_real>(scattered,perproc);
		if(world.rank() == world.root_id()) {
			for(int i=0; i<scattered.cols(); i++){
				for(int j=0; j<scattered.rows(); j++){
					CHECK(scattered(j,i) == double(i/perproc));
				}
			}
		} else {
			CHECK(result.size() == 0);
		}
	}


	SECTION("Scatter then gather columns of matrix (real)") {
		int perproc = 3;
		Matrix<t_real> localdata;
		Matrix<t_real> globaldata;
		if(world.rank() == world.root_id()) {
			globaldata.resize(10, world.size()*perproc);
			// Initialise the global data to be the rank of the receiving process
			for(int i=0; i<globaldata.cols(); i++){
				for(int j=0; j<globaldata.rows(); j++){
					globaldata(j,i) = i/perproc;
				}
			}
			// Scatter the data from the root to all processes
			localdata = world.scatter_eigen_simple_columns<t_real>(globaldata,perproc);
		} else {
			localdata = world.scatter_eigen_simple_columns<t_real>(perproc);
		}

		// Update the received data
		for(int j=0; j<localdata.cols(); j++){
			for(int i=0; i<localdata.rows(); i++){
				localdata(i,j) = localdata(i,j) + 11;
			}
		}

		// Gather the updated data back on to the root
		auto const gathereddata = world.gather_eigen_simple_columns<t_real>(localdata,perproc);
		if(world.rank() == world.root_id()) {
			// Check if the updated data has been received
			for(int i=0; i<gathereddata.cols(); i++){
				for(int j=0; j<gathereddata.rows(); j++){
					CHECK(gathereddata(j,i) == globaldata(j,i) + 11);
				}
			}
		} else {
			CHECK(gathereddata.size() == 0);
		}
	}


	SECTION("All sum all over image") {
		Image<t_int> image(2, 2);
		image.fill(world.rank());
		world.all_sum_all(image);
		CHECK((2 * image == world.size() * (world.size() - 1)).all());
	}

	SECTION("sum over image") {
		Image<t_int> image(2, 2);
		image.fill(world.rank());
		world.distributed_sum(image, world.root_id());
		if(world.rank() == world.root_id()) {
			CHECK((2 * image == world.size() * (world.size() - 1)).all());
		}
	}

	SECTION("sum over vector") {
		Vector<t_real> vec(world.size());
		vec.fill(world.rank());
		world.distributed_sum(vec, world.root_id());
		if(world.rank() == world.root_id()){
			for(int l=0; l<vec.size();l++){
				CHECK(vec[l]*2.0f == (world.size() * (world.size() - 1)));
			}
		}
	}

	SECTION("sum complex"){
		Matrix<t_complex> matrix;
		matrix.resize(world.size(),world.size());
		matrix.fill(std::complex<float>(world.rank(),world.rank()));
		world.distributed_sum(matrix, world.root_id());
		if(world.rank() == world.root_id()){
			for(int i=0; i<matrix.rows();i++){
				for(int l=0; l<matrix.cols();l++){
					CHECK(std::real(matrix(i,l))*2.0f == (world.size() * (world.size() - 1)));
					CHECK(std::imag(matrix(i,l))*2.0f == (world.size() * (world.size() - 1)));
				}
			}
		}
	}

	SECTION("sum complex matrix col"){
		Matrix<t_complex> matrix;
		matrix.resize(world.size(),world.size());
		matrix.fill(std::complex<float>(world.rank(),world.rank()));
		for(int i=0; i<matrix.cols(); i++){
			Vector<t_complex> temp_data = matrix.col(i);
			world.distributed_sum(temp_data, world.root_id());
			if(world.rank() == world.root_id()){
				matrix.col(i) = temp_data;
			}
		}
		if(world.rank() == world.root_id()){
			for(int i=0; i<matrix.rows();i++){
				for(int j=0; j<matrix.cols();j++){
					CHECK(std::real(matrix(i,j))*2.0f == (world.size() * (world.size() - 1)));
					CHECK(std::imag(matrix(i,j))*2.0f == (world.size() * (world.size() - 1)));
				}
			}
		}
	}

	SECTION("sum integer") {
		int number = 2;
		world.distributed_sum(&number, world.root_id());
		if(world.rank() == world.root_id()) {
			CHECK(number == world.size() * 2);
		}else{
			CHECK(number == 2);
		}
	}


	SECTION("Broadcast") {

		SECTION("boolean") {
			auto const result = world.broadcast(world.root_id() == world.rank() ? true : false, world.root_id());
			CHECK(result == true);
		}


		SECTION("integer") {
			auto const result = world.broadcast(world.root_id() == world.rank() ? 5 : 2, world.root_id());
			CHECK(result == 5);
		}

		SECTION("Eigen vector") {
			Vector<t_int> y0(3);
			y0 << 3, 2, 1;
			auto const y = world.rank() == world.root_id() ? world.broadcast(y0) :
					world.broadcast<Vector<t_int>>();
			CHECK(y == y0);

			std::vector<t_int> v0 = {3, 2, 1};
			auto const v = world.rank() == world.root_id() ? world.broadcast(v0) :
					world.broadcast<std::vector<t_int>>();
			CHECK(std::equal(v.begin(), v.end(), v0.begin()));
		}

		SECTION("Eigen image - and check for correct size initialization") {
			Image<t_int> image0(2, 2);
			image0 << 3, 2, 1, 0;
			auto const image = world.rank() == world.root_id() ? world.broadcast(image0) :
					world.broadcast<Image<t_int>>();
			CHECK(image.matrix() == image0.matrix());

			Image<t_int> const image1 = world.is_root() ? image0 : Image<t_int>();
			CHECK(world.broadcast(image1).matrix() == image0.matrix());
		}
	}

	SECTION("Send and Receive") {

		SECTION("integer") {
			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					int temp = 5;
					t_uint tag = 0;
					world.send_single(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					int temp = 0;
					t_uint tag = 0;
					world.recv_single(&temp, world.root_id(), tag);
					CHECK(temp == 5);
				}
			}
		}

		SECTION("float") {
			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					float temp = 5.54f;
					t_uint tag = 0;
					world.send_single(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					float temp = 0;
					t_uint tag = 0;
					world.recv_single(&temp, world.root_id(), tag);
					CHECK(temp == 5.54f);
				}
			}
		}

		SECTION("double") {
			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					double temp = 5.54;
					t_uint tag = 0;
					world.send_single(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					double temp = 0;
					t_uint tag = 0;
					world.recv_single(&temp, world.root_id(), tag);
					CHECK(temp == 5.54);
				}
			}
		}

		SECTION("long double") {
			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					long double temp = 7.72;
					t_uint tag = 0;
					world.send_single(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					long double temp = 0;
					t_uint tag = 0;
					world.recv_single(&temp, world.root_id(), tag);
					CHECK(temp == 7.72);
				}
			}
		}

		SECTION("complex float") {
			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					std::complex<float> temp(5.4, 3);
					t_uint tag = 0;
					world.send_single(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					std::complex<float> temp(0, 0);
					t_uint tag = 0;
					world.recv_single(&temp, world.root_id(), tag);
					CHECK(std::real(temp) == 5.4f);
					CHECK(std::imag(temp) == 3);
				}
			}
		}

		SECTION("complex double") {
			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					std::complex<double> temp(5.4, 3);
					t_uint tag = 0;
					world.send_single(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					std::complex<double> temp(0, 0);
					t_uint tag = 0;
					world.recv_single(&temp, world.root_id(), tag);
					CHECK(std::real(temp) == 5.4);
					CHECK(std::imag(temp) == 3);
				}
			}
		}

		SECTION("multiple sends") {
			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					int temp = 5;
					t_uint tag = 0;
					for(int j = 1; j < world.size(); j++){
						world.send_single(temp, j, tag);
					}
				}else if(world.rank()>0){
					int temp = 0;
					t_uint tag = 0;
					world.recv_single(&temp, world.root_id(), tag);
					CHECK(temp == 5);
				}
			}
		}

		SECTION("Eigen vector") {
			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					Vector<t_int> temp(3);
					temp << 3, 2, 1;
					t_uint tag = 0;
					world.send_eigen(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					Vector<t_int> answer(3);
					answer << 3, 2, 1;
					Vector<t_int> temp(3);
					t_uint tag = 0;
					world.recv_eigen(temp, world.root_id(), tag);
					CHECK(temp == answer);
				}
			}

			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					Vector<t_real> temp(3);
					temp << 3.2f, 2.1f, 1.4f;
					t_uint tag = 0;
					world.send_eigen(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					Vector<t_real> answer(3);
					answer << 3.2f, 2.1f, 1.4f;
					Vector<t_real> temp(3);
					t_uint tag = 0;
					world.recv_eigen(temp, world.root_id(), tag);
					CHECK(temp == answer);
				}
			}


			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					std::vector<t_int> temp =  {3, 2, 1};
					t_uint tag = 0;
					world.send_std_vector(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					std::vector<t_int> answer =  {3, 2, 1};
					std::vector<t_int> temp(3);
					t_uint tag = 0;
					world.recv_std_vector(temp, world.root_id(), tag);
					CHECK(std::equal(temp.begin(), temp.end(), answer.begin()));
				}
			}

			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					std::vector<t_real> temp =  {3.5f, 2.3f, 1.9f};
					t_uint tag = 0;
					world.send_std_vector(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					std::vector<t_real> answer =  {3.5f, 2.3f, 1.9f};
					std::vector<t_real> temp(3);
					t_uint tag = 0;
					world.recv_std_vector(temp, world.root_id(), tag);
					CHECK(std::equal(temp.begin(), temp.end(), answer.begin()));
				}
			}

		}

		SECTION("Eigen image - and check for correct size") {

			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					Image<t_int> temp(2, 2);
					temp << 3, 2, 1, 0;
					t_uint tag = 0;
					world.send_eigen<Image<t_int>>(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					Image<t_int> answer(2, 2);
					answer << 3, 2, 1, 0;
					Image<t_int> temp(2, 2);
					t_uint tag = 0;
					world.recv_eigen<Image<t_int>>(temp, world.root_id(), tag);
					CHECK(temp.matrix() == answer.matrix());
				}
			}

		}

		SECTION("Eigen matrix - and check for correct size") {

			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					Matrix<t_int> temp(2, 2);
					temp << 3, 2, 1, 0;
					t_uint tag = 0;
					world.send_eigen<Matrix<t_int>>(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					Matrix<t_int> answer(2, 2);
					answer << 3, 2, 1, 0;
					Matrix<t_int> temp(2, 2);
					t_uint tag = 0;
					world.recv_eigen<Matrix<t_int>>(temp, world.root_id(), tag);
					CHECK(temp.matrix() == answer.matrix());
				}
			}

		}

		SECTION("Eigen matrix row - and check for correct size") {

			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					Matrix<t_int> temp(2, 2);
					temp << 3, 2, 1, 0;
					t_uint tag = 0;
					auto temp_row = temp.row(0);
					world.send_eigen<Vector<t_int>>(temp_row, world.root_id()+1, tag);
					temp_row = temp.row(1);
					world.send_eigen<Vector<t_int>>(temp_row, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					Matrix<t_int> answer(2, 2);
					answer << 3, 2, 1, 0;
					Matrix<t_int> temp(2, 2);
					t_uint tag = 0;
					Vector<t_int> temp_row(2);
					world.recv_eigen<Vector<t_int>>(temp_row, world.root_id(), tag);
					temp.row(0) = temp_row;
					world.recv_eigen<Vector<t_int>>(temp_row, world.root_id(), tag);
					temp.row(1) = temp_row;
					CHECK(temp.row(0) == answer.row(0));
					CHECK(temp(0,0) == answer(0,0));
					CHECK(temp(0,1) == answer(0,1));
					CHECK(temp.row(1) == answer.row(1));
					CHECK(temp(1,0) == answer(1,0));
					CHECK(temp(1,1) == answer(1,1));
				}
			}

		}


		SECTION("Eigen Sparse Matrix int ") {

			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					Eigen::SparseMatrix<t_int> temp(10, 1);
					for(int i=0; i<10; i++){
						temp.insert(i,0) = i*10;
					}
					t_uint tag = 0;
					world.send_sparse_eigen<t_int>(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					Eigen::SparseMatrix<t_int> answer(10, 1);
					for(int i=0; i<10; i++){
						answer.insert(i,0) = i*10;
					}
					Eigen::SparseMatrix<t_int> temp;
					t_uint tag = 0;
					world.recv_sparse_eigen<t_int>(temp, world.root_id(), tag);
					for(int i=0; i<10; i++){
						CHECK(temp.coeff(i,0) == answer.coeff(i,0));
					}
				}
			}
		}

		SECTION("Eigen Sparse Matrix real ") {

			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					Eigen::SparseMatrix<t_real> temp(10, 1);
					for(int i=0; i<10; i++){
						temp.insert(i,0) = i*10;
					}
					t_uint tag = 0;
					world.send_sparse_eigen<t_real>(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					Eigen::SparseMatrix<t_real> answer(10, 1);
					for(int i=0; i<10; i++){
						answer.insert(i,0) = i*10;
					}
					Eigen::SparseMatrix<t_real> temp;
					t_uint tag = 0;
					world.recv_sparse_eigen<t_real>(temp, world.root_id(), tag);
					for(int i=0; i<10; i++){
						CHECK(temp.coeff(i,0) == answer.coeff(i,0));
					}
				}
			}
		}

		SECTION("Eigen Sparse Matrix complex ") {

			if(world.size() >= 2){
				if(world.rank() == world.root_id()){
					Eigen::SparseMatrix<t_complex> temp(10, 1);
					for(int i=0; i<10; i++){
						temp.insert(i,0) = i*10;
					}
					t_uint tag = 0;
					world.send_sparse_eigen<t_complex>(temp, world.root_id()+1, tag);
				}else if(world.rank() == world.root_id()+1){
					Eigen::SparseMatrix<t_complex> answer(10, 1);
					for(int i=0; i<10; i++){
						answer.insert(i,0) = i*10;
					}
					Eigen::SparseMatrix<t_complex> temp;
					t_uint tag = 0;
					world.recv_sparse_eigen<t_complex>(temp, world.root_id(), tag);
					for(int i=0; i<10; i++){
						CHECK(temp.coeff(i,0) == answer.coeff(i,0));
					}
				}
			}
		}

	}
}
#endif
