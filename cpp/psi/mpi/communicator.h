#ifndef PSI_MPI_COMMUNICATOR_H
#define PSI_MPI_COMMUNICATOR_H

#include "psi/config.h"

#ifdef PSI_MPI

#include <memory>
#include <mpi.h>
#include <set>
#include <string>
#include <type_traits>
#include <vector>
#include "psi/mpi/types.h"
#include "psi/types.h"
#include "psi/logging.h"
#include <iostream>

namespace psi {
namespace mpi {

//! \brief Wrapper for an mpi communicator
//!
class Communicator {
	//! Holds actual data associated with mpi
	struct Comm {
		//! The context
		MPI_Comm comm;
		//! The number of processes
		t_uint size;
		//! The rank of this object
		t_uint rank;
		//! Whether this process is active in this communicator
		bool active;
	};

public:

	//! No-op communicator
	Communicator() : mpi_comm() {}

	static Communicator None() { return Communicator(MPI_COMM_NULL, false); }
	static Communicator World() { return Communicator(MPI_COMM_WORLD, true); }
	static Communicator World(int padding) { return Communicator(MPI_COMM_WORLD, true, padding); }
	static Communicator Self() { return Communicator(MPI_COMM_SELF, true); }

	virtual ~Communicator(){};

	//! The number of processes
	//decltype(Comm::size) size() const { return mpi_comm ? mpi_comm->size : 1; }
	decltype(Comm::size) size() const {
		if(mpi_comm){
			return mpi_comm->size;
		}else{
			return 1;
		}
	}
	//! The rank of this proc
	decltype(Comm::rank) rank() const {
		if(mpi_comm){
			return mpi_comm->rank;
		}else{
			return 0;
		}
	}
	decltype(Comm::active) active() const { return mpi_comm ? mpi_comm->active: true; }

	// Enable returning the MPI communicator through the ** operator of the Comm object (i.e. **this)
	decltype(Comm::comm) operator*() const {
		if(not mpi_comm)
			abort("Communicator was not set");
		return mpi_comm->comm;
	}

	void barrier() const{
		MPI_Barrier(**this);
	}

	//! Get the number of padding processes
    t_uint number_of_padding_processes() const { return padding_processes; }
    //! Setting Root id for this communicator
	void root_id(int rank) { root_rank = rank; }
	//! Root id for this communicator
    t_uint root_id() const { return root_rank; }
	//! True if process is root
	bool is_root() const { return rank() == root_id(); }
	//! \brief Duplicate communicator
	Communicator duplicate() const;
	//! Alias for duplicate
	Communicator clone() const { return duplicate(); }
	//! Will call MPI_Abort then print the reason
	void abort(const std::string & reason) const;

	double time() const { return MPI_Wtime(); }

	//! In-place reduction over an image
	template <class T>
	typename std::enable_if<is_registered_type<T>::value>::type
	reduce(Matrix<T> &image, MPI_Op operation, t_uint const root) const {
		if(!active()) return;
		if(size() == 1){
			return;
		}
		assert(mpi_comm and image.size() and image.data());
		if(is_root()){
		MPI_Reduce(MPI_IN_PLACE, image.data(), image.size(), registered_type(T(0)), operation,
				root,**this);
		}else{
			MPI_Reduce(image.data(), image.data(), image.size(), registered_type(T(0)), operation,
					root,**this);
		}
	}
	template <class T>
	typename std::enable_if<is_registered_type<T>::value>::type
	reduce(Image<T> &image, MPI_Op operation, t_uint const root) const {
		if(!active()) return;
		if(size() == 1){
			return;
		}
		assert(mpi_comm and image.size() and image.data());
		if(is_root()){
		MPI_Reduce(MPI_IN_PLACE, image.data(), image.size(), registered_type(T(0)), operation,
				root,**this);
		}else{
			MPI_Reduce(image.data(), image.data(), image.size(), registered_type(T(0)), operation,
					root,**this);
		}
	}
	template <class T>
	typename std::enable_if<is_registered_type<T>::value>::type
	reduce(Vector<T> &vec, MPI_Op operation, t_uint const root) const {
		if(!active()) return;
		if(size() == 1){
			return;
		}
		assert(mpi_comm and vec.size() and vec.data());
		if(is_root()){
		MPI_Reduce(MPI_IN_PLACE, vec.data(), vec.size(), registered_type(T(0)), operation,
				root,**this);
		}else{
			MPI_Reduce(vec.data(), vec.data(), vec.size(), registered_type(T(0)), operation,
					root,**this);
		}
	}
	template <class T>
	typename std::enable_if<is_registered_type<T>::value>::type
	reduce(T number, MPI_Op operation, t_uint const root) const {
		if(!active()) return;
		if(size() == 1){
			return;
		}
		assert(mpi_comm);
		if(is_root()){
			MPI_Reduce(MPI_IN_PLACE, number, 1, registered_type(number), operation,
					root,**this);
		}else{
			MPI_Reduce(number, number, 1, registered_type(number), operation,
					root,**this);
		}
		return;
	}

	//! Helper function for reducing through sum
	// This one does the activity in place
	template <class T>
	typename std::enable_if<is_registered_type<T>::value>::type
	distributed_sum(T value,t_uint const root) const {
		if(!active()) return;
		reduce(value, MPI_SUM, root);
	}
	// This one does the activity in place
	template <class T>
	typename std::enable_if<is_registered_type<typename T::Scalar>::value>::type
	distributed_sum(T &image, t_uint const root) const {
		if(!active()) return;
		reduce(image, MPI_SUM, root);
	}
	template <class T>
	typename std::enable_if<is_registered_type<typename T::Scalar>::value, T>::type
	distributed_sum(T const &image, t_uint const root) const {
		T result(image);
		if(!active()) return result;
		reduce(result, MPI_SUM, root);
		return result;
	}



	//! Helper function for all reducing
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, T>::type
	all_reduce(T const &value, MPI_Op operation) const;

	//! In-place reduction over an image
	template <class T>
	typename std::enable_if<is_registered_type<T>::value>::type
	all_reduce(Matrix<T> &image, MPI_Op operation) const {
		if(!active()) return;
		if(size() == 1){
			return;
		}
		assert(mpi_comm and image.size() and image.data());
		MPI_Allreduce(MPI_IN_PLACE, image.data(), image.size(), registered_type(T(0)), operation,
				**this);
	}
	template <class T>
	typename std::enable_if<is_registered_type<T>::value>::type
	all_reduce(Image<T> &image, MPI_Op operation) const {
		if(!active()) return;
		if(size() == 1){
			return;
		}
		assert(mpi_comm and image.size() and image.data());
		MPI_Allreduce(MPI_IN_PLACE, image.data(), image.size(), registered_type(T(0)), operation,
				**this);
	}
	template <class T>
	typename std::enable_if<is_registered_type<T>::value>::type
	all_reduce(Vector<T> &image, MPI_Op operation) const {
		if(!active()) return;
		if(size() == 1){
			return;
		}
		assert(mpi_comm and image.size() and image.data());
		MPI_Allreduce(MPI_IN_PLACE, image.data(), image.size(), registered_type(T(0)), operation,
				**this);
	}
	template <class T>
	typename std::enable_if<is_registered_type<T>::value>::type
	all_reduce(VectorBlock<T> &image, MPI_Op operation) const {
		if(!active()) return;
		if(size() == 1){
			return;
		}
		assert(mpi_comm and image.size() and image.data());
		MPI_Allreduce(MPI_IN_PLACE, image.data(), image.size(), registered_type(T(0)), operation,
				**this);
	}


	//! Helper function for reducing through sum
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, T>::type all_sum_all(T const &value) const {
		if(!active()) return value;
		return all_reduce(value, MPI_SUM);
	}
	template <class T>
	typename std::enable_if<is_registered_type<typename T::Scalar>::value>::type
	all_sum_all(T &image) const {
		if(!active()) return;
		all_reduce(image, MPI_SUM);
	}
	template <class T>
	typename std::enable_if<is_registered_type<typename T::Scalar>::value, T>::type
	all_sum_all(T const &image) const {
		T result(image);
		if(!active()) return result;
		all_reduce(result, MPI_SUM);
		return result;
	}


	//! Broadcasts object
	bool broadcast(bool const &value, t_uint const root) const;
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, T>::type
	broadcast(T const &value, t_uint const root) const;
	//! Receive broadcast object
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, T>::type
	broadcast(t_uint const root) const;
	template <class T>
	typename std::enable_if<is_registered_type<typename T::Scalar>::value, T>::type
	broadcast(T const &vec, t_uint const root) const;
	template <class T>
	typename std::enable_if<is_registered_type<typename T::Scalar>::value, T>::type
	broadcast(t_uint const root) const;
	template <class T>
	typename std::enable_if<is_registered_type<typename T::value_type>::value
	and not std::is_base_of<Eigen::EigenBase<T>, T>::value,T>::type
	broadcast(T const &vec, t_uint const root) const;
	template <class T>
	typename std::enable_if<is_registered_type<typename T::value_type>::value
	and not std::is_base_of<Eigen::EigenBase<T>, T>::value,T>::type
	broadcast(t_uint const root) const;
	std::string broadcast(std::string const &input, t_uint const root) const;

	//! Scatter eigen Vector types
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, Matrix<T>>::type
	scatter_eigen_simple_columns(Matrix<T> const &values, t_uint const columnsperproc, t_uint const root) const;
	//! Receive scattered objects
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, Matrix<T>>::type
	scatter_eigen_simple_columns(t_uint const columnsperproc, t_uint const root) const;


	//! Scatter count objects per proc
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, T>::type
	scatter(std::vector<T> const &values, t_uint const count, t_uint const root) const;
	//! Receive scattered objects
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, T>::type
	scatter(t_uint const count, t_uint const root) const;


	//! Scatter one object per proc
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, T>::type
	scatter_one(std::vector<T> const &values, t_uint const root) const;
	//! Receive scattered objects
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, T>::type
	scatter_one(t_uint const root) const;

	//! Scatter
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, Vector<T>>::type
	scatterv(Vector<T> const &vec, std::vector<t_int> const &sizes,
			t_uint const root) const;
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, Vector<T>>::type
	scatterv(t_int local_size, t_uint const root) const;

	// Gather one object per proc
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, std::vector<T>>::type
	gather(T const value, t_uint const root) const;

	//! Gather
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, Vector<T>>::type
	gather(Vector<T> const &vec, std::vector<t_int> const &sizes,
			t_uint const root) const {
		if(!active()) return vec;
		return gather_<Vector<T>, T>(vec, sizes, root);
	}
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, Vector<T>>::type
	gather(Vector<T> const &vec, t_uint const root) const {
		if(!active()) return vec;
		return gather_<Vector<T>, T>(vec, root);
	}

	template <class T>
	typename std::enable_if<is_registered_type<T>::value, std::set<T>>::type
	gather(std::set<T> const &set, std::vector<t_int> const &sizes,
			t_uint const root) const;
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, std::set<T>>::type
	gather(std::set<T> const &vec, t_uint const root) const;
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, std::vector<T>>::type
	gather(std::vector<T> const &vec, std::vector<t_int> const &sizes,
			t_uint const root) const {
		if(!active()) return vec;
		return gather_<std::vector<T>, T>(vec, sizes, root);
	}
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, std::vector<T>>::type
	gather(std::vector<T> const &vec, t_uint const root) const {
		if(!active()) return vec;
		return gather_<std::vector<T>, T>(vec, root);
	}

	//! Gather
	template <class T>
	typename std::enable_if<is_registered_type<T>::value, Matrix<T>>::type
	gather_eigen_simple_columns(Matrix<T> const &input, t_uint columnsperproc,
			t_uint const root) const;

	//! Split current communicator
	Communicator split(t_int colour) const { return split(colour, rank()); }

	Communicator split(t_int colour, t_uint rank) const {
		MPI_Comm comm;
		MPI_Comm_split(**this, colour, static_cast<t_int>(rank), &comm);
		return comm;
	}

	template <typename T, typename = typename std::enable_if<is_registered_type<T>::value, T>::type>
	void send_single(T const &value, t_uint const target, t_uint const tag) const {
		assert(target < size());
		assert(target >= 0);
		if(size() == 1)
			return;
		if(not mpi_comm)
			return;
		MPI_Send(&value, 1, registered_type(value), target, tag, **this);
		return;
	}

	template <typename T, typename = typename std::enable_if<is_registered_type<typename T::Scalar>::value, T>::type>
	void send_eigen(T const &value, t_uint const target, t_uint const tag) const {
		assert(target < size());
		assert(target >= 0);
		if(size() == 1)
			return;
		if(not mpi_comm)
			return;
		auto const Nx = value.rows();
		auto const Ny = value.cols();
		MPI_Send(const_cast<typename T::Scalar *>(value.data()), Nx * Ny, Type<typename T::Scalar>::value,
				target, tag, **this);
		return;
	}

	template <typename T, typename = typename std::enable_if<is_registered_type<typename T::value_type>::value
			and not std::is_base_of<Eigen::EigenBase<T>, T>::value, T>::type>
	void send_std_vector(T const &value, t_uint const target, t_uint const tag) const {
		assert(target < size());
		assert(target >= 0);
		if(size() == 1)
			return;
		if(not mpi_comm)
			return;
		auto const N = value.size();
		MPI_Send(const_cast<typename T::value_type *>(value.data()), N,
				Type<typename T::value_type>::value, target, tag, **this);
		return;
	}


	template <typename T, typename = typename std::enable_if<is_registered_type<T>::value, T>::type>
	void recv_single(T value, t_uint const source, t_uint const tag) const {
		assert(source < size());
		assert(source >= 0);
		if(size() == 1)
			return;
		if(not mpi_comm)
			return;
		MPI_Status status;
		MPI_Recv(value, 1, registered_type(value), source, tag, **this, &status);
		return;
	}

	template <typename T, typename = typename std::enable_if<is_registered_type<typename T::Scalar>::value, T>::type>
	void recv_eigen(T &value, t_uint const source, t_uint const tag) const {
		assert(source < size());
		assert(source >= 0);
		if(size() == 1)
			return;
		if(not mpi_comm)
			return;
		MPI_Status status;
		auto const Nx = value.rows();
		auto const Ny = value.cols();
		MPI_Recv(const_cast<typename T::Scalar *>(value.data()), Nx * Ny, Type<typename T::Scalar>::value,
				source, tag, **this, &status);
		return;
	}

	template <typename T, typename = typename std::enable_if<is_registered_type<typename T::value_type>::value
			and not std::is_base_of<Eigen::EigenBase<T>, T>::value, T>::type>
	void recv_std_vector(T &value, t_uint const source, t_uint const tag) const {
		assert(source < size());
		assert(source >= 0);
		if(size() == 1)
			return;
		if(not mpi_comm)
			return;
		MPI_Status status;
		MPI_Recv(value.data(), value.size(), Type<typename T::value_type>::value, source, tag, **this, &status);
		return;
	}


	template <typename T, typename = typename std::enable_if<is_registered_type<T>::value, T>::type>
	void send_sparse_eigen(Eigen::SparseMatrix<T> &in,  t_uint const target, t_uint const tag) const {
		assert(target < size());
		in.makeCompressed();
	    int rows=in.rows();
	    int cols=in.cols();
	    int nnz=in.nonZeros();
	    assert(rows==in.innerSize() && cols==in.outerSize());
	    assert(in.outerIndexPtr()[cols]==nnz);
	    int shape[3] = {rows, cols, nnz};
	    MPI_Send(shape, 3, MPI_INT, target, tag, **this);
	    MPI_Send(in.valuePtr(), nnz, registered_type(in.valuePtr()[0]), target, tag, **this);
	    MPI_Send(in.innerIndexPtr(), nnz, registered_type(in.innerIndexPtr()[0]), target, tag, **this);
	    MPI_Send(in.outerIndexPtr(), cols, registered_type(in.outerIndexPtr()[0]), target, tag, **this);
	}


	template <typename T, typename = typename std::enable_if<is_registered_type<T>::value, T>::type>
	void nonblocking_send_sparse_eigen(Eigen::SparseMatrix<T> &in,  t_uint const target, t_uint const tag, t_int const *shape, MPI_Request *requests) const {
		assert(target < size());
		in.makeCompressed();
	    MPI_Isend(shape, 3, MPI_INT, target, tag, **this, &requests[0]);
	    MPI_Isend(in.valuePtr(), shape[2], registered_type(in.valuePtr()[0]), target, tag, **this, &requests[1]);
	    MPI_Isend(in.innerIndexPtr(), shape[2], registered_type(in.innerIndexPtr()[0]), target, tag, **this, &requests[2]);
	    MPI_Isend(in.outerIndexPtr(), shape[1], registered_type(in.outerIndexPtr()[0]), target, tag, **this, &requests[3]);
	}


	template <typename T, typename = typename std::enable_if<is_registered_type<T>::value, T>::type>
	void recv_sparse_eigen(Eigen::SparseMatrix<T> &out,  t_uint const source, t_uint const tag) const {
		assert(source < size());
		int shape[3];
		MPI_Status status;
		MPI_Recv(shape, 3, MPI_INT, source, tag, **this, &status);
		out.resize(shape[0], shape[1]);
		out.reserve(shape[2]);
		MPI_Recv(out.valuePtr(), shape[2], registered_type(out.valuePtr()[0]), source, tag, **this, &status);
		MPI_Recv(out.innerIndexPtr(), shape[2], registered_type(out.innerIndexPtr()[0]), source, tag, **this, &status);
		MPI_Recv(out.outerIndexPtr(), shape[1], registered_type(out.outerIndexPtr()[0]), source, tag, **this, &status);
		out.outerIndexPtr()[shape[1]] = shape[2];
		out.uncompress();
	}


	void wait_on_all(MPI_Request requests[], t_uint count) const {
		MPI_Status statuses[count];
		// Currently we are ignoring the statuses, we should check them for errors.
		// TODO Check statuses for errors.
		MPI_Waitall(count, requests, statuses);
	}


private:

	//! Rank of the root process
	int root_rank = 0;
	//! Number of processes to pad the root rank process by
	int padding_processes = 0;
	//! Class data
	std::shared_ptr<Comm const> mpi_comm;

	//! Deletes an mpi communicator
	static void delete_comm(Comm *mpi_comm);

	//! \brief Constructs a communicator
	Communicator(MPI_Comm const &comm);
	Communicator(MPI_Comm const &comm, bool const active);
	//! \brief Construct a communicator, set whether active in this communicator or not, and
	//! set the number of padding processes to use.
	//! \details padding_processes allows extra MPI processes to be used to to pad out the root rank and give
	//! it access to more memory. active enables split communicators where a communicator is split into communicators
	//! based on the active variable value.
	Communicator(MPI_Comm const &comm, bool const active, int const padding_processes);
	//! Gather
	template <class CONTAINER, class T>
	CONTAINER gather_(CONTAINER const &vec, std::vector<t_int> const &sizes, t_uint const root) const;
	template <class CONTAINER, class T>
	CONTAINER gather_(CONTAINER const &vec, t_uint const root) const;
};

bool init(int argc, const char **argv);
//! True if mpi has been initialized
bool initialized();
//! True if mpi has been finalized
bool finalized();

void finalize();


template <class T>
typename std::enable_if<is_registered_type<T>::value, T>::type
Communicator::all_reduce(T const &value, MPI_Op operation) const {
	if(size() == 1)
		return value;
	assert(mpi_comm);
	T result;
	MPI_Allreduce(&value, &result, 1, registered_type(value), operation, **this);
	return result;
}

/* Send rows of the image to different processes
 * Number of columns per process specified in the columnsperproc variable.
 */

template <class T>
typename std::enable_if<is_registered_type<T>::value, Matrix<T>>::type
Communicator::scatter_eigen_simple_columns(Matrix<T> const &values, t_uint const columnsperproc, t_uint const root) const {
	assert(root < size());
	if(values.cols() != size()*columnsperproc){
		abort("Scatter eigen expected a fixed number of elements per process");
	}
	if(size() == 1)
		return values;
	int rows = values.rows();
	auto const Ny = broadcast(rows, root);
	Matrix<T> result(rows, columnsperproc);
	MPI_Scatter(values.data(), columnsperproc*Ny, Type<T>::value, result.data(), columnsperproc*Ny, Type<T>::value, root, **this);
	return result;
}
//! Receive scattered objects
template <class T>
typename std::enable_if<is_registered_type<T>::value, Matrix<T>>::type
Communicator::scatter_eigen_simple_columns(t_uint const columnsperproc, t_uint const root) const {
	if(rank() == root)
		abort("Root calling wrong scatter_eigen");
	int temp = 0;
	auto const Ny =  broadcast(temp, root);
	Matrix<T> result(Ny, columnsperproc);
	MPI_Scatter(nullptr, columnsperproc*Ny, Type<T>::value, result.data(), columnsperproc*Ny, Type<T>::value, root, **this);
	return result;
}


template <class T>
typename std::enable_if<is_registered_type<T>::value, T>::type
Communicator::scatter(std::vector<T> const &values, t_uint const count, t_uint const root) const {
	assert(root < size());
	if(values.size() != size()*count)
		abort("Expected a fixed number of elements per process");
	if(size() == 1)
		return values.at(0);
	T result;
	MPI_Scatter(values.data(), count, registered_type(result), &result, count, registered_type(result), root,
			**this);
	return result;
}
//! Receive scattered objects
template <class T>
typename std::enable_if<is_registered_type<T>::value, T>::type
Communicator::scatter(t_uint const count, t_uint const root) const {
	T result;
	MPI_Scatter(nullptr, count, registered_type(result), &result, count, registered_type(result), root,
			**this);
	return result;
}

template <class T>
typename std::enable_if<is_registered_type<T>::value, T>::type
Communicator::scatter_one(std::vector<T> const &values, t_uint const root) const {
	assert(root < size());
	if(values.size() != size())
		abort("Expected a single object per process");
	if(size() == 1)
		return values.at(0);
	T result;
	MPI_Scatter(values.data(), 1, registered_type(result), &result, 1, registered_type(result), root,
			**this);
	return result;
}
//! Receive scattered objects
template <class T>
typename std::enable_if<is_registered_type<T>::value, T>::type
Communicator::scatter_one(t_uint const root) const {
	T result;
	MPI_Scatter(nullptr, 1, registered_type(result), &result, 1, registered_type(result), root,
			**this);
	return result;
}

template <class T>
typename std::enable_if<is_registered_type<T>::value, Vector<T>>::type
Communicator::scatterv(Vector<T> const &vec, std::vector<t_int> const &sizes,
		t_uint const root) const {
	if(size() == 1) {
		if(sizes.size() == 1 and vec.size() != sizes.front())
			abort("Input vector size and sizes are inconsistent on root");
		return vec;
	}
	if(rank() != root)
		return scatterv<T>(sizes.at(rank()), root);
	std::vector<int> sizes_, displs;
	int i = 0;
	for(auto const size : sizes) {
		sizes_.push_back(static_cast<int>(size));
		displs.push_back(i);
		i += size;
	}
	if(vec.size() != i)
		abort("Input vector size and sizes are inconsistent");

	Vector<T> result(sizes[rank()]);
	if(not mpi_comm)
		result = vec.head(sizes[rank()]);
	else
		MPI_Scatterv(vec.data(), sizes_.data(), displs.data(), registered_type(T(0)), result.data(),
				sizes_[rank()], registered_type(T(0)), root, **this);
	return result;
}

template <class T>
typename std::enable_if<is_registered_type<T>::value, Vector<T>>::type
Communicator::scatterv(t_int local_size, t_uint const root) const {
	if(rank() == root)
		abort("Root calling wrong scatterv");
	std::vector<int> sizes(size());
	sizes[rank()] = local_size;
	Vector<T> result(sizes[rank()]);
	MPI_Scatterv(nullptr, sizes.data(), nullptr, registered_type(T(0)), result.data(), local_size,
			registered_type(T(0)), root, **this);
	return result;
}

template <class T>
typename std::enable_if<is_registered_type<T>::value, std::vector<T>>::type
Communicator::gather(T const value, t_uint const root) const {
	assert(root < size());
	if(size() == 1)
		return {value};
	std::vector<T> result;
	if(rank() == root) {
		result.resize(size());
		MPI_Gather(&value, 1, registered_type(value), result.data(), 1, registered_type(value), root,
				**this);
	} else
		MPI_Gather(&value, 1, registered_type(value), nullptr, 1, registered_type(value), root, **this);
	return result;
}

template <class CONTAINER, class T>
CONTAINER Communicator::gather_(CONTAINER const &vec, std::vector<t_int> const &sizes,
		t_uint const root) const {
	assert(root < size());
	if(sizes.size() != size() and rank() == root)
		abort("Sizes and communicator size do not match on root");
	else if(rank() != root and sizes.size() != 0 and sizes.size() != size())
		abort("Outside root, sizes should be either empty or match the number of procs");
	else if(sizes.size() == size() and sizes[rank()] != static_cast<t_int>(vec.size()))
		abort("Sizes and input vector size do not match");

	if(size() == 1)
		return vec;

	if(rank() != root)
		return gather_<CONTAINER, T>(vec, root);

	std::vector<int> sizes_, displs;
	int result_size = 0;
	for(auto const size : sizes) {
		sizes_.push_back(static_cast<int>(size));
		displs.push_back(result_size);
		result_size += size;
	}
	CONTAINER result(result_size);
	MPI_Gatherv(vec.data(), sizes_[rank()], mpi::Type<T>::value, result.data(), sizes_.data(),
			displs.data(), mpi::Type<T>::value, root, **this);
	return result;
}

template <class CONTAINER, class T>
CONTAINER Communicator::gather_(CONTAINER const &vec, t_uint const root) const {
	assert(root < size());
	if(rank() == root)
		abort("Root calling wrong gather");

	MPI_Gatherv(vec.data(), vec.size(), mpi::Type<T>::value, nullptr, nullptr, nullptr,
			mpi::Type<T>::value, root, **this);
	return CONTAINER();
}

template <class T>
typename std::enable_if<is_registered_type<T>::value, std::set<T>>::type
Communicator::gather(std::set<T> const &set, std::vector<t_int> const &sizes,
		t_uint const root) const {
	assert(root < size());
	if(rank() != root)
		return gather(set, root);
	else if(size() == 1)
		return set;

	assert(sizes.size() == size());
	assert(sizes[root] == set.size());
	Vector<T> buffer(set.size());
	std::copy(set.begin(), set.end(), buffer.data());
	buffer = gather(buffer, sizes, root);
	return std::set<T>(buffer.data(), buffer.data() + buffer.size());
}

template <class T>
typename std::enable_if<is_registered_type<T>::value, std::set<T>>::type
Communicator::gather(std::set<T> const &set, t_uint const root) const {
	assert(root < size());
	if(rank() == root)
		abort("Root calling wrong gather");

	Vector<T> buffer(set.size());
	std::copy(set.begin(), set.end(), buffer.data());
	gather(buffer, root);
	return std::set<T>();
}

template <class T>
typename std::enable_if<is_registered_type<T>::value, Matrix<T>>::type
Communicator::gather_eigen_simple_columns(Matrix<T> const &input, t_uint columnsperproc, t_uint const root) const {
	assert(root < size());
	if(size() == 1)
		return {input};
	Matrix<T> result;
	auto const Nx = input.cols();
	if(Nx != columnsperproc){
		abort("Then number of columns in the input dataset for gather_eigen_simple_columns is bigger than the parameter columnsperproc. This routine is not designed to deal with that scenario.");
	}
	auto const Ny = input.rows();
	if(rank() == root) {
		// The total dataset size is the size of the individual inputs (that each process passes) multiplied by the total number of processes in the communicator.
		// This routine gathers over columns of the matrix.
		result.resize(Ny, Nx*size());
		MPI_Gather(input.data(), Ny*Nx, Type<T>::value, result.data(), Nx*Ny, Type<T>::value, root, **this);
	} else
		MPI_Gather(input.data(), Nx*Ny, Type<T>::value, nullptr, Nx*Ny, Type<T>::value, root, **this);
	return result;
}


template <class T>
typename std::enable_if<is_registered_type<T>::value, T>::type
Communicator::broadcast(T const &value, t_uint const root) const {
	assert(root < size());
	if(size() == 1)
		return value;
	if(not mpi_comm)
		return value;

	auto result = value;
	MPI_Bcast(&result, 1, registered_type(result), root, **this);
	return result;
}
//! Receive broadcast object
template <class T>
typename std::enable_if<is_registered_type<T>::value, T>::type
Communicator::broadcast(t_uint const root) const {
	assert(root < size());
	if(root == rank())
		abort("Root calling wrong broadcasting function");
	T result;
	MPI_Bcast(&result, 1, registered_type(result), root, **this);
	return result;
}

template <class T>
typename std::enable_if<is_registered_type<typename T::Scalar>::value, T>::type
Communicator::broadcast(T const &vec, t_uint const root) const {
	if(size() == 1)
		return vec;
	if(not mpi_comm)
		return vec;
	if(rank() != root)
		return broadcast<T>(root);
	assert(root < size());
	auto const Nx = broadcast(vec.rows(), root);
	auto const Ny = broadcast(vec.cols(), root);
	MPI_Bcast(const_cast<typename T::Scalar *>(vec.data()), Nx * Ny, Type<typename T::Scalar>::value,
			root, **this);
	return vec;
}
template <class T>
typename std::enable_if<is_registered_type<typename T::Scalar>::value, T>::type
Communicator::broadcast(t_uint const root) const {
	assert(root < size());
	if(root == rank())
		abort("Root calling wrong broadcasting function");
	auto const Nx = broadcast(decltype(std::declval<T>().rows())(0), root);
	auto const Ny = broadcast(decltype(std::declval<T>().cols())(0), root);
	T result(Nx, Ny);
	MPI_Bcast(result.data(), result.size(), Type<typename T::Scalar>::value, root, **this);
	return result;
}
template <class T>
typename std::enable_if<is_registered_type<typename T::value_type>::value
and not std::is_base_of<Eigen::EigenBase<T>, T>::value, T>::type
Communicator::broadcast(T const &vec, t_uint const root) const {
	assert(root < size());
	if(size() == 1)
		return vec;
	if(not mpi_comm)
		return vec;
	if(rank() != root)
		return broadcast<T>(root);
	auto const N = broadcast(vec.size(), root);
	MPI_Bcast(const_cast<typename T::value_type *>(vec.data()), N,
			Type<typename T::value_type>::value, root, **this);
	return vec;
}
template <class T>
typename std::enable_if<is_registered_type<typename T::value_type>::value
and not std::is_base_of<Eigen::EigenBase<T>, T>::value,T>::type
Communicator::broadcast(t_uint const root) const {
	assert(root < size());
	if(root == rank())
		abort("Root calling wrong broadcasting function");
	auto const N = broadcast(decltype(std::declval<T>().size())(0), root);
	T result(N);
	MPI_Bcast(result.data(), result.size(), Type<typename T::value_type>::value, root, **this);
	return result;
}


} // namespace mpi
} // namespace psi
#endif /* ifdef PSI_MPI */
#endif /* ifndef PSI_MPI_COMMUNICATOR */
