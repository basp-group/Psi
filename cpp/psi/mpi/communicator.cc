#include <exception>
#include <mpi.h>
#include "psi/mpi/communicator.h"
#include "psi/logging.h"
#include <iostream>

namespace psi {
namespace mpi {
#ifdef PSI_MPI



bool Communicator::broadcast(bool const &value, t_uint const root) const {
	assert(root < size());
	if(size() == 1)
		return value;
	if(not mpi_comm)
		return value;
	//! Handle bool values by converting them into 1 or 0 and broadcasting that
	//! instead and then converting back to a boolean.
	int result;
	if(value){
		result = 1;
	}else{
		result = 0;
	}
	MPI_Bcast(&result, 1, MPI_INTEGER, root, **this);
	if(result == 1){
		return true;
	}else{
		return false;
	}
}

void Communicator::delete_comm(Communicator::Comm *const mpi_comm) {
	if(mpi_comm->comm != MPI_COMM_WORLD and mpi_comm->comm != MPI_COMM_SELF and mpi_comm->comm != MPI_COMM_NULL and not finalized()){
		MPI_Comm_free(&mpi_comm->comm);
	}
	delete mpi_comm;
}

Communicator::Communicator(MPI_Comm const &external_comm) : Communicator(external_comm, true){

}

Communicator::Communicator(MPI_Comm const &external_comm, bool const active) : mpi_comm(nullptr){
	if(external_comm == MPI_COMM_NULL){
		return;
	}
	int size, rank;
	MPI_Comm_size(external_comm, &size);
	MPI_Comm_rank(external_comm, &rank);
	Comm const data{external_comm, static_cast<t_uint>(size), static_cast<t_uint>(rank), active};
	mpi_comm = std::shared_ptr<Comm const>(new Comm(data), &delete_comm);
}

void Communicator::abort(const std::string & reason) const{
	fprintf(stderr, "MPI Error on Rank %i: %s\n", rank(), reason.c_str());
	MPI_Abort(**this, MPI_ERR_OTHER);
}


Communicator Communicator::duplicate() const {
	if(not mpi_comm){
		return Communicator(MPI_COMM_NULL);
	}
	MPI_Comm newcomm;
	MPI_Comm_dup(**this, &newcomm);
	Communicator dupcomm = Communicator(newcomm);
	return dupcomm;
}


bool init(int argc, const char **argv) {
	if(finalized())
		throw std::runtime_error("MPI has already been finalized");
	if(not initialized()) {
#ifdef PSI_HYBRID
		t_int provided;
		if(MPI_Init_thread(&argc, const_cast<char ***>(&argv), MPI_THREAD_FUNNELED, &provided)
				== MPI_SUCCESS){
			if(provided < MPI_THREAD_FUNNELED){
				PSI_THROW("MPI threading support not sufficient.");
			}else{
				return true;
			}
#else
			if(MPI_Init(&argc, const_cast<char ***>(&argv)) == MPI_SUCCESS) return true;
#endif
			return false;
		}else{
			return true;
		}
	}


	bool initialized() {
		int flag;
		auto const error = MPI_Initialized(&flag);
		if(error != MPI_SUCCESS) {
			PSI_ERROR("Error while calling MPI_Initialized ({})", error);
			throw std::runtime_error("MPI error");
		}
		return static_cast<bool>(flag);
	}

	bool finalized() {
		int finalized;
		MPI_Finalized(&finalized);
		return static_cast<bool>(finalized);
	}

	void finalize() {
		if(finalized() or not initialized())
			return;
		MPI_Finalize();
	}
#endif

} /* psi::mpi */
}/* psi  */
