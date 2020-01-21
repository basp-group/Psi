#ifndef PSI_IO_H
#define PSI_IO_H

#include "psi/config.h"

#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>
#include "psi/types.h"
#include <iostream>
#include "psi/logging.h"
#include "psi/mpi/decomposition.h"

namespace psi {
namespace io {

enum class IOStatus
{
	Failure 			   = 0,
	Success 			   = 1,
	WrongImageSize       = 2,
	OpenFailure          = 3,
	FileWriteError       = 4,
	FileReadError        = 5,
	WrongNumberOfFrequencies = 6,
	WrongNumberOfTimeBlocks = 7,
};

//! TODO: Refactor the code to allow the time blocking and wideband stuff to be done by the the same routines, rather than requiring separate routines.


//! \brief Reads and writes data to file
//!
template <class SCALAR> class IO {


public:

	static std::string GetErrorMessage(IOStatus status){
		switch(status) {
		case IOStatus::Failure:
			return "General Failure";
		case IOStatus::Success:
			return "Success";
		case IOStatus::WrongImageSize:
			return "Size of the data read in from file is different from the size of data expected";
		case IOStatus::OpenFailure:
			return "File open failure";
		case IOStatus::FileWriteError:
			return "Failure with file write";
		case IOStatus::FileReadError:
			return "Failure with file read";
		case IOStatus::WrongNumberOfFrequencies:
			return "Wrong number of frequencies in checkpoint file";
		case IOStatus::WrongNumberOfTimeBlocks:
			return "Wrong number of time blocks for this frequency in checkpoint file";
		}
		return "General Failure: unidentified";
	}

	//! Scalar type
	typedef SCALAR value_type;
	//! Scalar type
	typedef value_type Scalar;
	//! Real type
	typedef typename real_type<Scalar>::type Real;
	//! Type of the underlying vectors
	typedef Vector<Scalar> t_Vector;
	//! Type of the underlying matrix
	typedef Matrix<Scalar> t_Matrix;

	//! Constructor
	IO() {};

	virtual ~IO(){};

	IOStatus checkpoint_time_blocking_with_collect(psi::mpi::Decomposition decomp, std::string checkpoint_filename, t_Vector out, Vector<Real> local_epsilons, Vector<Real> local_l1_weights, Real kappa, Real sigma2, t_uint image_size, Real delta, int current_reweighting_iter);
	IOStatus checkpoint_time_blocking(std::string checkpoint_filename, t_Vector out, Vector<Real> epsilons, Vector<Real> l1_weights, Real kappa, Real sigma2, Real delta, int current_reweighting_iter);
	IOStatus restore_time_blocking_with_distribute(psi::mpi::Decomposition decomp, std::string checkpoint_filename, t_Vector &out, Vector<Real> &epsilons, Vector<Real> &l1_weights, Real &kappa, Real &sigma2, t_uint image_size, Real &delta, int &current_reweighting_iter);
	IOStatus restore_time_blocking(std::string checkpoint_filename, t_Vector &out, Vector<Real> &epsilons, Vector<Real> &l1_weights, Real &kappa, Real &sigma2, t_uint image_size, Real &delta, int &current_reweighting_iter);

	IOStatus checkpoint_wideband_with_collect(psi::mpi::Decomposition decomp, std::string checkpoint_filename, t_Matrix out, Vector<Vector<Real>> local_epsilons, Vector<Real> total_l21_weights, Vector<Real> total_nuclear_weights, Real kappa1, Real kappa2, Real kappa3, t_uint image_size, Real delta, int current_reweighting_iter);
	IOStatus checkpoint_wideband(std::string checkpoint_filename, t_Matrix out, Vector<Vector<Real>> epsilons, Vector<Real> l21_weights, Vector<Real> nuclear_weights, Real kappa1, Real kappa2, Real kappa3, Real delta, int current_reweighting_iter);
	IOStatus restore_wideband_with_distribute(psi::mpi::Decomposition decomp, std::string checkpoint_filename, t_Matrix &out, Vector<Vector<Real>> &epsilons, Vector<Real> &l21_weights, Vector<Real> &nuclear_weights, Real &kappa1, Real &kappa2, Real &kappa3, t_uint number_of_frequencies, t_uint image_size, Real &delta, int &current_reweighting_iter);
	IOStatus restore_wideband(std::string checkpoint_filename, t_Matrix &out, Vector<Vector<Real>> &epsilons, Vector<Real> &l21_weights, Vector<Real> &nuclear_weights, Real &kappa1, Real &kappa2, Real &kappa3, t_uint number_of_frequencies, t_uint image_size, Real &delta, int &current_reweighting_iter);


protected:


private:




};


template <class T>
IOStatus IO<T>::checkpoint_time_blocking_with_collect(psi::mpi::Decomposition decomp, std::string checkpoint_filename, t_Vector out, Vector<Real> local_epsilons, Vector<Real> local_l1_weights, Real kappa, Real sigma2, t_uint image_size, Real delta, int current_reweighting_iter){

	IOStatus status = IOStatus::Success;
	Vector<Real> total_l1_weights;
	Vector<Real> total_epsilons;

	if(!decomp.parallel_mpi() or decomp.global_comm().is_root()){
		total_l1_weights = Vector<Real>(decomp.frequencies()[0].number_of_wavelets*image_size);
		total_epsilons = Vector<Real>(decomp.frequencies()[0].number_of_time_blocks);
	}

	decomp.template collect_epsilons<Vector<Real>>(local_epsilons, total_epsilons);
	decomp.template collect_l1_weights<Vector<Real>>(local_l1_weights, total_l1_weights, image_size);

	if(!decomp.parallel_mpi() or decomp.global_comm().is_root()){
		status = IO<T>::checkpoint_time_blocking(checkpoint_filename, out, total_epsilons, total_l1_weights, kappa, sigma2, delta, current_reweighting_iter);
	}
	return status;

}

template <class T>
IOStatus IO<T>::checkpoint_time_blocking(std::string checkpoint_filename, t_Vector out, Vector<Real> epsilons, Vector<Real> l1_weights, Real kappa, Real sigma2, Real delta, int current_reweighting_iter){

	IOStatus error = IOStatus::Success;
	FILE* pFile;
	pFile = fopen(checkpoint_filename.c_str(), "wb");
	if(pFile != NULL){
		//! Write the kappa to the file
		if((error == IOStatus::Success) and (fwrite(&kappa, sizeof(kappa), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the sigma2 to the file
		if((error == IOStatus::Success) and (fwrite(&sigma2, sizeof(sigma2), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the delta to the file
		if((error == IOStatus::Success) and (fwrite(&delta, sizeof(delta), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the current_reweighting_iter to the file
		if((error == IOStatus::Success) and (fwrite(&current_reweighting_iter, sizeof(current_reweighting_iter), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the size of the out data set to the file to enable the routine reading the file to setup the
		//! data structure to store the data before it reads it.
		int size = out.size();
		if((error == IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the out vector to the file
		if((error == IOStatus::Success) and (fwrite(out.data(), sizeof(out.data()[0]), out.size(), pFile) != out.size())){
			error = IOStatus::FileWriteError;
		}
		//! Write the size of the epsilons data set to the file to enable the routine reading the file to setup the
		//! data structure to store the data before it reads it.
		size = epsilons.size();
		if((error == IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the epsilons vector to the file
		if((error == IOStatus::Success) and (fwrite(epsilons.data(), sizeof(epsilons[0]), epsilons.size(), pFile) != epsilons.size())){
			error = IOStatus::FileWriteError;
		}
		//! Write the size of the weights data set to the file to enable the routine reading the file to setup the
		//! data structure to store the data before it reads it.
		size = l1_weights.size();
		if((error == IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the l1_weights vector to the file
		if((error == IOStatus::Success) and (fwrite(l1_weights.data(), sizeof(l1_weights[0]), l1_weights.size(), pFile) != l1_weights.size())){
			error = IOStatus::FileWriteError;
		}
		fclose(pFile);
	}else{
		error = IOStatus::OpenFailure;
	}
	return error;
}

template <class T>
IOStatus IO<T>::restore_time_blocking_with_distribute(psi::mpi::Decomposition decomp, std::string checkpoint_filename, t_Vector &out, Vector<Real> &epsilons, Vector<Real> &l1_weights, Real &kappa, Real &sigma2, t_uint image_size, Real &delta, int &current_reweighting_iter){

	psi::io::IOStatus restore_status = IOStatus::Success;
	Vector<Real> total_epsilons;
	Vector<Real> total_l1_proximal_weights;

	if(!decomp.parallel_mpi() or decomp.global_comm().is_root()){

		total_epsilons = Vector<Real>(decomp.frequencies()[0].number_of_time_blocks);
		total_l1_proximal_weights = Vector<Real>(decomp.frequencies()[0].number_of_wavelets*image_size);

		psi::io::IOStatus restore_status = restore_time_blocking(checkpoint_filename, out, total_epsilons, total_l1_proximal_weights, kappa, sigma2, image_size, delta, current_reweighting_iter);
		if(restore_status != psi::io::IOStatus::Success){
			if(restore_status == psi::io::IOStatus::WrongImageSize){
				PSI_HIGH_LOG("Problem restoring from checkpoint. Restored image size is different to target size. You probably restored an incorrect restore file.");
			}else{
				PSI_HIGH_LOG("Problem restoring checkpoint from file. Error is: {}", psi::io::IO<Scalar>::GetErrorMessage(restore_status));
			}
			decomp.global_comm().abort("Problem restoring from checkpoint. Quitting.");
		}
	}

	decomp.template distribute_epsilons<Vector<t_real>>(epsilons, total_epsilons);
	decomp.template distribute_l1_weights<Vector<t_real>>(l1_weights, total_l1_proximal_weights, image_size);
	kappa = decomp.global_comm().broadcast(kappa, decomp.global_comm().root_id());
	sigma2 = decomp.global_comm().broadcast(sigma2, decomp.global_comm().root_id());
	delta = decomp.global_comm().broadcast(delta, decomp.global_comm().root_id());
	current_reweighting_iter = decomp.global_comm().broadcast(current_reweighting_iter, decomp.global_comm().root_id());
	return restore_status;
}

template <class T>
IOStatus IO<T>::restore_time_blocking(std::string checkpoint_filename, t_Vector &out, Vector<Real> &epsilons, Vector<Real> &l1_weights, Real &kappa, Real &sigma2, t_uint out_size, Real &delta, int &current_reweighting_iter){

	IOStatus error = IOStatus::Success;
	FILE* pFile;
	pFile = fopen(checkpoint_filename.c_str(), "r");
	if(pFile != NULL){
		//! First read kappa
		if(error == IOStatus::Success and fread(&kappa, sizeof(kappa), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}
		//! Now read sigma2
		if(error == IOStatus::Success and fread(&sigma2, sizeof(sigma2), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}
		//! Now read delta
		if(error == IOStatus::Success and fread(&delta, sizeof(delta), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}
		//! Now read current_reweighting_iter
		if(error == IOStatus::Success and fread(&current_reweighting_iter, sizeof(current_reweighting_iter), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}
		//! Now read the size of the out data set
		int size;
		if(error == IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}else{
			if(size != out_size){
				error = IOStatus::WrongImageSize;
			}else if(out.size() != size){
				out = t_Vector(size);
			}
		}

		//! Read the out vector from the file
		if(error == IOStatus::Success and fread(out.data(), sizeof(out.data()[0]), out.size(), pFile) != out.size()){
			error = IOStatus::FileReadError;
		}
		//! Read the size of the epsilons data set
		if(error == IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}else{
			if(epsilons.size() != size){
				epsilons = Vector<Real>(size);
			}
		}
		//! Read the epsilons vector from the file
		if(error == IOStatus::Success and fread(epsilons.data(), sizeof(epsilons[0]), epsilons.size(), pFile) != epsilons.size()){
			error = IOStatus::FileReadError;
		}
		//! Read the size of the weights data set
		if(error == IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}else{
			if(l1_weights.size() != size){
				l1_weights = Vector<Real>(size);
			}
		}
		//! Read the l1_weights vector from the file
		if(error == IOStatus::Success and fread(l1_weights.data(), sizeof(l1_weights[0]), l1_weights.size(), pFile) != l1_weights.size()){
			error = IOStatus::FileReadError;
		}
		fclose(pFile);
	}else{
		error = IOStatus::FileWriteError;
	}
	return error;
}

template <class T>
IOStatus IO<T>::checkpoint_wideband_with_collect(psi::mpi::Decomposition decomp, std::string checkpoint_filename, t_Matrix out, Vector<Vector<Real>> local_epsilons, Vector<Real> total_l21_weights, Vector<Real> total_nuclear_weights, Real kappa1, Real kappa2, Real kappa3, t_uint image_size, Real delta, int current_reweighting_iter){

	IOStatus status = IOStatus::Success;
	Vector<Vector<Real>> total_epsilons;

	if(!decomp.parallel_mpi() or decomp.global_comm().is_root()){
		total_epsilons = Vector<Vector<Real>>(decomp.global_number_of_frequencies());
		for(int i=0; i<decomp.global_number_of_frequencies(); i++){
			total_epsilons(i) = Vector<Real>(decomp.frequencies()[i].number_of_time_blocks);
		}
	}

	decomp.template collect_epsilons_wideband_blocking<Vector<Vector<Real>>>(local_epsilons, total_epsilons);

	if(!decomp.parallel_mpi() or decomp.global_comm().is_root()){
		status = IO<T>::checkpoint_wideband(checkpoint_filename, out, total_epsilons, total_l21_weights, total_nuclear_weights, kappa1, kappa2, kappa3, delta, current_reweighting_iter);
	}
	return status;

}

template <class T>
IOStatus IO<T>::checkpoint_wideband(std::string checkpoint_filename, t_Matrix out, Vector<Vector<Real>> epsilons, Vector<Real> l21_weights, Vector<Real> nuclear_weights, Real kappa1, Real kappa2, Real kappa3, Real delta, int current_reweighting_iter){

	IOStatus error = IOStatus::Success;
	FILE* pFile;
	pFile = fopen(checkpoint_filename.c_str(), "wb");
	if(pFile != NULL){
		//! Write the kappa1 to the file
		if((error == IOStatus::Success) and (fwrite(&kappa1, sizeof(kappa1), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the kappa2 to the file
		if((error == IOStatus::Success) and (fwrite(&kappa2, sizeof(kappa2), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the kappa3 to the file
		if((error == IOStatus::Success) and (fwrite(&kappa3, sizeof(kappa3), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the delta to the file
		if((error == IOStatus::Success) and (fwrite(&delta, sizeof(delta), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the current_reweighting_iter to the file
		if((error == IOStatus::Success) and (fwrite(&current_reweighting_iter, sizeof(current_reweighting_iter), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the size of the out data set to the file to enable the routine reading the file to setup the
		//! data structure to store the data before it reads it.
		int size = out.rows();
		if((error == IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		size = out.cols();
		if((error == IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the out vector to the file
		if((error == IOStatus::Success) and (fwrite(out.data(), sizeof(out.data()[0]), out.size(), pFile) != out.size())){
			error = IOStatus::FileWriteError;
		}
		//! Write the size of the overall epsilon vector to the file to enable the routine reading the file to setup the
		//! data structure to store the data before it reads it.
		size = epsilons.size();
		if((error == IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		for(int i=0; i<epsilons.size(); i++){
			//! Write the size of the individual epsilon vector to the file to enable the routine reading the file to setup the
			//! data structure to store the data before it reads it.
			size = epsilons(i).size();
			if((error == IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
				error = IOStatus::FileWriteError;
			}
			//! Write the epsilons vector to the file
			if((error == IOStatus::Success) and (fwrite(epsilons(i).data(), sizeof(epsilons(i)[0]), epsilons(i).size(), pFile) != epsilons(i).size())){
				error = IOStatus::FileWriteError;
			}
		}
		//! Write the size of the weights data set to the file to enable the routine reading the file to setup the
		//! data structure to store the data before it reads it.
		size = l21_weights.size();
		if((error == IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the l21_weights vector to the file
		if((error == IOStatus::Success) and (fwrite(l21_weights.data(), sizeof(l21_weights[0]), l21_weights.size(), pFile) != l21_weights.size())){
			error = IOStatus::FileWriteError;
		}
		size = nuclear_weights.size();
		if((error == IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
			error = IOStatus::FileWriteError;
		}
		//! Write the nuclear_weights vector to the file
		if((error == IOStatus::Success) and (fwrite(nuclear_weights.data(), sizeof(nuclear_weights[0]), nuclear_weights.size(), pFile) != nuclear_weights.size())){
			error = IOStatus::FileWriteError;
		}
		fclose(pFile);
	}else{
		error = IOStatus::OpenFailure;
	}
	return error;
}

template <class T>
IOStatus IO<T>::restore_wideband_with_distribute(psi::mpi::Decomposition decomp, std::string checkpoint_filename, t_Matrix &out, Vector<Vector<Real>> &epsilons, Vector<Real> &l21_weights, Vector<Real> &nuclear_weights, Real &kappa1, Real &kappa2, Real &kappa3, t_uint frequencies, t_uint image_size, Real &delta, int &current_reweighting_iter){

	psi::io::IOStatus restore_status = IOStatus::Success;
	Vector<Vector<Real>> total_epsilons;

	if(!decomp.parallel_mpi() or decomp.global_comm().is_root()){

		total_epsilons = Vector<Vector<Real>>(decomp.global_number_of_frequencies());
		for(int i=0; i<decomp.global_number_of_frequencies(); i++){
			total_epsilons(i) = Vector<Real>(decomp.frequencies()[i].number_of_time_blocks);
		}
		psi::io::IOStatus restore_status = restore_wideband(checkpoint_filename, out, total_epsilons, l21_weights, nuclear_weights, kappa1, kappa2, kappa3, frequencies, image_size, delta, current_reweighting_iter);
		if(restore_status != psi::io::IOStatus::Success){
			if(restore_status == psi::io::IOStatus::WrongImageSize){
				PSI_HIGH_LOG("Problem restoring from checkpoint. Restored image size is different to target size. You probably restored an incorrect restore file.");
			}else{
				PSI_HIGH_LOG("Problem restoring checkpoint from file. Error is: {}", psi::io::IO<Scalar>::GetErrorMessage(restore_status));
			}
			decomp.global_comm().abort("Problem restoring from checkpoint. Quitting.");
		}
	}

	decomp.template distribute_epsilons_wideband_blocking<Vector<Vector<t_real>>>(epsilons, total_epsilons);
	kappa1 = decomp.global_comm().broadcast(kappa1, decomp.global_comm().root_id());
	kappa2 = decomp.global_comm().broadcast(kappa2, decomp.global_comm().root_id());
	kappa3 = decomp.global_comm().broadcast(kappa3, decomp.global_comm().root_id());
	delta = decomp.global_comm().broadcast(delta, decomp.global_comm().root_id());
	current_reweighting_iter = decomp.global_comm().broadcast(current_reweighting_iter, decomp.global_comm().root_id());
	return restore_status;
}



template <class T>
IOStatus IO<T>::restore_wideband(std::string checkpoint_filename, t_Matrix &out, Vector<Vector<Real>> &epsilons, Vector<Real> &l21_weights, Vector<Real> &nuclear_weights, Real &kappa1, Real &kappa2, Real &kappa3, t_uint frequencies, t_uint image_size, Real &delta, int &current_reweighting_iter){

	IOStatus error = IOStatus::Success;
	FILE* pFile;
	pFile = fopen(checkpoint_filename.c_str(), "r");
	if(pFile != NULL){
		//! First read kappa1
		if(error == IOStatus::Success and fread(&kappa1, sizeof(kappa1), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}
		//! Now read kappa2
		if(error == IOStatus::Success and fread(&kappa2, sizeof(kappa2), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}
		//! Now read kappa3
		if(error == IOStatus::Success and fread(&kappa3, sizeof(kappa3), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}
		//! Now read delta
		if(error == IOStatus::Success and fread(&delta, sizeof(delta), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}
		//! Now read current_reweighting_iter
		if(error == IOStatus::Success and fread(&current_reweighting_iter, sizeof(current_reweighting_iter), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}
		//! Now read the size of the out data set
		int size;
		if(error == IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}else{
			if(size != image_size){
				PSI_HIGH_LOG("Problem with size of rows {} {}",size,image_size);
				error = IOStatus::WrongNumberOfFrequencies;
			}
		}
		if(error == IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}else{
			if(size != frequencies){
				PSI_HIGH_LOG("Problem with size of cols {} {}",size,frequencies);
				error = IOStatus::WrongNumberOfFrequencies;
			}else if(out.size() != frequencies*image_size){
				out = t_Matrix(image_size,frequencies);
			}
			//! Read the out matrix from the file
			if(error == IOStatus::Success and fread(out.data(), sizeof(out.data()[0]), out.size(), pFile) != out.size()){
				error = IOStatus::FileReadError;
			}
		}


		//! Read the size of the vector of epsilons data set
		if(error == IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}else{
			if(epsilons.size() != size){
				epsilons = Vector<Vector<Real>>(size);
			}
		}
		int innersize;
		for(int i=0; i<size; i++){
			//! Read the size individual epsilons vector
			if(error == IOStatus::Success and fread(&innersize, sizeof(innersize), 1, pFile) != 1){
				error = IOStatus::FileReadError;
			}else{
				if(epsilons(i).size() != innersize){
					epsilons(i) = Vector<Real>(innersize);
				}
				//! Read the epsilons vector from the file
				if(error == IOStatus::Success and fread(epsilons(i).data(), sizeof(epsilons(i)[0]), epsilons(i).size(), pFile) != epsilons(i).size()){
					error = IOStatus::FileReadError;
				}
			}
		}

		//! Read the size of the l21 weights data set
		if(error == IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}else{
			if(l21_weights.size() != size){
				l21_weights = Vector<Real>(size);
			}
		}
		//! Read the l21_weights vector from the file
		if(error == IOStatus::Success and fread(l21_weights.data(), sizeof(l21_weights[0]), l21_weights.size(), pFile) != l21_weights.size()){
			error = IOStatus::FileReadError;
		}
		//! Read the size of the nulcear weights data set
		if(error == IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
			error = IOStatus::FileReadError;
		}else{
			if(nuclear_weights.size() != size){
				nuclear_weights = Vector<Real>(size);
			}
		}
		//! Read the nuclear_weights vector from the file
		if(error == IOStatus::Success and fread(nuclear_weights.data(), sizeof(nuclear_weights[0]), nuclear_weights.size(), pFile) != nuclear_weights.size()){
			error = IOStatus::FileReadError;
		}
		fclose(pFile);
	}else{
		error = IOStatus::FileWriteError;
	}
	return error;
}


} // namespace IO
} // namespace psi
#endif /* ifndef PSI_IO_H */
