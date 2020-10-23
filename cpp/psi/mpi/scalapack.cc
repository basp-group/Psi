#include <exception>
#include "psi/mpi/scalapack.h"
#ifdef PSI_SCALAPACK

namespace psi {
namespace mpi {


Scalapack::Scalapack(bool using_scalapack){
	using_scalapack_ = using_scalapack;
}

//! Setup the blacs process grid for the Scalapack SVD work
bool Scalapack::setupBlacs(Decomposition decomp, int number_of_processes, int M, int N){

	int info;
	//decomp_ = decomp;

	M_ = M;

	N_ = N;

	number_of_processes_ = number_of_processes;
	if(decomp.global_comm().size() < number_of_processes_){
		PSI_HIGH_LOG("Problem with SVD process count {} is bigger than the total processes available {}. Setting to total processes available", number_of_processes, decomp.global_comm().size());
		number_of_processes_ = decomp.global_comm().size();
	}

	blacs_get_(&i_zero_, &i_zero_, &initial_process_handle_);

	all_processes_ = initial_process_handle_;

	int nprocs = decomp.global_comm().size();

	blacs_gridinit_(&all_processes_, "R", &nprocs, &i_one_);

	involved_process_group_ = initial_process_handle_;

	std::tie(nprow_, npcol_) = process_grid_size(number_of_processes_);

	// involved_process_group_ is the
	blacs_gridinit_(&involved_process_group_, "R", &nprow_, &npcol_);

	minMN_ = std::min(M_,N_);

	int in_scalapack = involvedInSVD();

        scalapack_comm_ = Communicator(decomp.global_comm().split(in_scalapack));

	if(involvedInSVD()){

		blacs_gridinfo_(&involved_process_group_, &nprow_, &npcol_, &myrow_, &mycol_);

        //! Part of potential functionality for an adaptive decomposition worked out on the fly
        //        //! Currently not used, use specifies SVD process count instead.
        //
      //          bool active = true;
	//	bool failure;
		int n_param = N_;

        //! Part of potential functionality for an adaptive decomposition worked out on the fly
        //        //! Currently not used, use specifies SVD process count instead.
        //
	//	while(active){
//			failure = false;
//			if(n_param <= 1){
//				decomp.global_comm().abort("Problem in initialising Scalapack, n_param is one or less. Try a different number of SVD processes");
//			}

			mb_ = std::min(M_/number_of_processes_, n_param);
			nb_ = std::min(M_/number_of_processes_, n_param);

        //! Part of potential functionality for an adaptive decomposition worked out on the fly
        //        //! Currently not used, use specifies SVD process count instead.
        //
//			if(mb_ == 0 or nb_ == 0){
//			decomp.global_comm().abort("Problem in initialising Scalapack, mb or nb parameter is zero, which is an error. Try a different number of SVD processes");
//				failure = true;
//			}

			npa_ = numroc_(&M_, &mb_, &myrow_, &i_zero_, &nprow_);
			mpa_ = numroc_(&N_, &nb_, &mycol_, &i_zero_, &npcol_);

        //! Part of potential functionality for an adaptive decomposition worked out on the fly
        //        //! Currently not used, use specifies SVD process count instead.
        //
//			if(mpa_ == 0 or npa_ == 0){
//                        decomp.global_comm().abort("Problem in initialising Scalapack, mpa or npa parameter is zero, which is an error. Try a different number of SVD processes");
//				failure = true;
 //			}


			llda_ = std::max(i_one_,npa_);
			descinit_(desca_, &M_, &N_, &mb_, &nb_, &i_zero_, &i_zero_, &involved_process_group_, &llda_, &info);
	
			if(info != 0){
				decomp.global_comm().abort("Scalapack setup BLACS descinit desca has failed");
			}

			npu_ = numroc_(&M_, &mb_, &myrow_, &i_zero_, &nprow_);
			mpu_ = numroc_(&minMN_, &nb_, &mycol_, &i_zero_, &npcol_);

        //! Part of potential functionality for an adaptive decomposition worked out on the fly
        //        //! Currently not used, use specifies SVD process count instead.
	//                if(mpu_ == 0 or npu_ == 0){
  //                      decomp.global_comm().abort("Problem in initialising Scalapack, mpu or npu parameter is zero, which is an error. Try a different number of SVD processes");
	//			failure = true;
	//		}

			lldu_ = std::max(i_one_,npu_);
			descinit_(descu_, &M_, &minMN_, &mb_, &nb_, &i_zero_, &i_zero_, &involved_process_group_, &lldu_, &info);

			if(info != 0){
				decomp.global_comm().abort("Scalapack setup BLACS descinit descu has failed");
			}

			npvt_ = numroc_(&minMN_, &mb_, &myrow_, &i_zero_, &nprow_);
			mpvt_ = numroc_(&N_, &nb_, &mycol_, &i_zero_, &npcol_);

        //! Part of potential functionality for an adaptive decomposition worked out on the fly
        //! Currently not used, use specifies SVD process count instead.
        //              if(mpvt_ == 0 or npvt_ == 0){
        //                       decomp.global_comm().abort("Problem in initialising Scalapack, mpvt or npvt parameter is zero, which is an error. Try a different number of SVD processes");
//	 			failure = true;
//	                }

			lldvt_ = std::max(i_one_,npvt_);
			descinit_(descvt_, &minMN_, &N_, &mb_, &nb_, &i_zero_, &i_zero_, &involved_process_group_, &lldvt_, &info);
	
			if(info != 0){
				decomp.global_comm().abort("Scalapack setup BLACS descinit descvt has failed");
			}

	//! Part of potential functionality for an adaptive decomposition worked out on the fly
	//! Currently not used, use specifies SVD process count instead.
	//		int agree_finished = failure;
	//		scalapack_comm_.all_sum_all(agree_finished);
	//		if(agree_finished == 0){
	//			active = false;
	//		}			
	//		active = false;
	//		n_param = n_param/2;
	//	}

		//! These values are used to create the size of arrays in calling code, so if any are set to 0 then reset them to 1 so the 
		//! arrays actually have a size. This stops Eigen not creating the array, and then running into assert errors. The array 
		//! sizes don't actually matter if the parameter is zero because they won't be used, but Eigen still complains as the 
		//! arrays have not been created.
		if(mb_ == 0) mb_ = 1;
                if(nb_ == 0) nb_ = 1;
                if(mpa_ == 0) mpa_ = 1;
                if(npa_ == 0) npa_ = 1;
                if(mpu_ == 0) mpu_ = 1;
                if(npu_ == 0) npu_ = 1;
                if(mpvt_ == 0) mpvt_ = 1;
                if(npvt_ == 0) npvt_ = 1;

	}


	setup_ = true;

	return setup_;
}

//! Calculate a square as possible a process grid configuration (i.e. if using 20 processes favour 4 x 5 rather than
//! 20 x 1 or 2 x 10).
std::pair<int, int> Scalapack::process_grid_size(int number_of_processes){
	int nprow = 0;
	int npcol = 0;
	if(number_of_processes > 0){
		int middle = std::ceil(number_of_processes / 2);
		nprow = number_of_processes;
		npcol = 1;
		for(int i=2; i<=middle; i++){
			if(number_of_processes % i == 0 && (number_of_processes / i) >= i){
				nprow = number_of_processes / i;
				npcol = i;
			}
		}
	}
	return(std::make_pair(nprow, npcol));
}

//! Run the configuration step for the SVD where U and VT output is not required
bool Scalapack::setupSVD(Vector<t_real> local_Data, Vector<t_real> &output){

	double u[1];
	double vt[1];

	if(setup_){

		if(involvedInSVD()){

			lwork_ = -1;
			work_.resize(1);

			pdgesvd_("N","N",&M_,&N_,&local_Data[0],&i_one_,&i_one_,desca_,&output[0],
					&u[0],&i_one_,&i_one_,descu_,
					&vt[0],&i_one_,&i_one_,descvt_,
					&work_[0],&lwork_,&info_);

			lwork_ = static_cast<int>(work_[0]);
			work_.resize(lwork_);

			if(info_ == 0){
				work_setup_ = true;
				return true;
			}else{
				PSI_ERROR("Scalapack SVD setup has failed with error: {}", info_);
				return false;
			}

		}
		return true;
	}else{
		PSI_ERROR("Scalapack not setup!");
		return false;
	}

}

//! Run the configuration step for the SVD where U and VT output is required
bool Scalapack::setupSVD(Vector<t_real> local_Data, Vector<t_real> &output, Vector<t_real> &output_u, Vector<t_real> &output_vt){

	if(involvedInSVD()){

		if(setup_){

			lwork_ = -1;
			work_.resize(1);

			pdgesvd_("V","V",&M_,&N_,&local_Data[0],&i_one_,&i_one_,desca_,&output[0],
					&output_u[0],&i_one_,&i_one_,descu_,
					&output_vt[0],&i_one_,&i_one_,descvt_,
					&work_[0],&lwork_,&info_);

			lwork_ = static_cast<int>(work_[0]);
			work_.resize(lwork_);

			if(info_ == 0){
				work_setup_ = true;
				return true;
			}else{
				PSI_ERROR("Scalapack SVD setup has failed with error: {}", info_);
				return false;
			}

		}else{
			PSI_ERROR("Scalapack not setup!");
			return false;
		}

	}

	return true;
}

//! Run the scalapack SVD without the the U and VT output
bool Scalapack::runSVD(Vector<t_real> local_Data, Vector<t_real> &output){

	double u[1];
	double vt[1];

	if(involvedInSVD()){

		if(setup_ && work_setup_){

			pdgesvd_("N","N",&M_,&N_,&local_Data[0],&i_one_,&i_one_,desca_,&output[0],
					&u[0],&i_one_,&i_one_,descu_,
					&vt[0],&i_one_,&i_one_,descvt_,
					&work_[0],&lwork_,&info_);

			if(info_ == 0){
				return true;
			}else{
				PSI_ERROR("Scalapack SVD has failed with error: {}", info_);
				return false;
			}

		}else{
			if(not setup_){
				PSI_ERROR("Scalapack not setup!");
				return false;
			}
			if(not work_setup_){
				PSI_ERROR("Scalapack work space not setup. You need to run setupSVD once before running runSVD!");
				return false;
			}
		}
	}

	return true;

}

//! Run the scalapack SVD with the the U and VT output
bool Scalapack::runSVD(Vector<t_real> local_Data, Vector<t_real> &output, Vector<t_real> &output_u, Vector<t_real> &output_vt){

	if(setup_ && work_setup_){

		if(involvedInSVD()){

			pdgesvd_("V","V",&M_,&N_,&local_Data[0],&i_one_,&i_one_,desca_,&output[0],
					&output_u[0],&i_one_,&i_one_,descu_,
					&output_vt[0],&i_one_,&i_one_,descvt_,
					&work_[0],&lwork_,&info_);

			if(info_ == 0){
				return true;
			}else{
				PSI_ERROR("Scalapack SVD has failed with error: {}", info_);
				return false;
			}

		}

		return true;

	}else{
		if(not setup_){
			PSI_ERROR("Scalapack not setup!");
			return false;
		}
		if(not work_setup_){
			PSI_ERROR("Scalapack work space not setup. You need to run setupSVD once before running runSVD!");
			return false;
		}
		return true;
	}

}



//! Interface routine to scatter a matrix, required to cast it to a vector for the underlying functionality
bool Scalapack::scatter(Decomposition decomp, Vector<t_real> &local_data, Matrix<t_real> total_data, int M, int N, int mp, int np){


	// Moving data around like this is expensive, investigate making all eigen matricies row major for Psi.
	//Matrix<t_real> row_major = Eigen::Map<RowMatrix<t_real>>(total_data.data(), total_data.cols(), total_data.rows());

	Vector<t_real> data_vector = Eigen::Map<Vector<t_real>>(total_data.data(), total_data.cols()*total_data.rows());
	scatter(decomp, local_data, data_vector, M, N, mp, np);

	return true;
}

//! Scatter input data for the scalapack SVD to participating processes.
bool Scalapack::scatter(Decomposition decomp, Vector<t_real> &local_data, Vector<t_real> total_data, int M, int N, int mp, int np){

	if(setup_){

		if(involvedInSVD()){

			int send_row = 0;
			int send_col = 0;
			int recv_row = 0;
			int recv_col = 0;
			int datasize = 0;

			for(int row = 0; row < M; row += mb_, send_row=(send_row+1)%nprow_){
				send_col = 0;
				int nr = mb_;
				if(M-row < nb_){
					nr = M-row;
				}

				for(int col = 0; col < N; col += nb_, send_col=(send_col+1)%npcol_){
					int nc = nb_;

					if (N-col < nb_){
						nc = N-col;
					}

					if(scalapack_comm_.is_root()){
						datasize = M;
						dgesd2d_(&involved_process_group_, &nr, &nc, &(total_data).data()[(M*col+row)], &datasize, &send_row, &send_col);
					}

					if(myrow_ == send_row && mycol_ == send_col){
						datasize = np;
						dgerv2d_(&involved_process_group_, &nr, &nc, (double *)&(local_data).data()[(np*recv_col+recv_row)], &datasize, &i_zero_, &i_zero_);
						recv_col = (recv_col+nc)%mp;
					}
				}

				if(myrow_ == send_row){
					recv_row = (recv_row+nr)%np;
				}

			}

		}

		return true;

	}else{

		PSI_ERROR("Scalapack not setup!");
		return false;

	}


}

//! Gather the data that has been calculated with the parallel SVD on to a single process.
bool Scalapack::gather(Decomposition decomp, Vector<t_real> local_data, Vector<t_real> &total_data, int M, int N, int mp, int np){

	if(setup_){

		if(involvedInSVD()){

			int send_row = 0;
			int send_col = 0;
			int recv_row = 0;
			int recv_col = 0;
			int datasize = 0;

			for(int row = 0; row < M; row += mb_, send_row=(send_row+1)%nprow_){
				send_col = 0;
				int nr = mb_;
				if(M-row < nb_){
					nr = M-row;
				}

				for(int col = 0; col < N; col += nb_, send_col=(send_col+1)%npcol_){
					int nc = nb_;

					if (N-col < nb_){
						nc = N-col;
					}

					if(myrow_ == send_row && mycol_ == send_col){
						datasize = np;
						dgesd2d_(&involved_process_group_, &nr, &nc, &(local_data).data()[(np*recv_col+recv_row)], &datasize, &i_zero_, &i_zero_);
						recv_col = (recv_col+nc)%mp;
					}

					if(scalapack_comm_.is_root()){
						datasize = M;
						dgerv2d_(&involved_process_group_, &nr, &nc, (double *)&(total_data).data()[(M*col+row)], &datasize, &send_row, &send_col);
					}
				}

				if(myrow_ == send_row){
					recv_row = (recv_row+nr)%np;
				}

			}

		}

		return true;

	}else{

		PSI_ERROR("Scalapack not setup!");
		return false;

	}
}

//! Send data from the global root process to the scalapack root process
//! The global_comm communicator is used because the global root process
//! is not guaranteed to be in the scalapack communicator.
bool Scalapack::sendToScalapackRoot(Decomposition decomp, Vector<t_real> &data_svd){
	int tag = 12;
	//! Currently we're assuming the scalapack root process is rank 0 in the global communicator. Ensure this is the case.
	if(involvedInSVD() and scalapack_comm_.is_root()){
		assert(decomp.global_comm().rank() == 0);
		if(decomp.global_comm().is_root()){
			return false;
		}
	}
	if((involvedInSVD() and scalapack_comm_.is_root()) and not decomp.global_comm().is_root()){
		decomp.global_comm().recv_eigen<psi::Vector<t_real>>(data_svd, decomp.global_comm().root_id(), tag);
	}else if(decomp.global_comm().is_root() and not (involvedInSVD() and scalapack_comm_.is_root())){
		decomp.global_comm().send_eigen<psi::Vector<t_real>>(data_svd, 0, tag);
	}
	return true;
}

//! Send data from the global root process to the scalapack root process
//! The global_comm communicator is used because the global root process
//! is not guaranteed to be in the scalapack communicator.
bool Scalapack::recvFromScalapackRoot(Decomposition decomp, Vector<t_real> &data_svd){
	int tag = 12;

	//! Currently we're assuming the scalapack root process is rank 0 in the global communicator. Ensure this is the case.
	if(involvedInSVD() and scalapack_comm_.is_root()){
		assert(decomp.global_comm().rank() == 0);
		if(decomp.global_comm().is_root()){
			return false;
		}
	}
	if((involvedInSVD() and scalapack_comm_.is_root()) and not decomp.global_comm().is_root()){
		decomp.global_comm().send_eigen<psi::Vector<t_real>>(data_svd, decomp.global_comm().root_id(), tag);
	}else if(decomp.global_comm().is_root() and not (involvedInSVD() and scalapack_comm_.is_root())){
		decomp.global_comm().recv_eigen<psi::Vector<t_real>>(data_svd, 0, tag);
	}
	return true;
}

} /* psi::mpi */
}/* psi  */
#endif /* #ifdef PSI_SCALAPACK */
