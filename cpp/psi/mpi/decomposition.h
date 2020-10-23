#ifndef PSI_MPI_DECOMPOSITION_H
#define PSI_MPI_DECOMPOSITION_H

#include "psi/config.h"

#include <memory>
#include <mpi.h>
#include <set>
#include <string>
#include <type_traits>
#include <vector>
#include "psi/mpi/communicator.h"
#include "psi/types.h"
#include <iostream>

namespace psi {
namespace mpi {

//! \brief Calculates the MPI decomposition and holds the functions to undertake the decomposition
//!
class Decomposition {


	struct WaveletDecomp {
		//! Rank (in the global communicator) of the process with this wavelet
		t_uint global_owner;
		//! Wavelet number
		t_uint wavelet_number;
	};

	struct SubDecomp {
		//! Rank (in the global communicator) of the process involved in this sub-block
		t_uint global_owner;
		//! Sub block number
		t_uint sub_block_number;
	};

	struct TimeDecomp {
		//! Time block number
		t_uint time_block_number;
		//! Communicator for this time block
		Communicator time_block_comm = Communicator::None();
		//! The number of sub blocks in this time block
		t_uint number_of_sub_blocks;
		//! Size of time block (number of visibilities)
		t_uint size_of_block;
		//! Rank in the global communicator of the root process for this time block
		t_uint global_owner;
		//! Rank in the frequency communicator of the root process for this time block
		t_uint local_owner;
		////! Ranks (in the global communicator) of the processes involved in this time block
		//std::vector<t_uint> process_ids;
		//! Sub block decomp data, i.e. sub blocks this process is involved with from this time block
		std::vector<SubDecomp> subblocks;

	};

	struct FreqDecomp {
		//! Frequency number
		t_uint freq_number;
		//! Communicator for this frequency
		Communicator freq_comm = Communicator::None();
		//! The number of time blocks in this frequency
		t_uint number_of_time_blocks;
		//! Rank in the global communicator of the root process for this frequency
		t_uint global_owner;
		//! Rank in the frequency communicator of the root process for this frequency
		t_uint local_owner;
		//! Ranks (in the global communicator) of the processes involved in this frequency
		std::vector<t_uint> process_ids;
		//! Time block decomp data, i.e. time blocks this process is involved with from this frequency
		std::vector<TimeDecomp> time_blocks;
		//! Global rank of the first process assigned to this frequency
		t_uint lower_process;
		//! Global rank of the last process assigned to this frequency
		t_uint upper_process;
		//! Number of wavelet dictionaries for this frequency
		t_uint number_of_wavelets;
		//! Wavelet decomp data, i.e. wavelets this process is involved with
		std::vector<WaveletDecomp> wavelets;
		//! Lower bound on the number of wavelets
		int lower_wavelet;
		//! Communicator for the wavelets for this frequency
		Communicator wavelet_comm = Communicator::None();
		//! Global rank of the root process for the wavelets
		int global_wavelet_owner;
		//! MPI_Request array for holding outstanding requests for non-blocking communications
		MPI_Request *requests;
		//! Has the requests array been created and initialised. Used to stop double mallocing of
		//! the request array and thereby memory leaks, and also ensure that all requests are
		//! suitably initialised so if they aren't used they don't cause the wait functions to
		//! block.
		bool requests_initialised = false;
		//! Whether this process is involved in this frequency or not. Used to allow looping through the
		//! global set of frequencies but only undertaking work if this process is in this frequency
		bool in_this_frequency = false;
	};

	struct Decomp {
		//! Whether we are using MPI or not
		bool parallel_mpi;
		//! Global communicator object
		Communicator global_comm = Communicator::None();
		//! The number of frequencies
		t_uint number_of_frequencies;
		//! Lower bound on the number of frequencies
		t_uint lower_freq;
		//! Upper bound on the number of frequencies
		t_uint upper_freq;
		//! Frequency decomp data, i.e. frequencies this process is involved with
		std::vector<FreqDecomp> frequencies;
		//! Number of wavelet dictionaries for the root frequency
		t_uint number_of_root_wavelets;
		//! Root wavelets (i.e. those associated with the root process operations, not those in the iteration step).
		std::vector<WaveletDecomp> root_wavelets;
		//! Lower bound on the number of root wavelets for this process
		int lower_root_wavelet;
		//! Communicator for the root wavelets
		Communicator root_wavelet_comm = Communicator::None();
		//! Communicators across all wavelets
		std::vector<Communicator> wavelet_comms;
		//! Involvement in wavelet_comms
		std::vector<bool> wavelet_comms_involvement;
		//! Are we doing wavelet decomposition
		bool wavelet_decomp = false;
		//! Are we doing frequency decomposition
		bool freq_decomp = false;
		//! Are we doing time block decomposition
		bool time_decomp = false;
		//! Are we doing sub block decomposition
		bool sub_decomp = false;
		//! Are we decomposing the root wavelets (for wideband)
		bool root_wavelet_decomp = false;
		//! Are we checkpointing for this simulation
		bool checkpointing = false;
		//! Number of algorithm iterations to checkpoint on
		t_uint checkpointing_frequency = 0;
		//! Are we restoring from a checkpoint
		bool restoring = false;
		//! Have we already restored from a checkpoint
		bool checkpoint_restored = false;
	};

public:

	//! Constructor
	//! Initialise with an empty MPI communicator by default so this can be called when run in serial (done in header).
	Decomposition(const bool parallel, Communicator comm = Communicator::None());

	virtual ~Decomposition(){};

	void decompose_primal_dual(bool const freq_decomp, bool const time_decomp, bool const subblock_decomp, bool const wavelet_decomp, bool const wavelet_root_decomp, t_int const frequencies, std::vector<t_int> wavelet_levels, std::vector<t_int> time_blocks, std::vector<std::vector<t_int>> sub_blocks, bool quiet = false);
	int decompose_time_blocks(t_uint freq_number, t_uint number_of_blocks,bool time_decomp);
	int decompose_wavelets(t_uint freq_number, t_uint number_of_wavelets, bool wavelet_decomp);
	void decompose_root_wavelets(int number_of_wavelets, bool root_wavelet_decomp);
	int decompose_frequencies(t_uint number_of_frequencies, t_uint lower_process, t_uint upper_process, bool time_decomp, bool subblock_decomp, bool wavelet_decomp);

	bool checkpoint_now(t_uint niters, t_uint maxiters) const;
	bool restore_checkpoint() const;
	void restore_complete();
	void epsilon_and_weights_used();

	//template <class T> typename std::enable_if<is_registered_type<T>::value, T>::type all_reduce(T const &value, MPI_Op operation) const;

	template <class T> void collect_indices(std::vector<std::vector<T>> const indices, std::vector<std::vector<T>> &global_indices, bool global_root) const;
	template <class T> void distribute_fourier_data(std::vector<std::vector<Eigen::SparseMatrix<T>>> &local_sparse, std::vector<std::vector<Eigen::SparseMatrix<T>>> &global_sparse) const;
	template <class T> void receive_fourier_data(std::vector<Eigen::SparseMatrix<T>> &local_sparse, int freq, int my_freq, bool global_root) const;
	template <class T> void send_fourier_data(std::vector<Eigen::SparseMatrix<T>> &local_sparse, std::vector<Eigen::SparseMatrix<T>> &global_sparse, int *shapes, int k, int &my_index, int freq, bool &used_this_freq, bool global_root);
	template <class T> void collect_residual_norms(Vector<Vector<T>> const residual_norms, Vector<Vector<T>> &total_residual_norms) const;
	template <class T> void collect_residual_sizes(std::vector<std::vector<T>> const residual, std::vector<std::vector<int>> &sizes) const;
	template <class T> void collect_residuals(std::vector<T> const residual, std::vector<T> &total_residuals) const;
	template <class T> void collect_frequency_root_data(Matrix<T> const frequency_data, Matrix<T> &total_frequency_data) const;
	template <class T1, class T2> void distribute_frequency_data(T1 &frequency_data, T1 const total_frequency_data, bool const freq_root_only) const;
	template <class T> void collect_svd_data(Matrix<T> const local_VT, Matrix<T> &total_VT, Matrix<T> const local_data_svd, Matrix<T> &total_data_svd) const;
	template <class T1, class T2> void distribute_svd_data(T1 &local_VT, T1 const total_VT, T1 &local_data_svd, Vector<T2> total_data_svd, int const image_size) const;
	template <class T1, class T2> void collect_svd_result_data(Matrix<T1> const local_p, Matrix<T2> &total_p) const;
	template <class T1, class T2> void distribute_svd_result_data(Matrix<T1> &local_p, Matrix<T2> const total_p) const;
	template <class T> void collect_wavelet_root_data(Matrix<T> const wavelet_data, Matrix<T> &total_wavelet_data, int const image_size) const;
	template <class T> void collect_epsilons(T const epsilons, T &total_epsilons) const;
	template <class T> void distribute_epsilons(T &epsilons, T const total_epsilons) const;
	template <class T> void collect_epsilons_wideband_blocking(T const epsilons, T &total_epsilons) const;
	template <class T> void distribute_epsilons_wideband_blocking(T &epsilons, T const total_epsilons) const;
	template <class T> void collect_l1_weights(T const l1_weights, T &total_l1_weights, t_uint image_size) const;
	template <class T> void distribute_l1_weights(T &l1_weights, T const total_l1_weights, t_uint image_size) const;
	template <class T> void collect_l21_weights(T const l21_weights, T &total_l21_weights, t_uint image_size) const;
	template <class T> void distribute_l21_weights(T &l21_weights, T const total_l21_weights, t_uint image_size) const;
	void wait_on_requests(int freq, int count);
	void cleanup_requests(int freq);
	void initialise_requests(int freq, int count);


	//! The decomp object
	decltype(Decomp::parallel_mpi) parallel_mpi() const { return decomp_.parallel_mpi; }
	decltype(Decomp::global_comm) global_comm() const {
		if(parallel_mpi()){
			return decomp_.global_comm;
		}else{
			PSI_ERROR("Accessing global_comm object when MPI has not been setup");
			return decomp_.global_comm;
		}
	}

	decltype(Decomp::number_of_frequencies) global_number_of_frequencies() const { return decomp_.number_of_frequencies; }
	decltype(Decomp::frequencies) frequencies() const { return decomp_.frequencies; }

	decltype(FreqDecomp::number_of_time_blocks) global_number_of_time_blocks(int freq) const { return decomp_.frequencies[freq].number_of_time_blocks; }
	decltype(FreqDecomp::time_blocks) time_blocks(int freq) const { return decomp_.frequencies[freq].time_blocks; }

	decltype(Decomp::frequencies) my_frequencies() const { return my_decomp_.frequencies; }
	decltype(Decomp::number_of_frequencies) my_number_of_frequencies() const { return my_decomp_.number_of_frequencies; }

	decltype(Decomp::root_wavelet_comm) my_root_wavelet_comm() const { return my_decomp_.root_wavelet_comm; }

	decltype(Decomp::root_wavelets) global_root_wavelets() const { return decomp_.root_wavelets; }
	decltype(Decomp::number_of_root_wavelets) global_number_of_root_wavelets() const { return decomp_.number_of_root_wavelets; }

	decltype(Decomp::root_wavelets) my_root_wavelets() const { return my_decomp_.root_wavelets; }
	decltype(Decomp::number_of_root_wavelets) my_number_of_root_wavelets() const { return my_decomp_.number_of_root_wavelets; }
	decltype(Decomp::lower_root_wavelet) my_lower_root_wavelet() const { return my_decomp_.lower_root_wavelet; }


	decltype(Decomp::checkpointing) checkpointing() const { return decomp_.checkpointing; }
	decltype(Decomp::checkpointing) my_checkpointing() const { return my_decomp_.checkpointing; }

	void set_checkpointing(bool check) {decomp_.checkpointing = check; }
	void set_my_checkpointing(bool check) { my_decomp_.checkpointing = check; }

	decltype(Decomp::checkpointing_frequency) checkpointing_frequency() const { return decomp_.checkpointing_frequency; }
	decltype(Decomp::checkpointing_frequency) my_checkpointing_frequency() const { return my_decomp_.checkpointing_frequency; }

	void set_checkpointing_frequency(int freq)  { decomp_.checkpointing_frequency = freq; }
	void set_my_checkpointing_frequency(int freq) { my_decomp_.checkpointing_frequency = freq; }

	decltype(Decomp::restoring) restoring() const { return decomp_.restoring; }
	decltype(Decomp::restoring) my_restoring() const { return my_decomp_.restoring; }

	void set_restoring(bool restore) {decomp_.restoring = restore; }
	void set_my_restoring(bool restore) { my_decomp_.restoring = restore; }

	decltype(Decomp::checkpoint_restored) checkpoint_restored() const { return decomp_.checkpoint_restored; }
	decltype(Decomp::checkpoint_restored) my_checkpoint_restored() const { return my_decomp_.checkpoint_restored; }

	void set_checkpoint_restored(bool restored) {decomp_.checkpoint_restored = restored; }
	void set_my_checkpoint_restored(bool restored) { my_decomp_.checkpoint_restored = restored; }

	std::pair<int, int> number_and_start_of_local_wavelets(int number_of_wavelets, std::vector<WaveletDecomp> wavelets);
	bool own_this_frequency(int frequency);

	std::vector<Communicator> get_my_wavelet_comms() const { return my_decomp_.wavelet_comms; }
	std::vector<bool> get_my_wavelet_comms_involvement() const { return my_decomp_.wavelet_comms_involvement; }

protected:

	// Class data
	Decomp decomp_;
	Decomp my_decomp_;

private:

	void build_my_decomposition();
	void report_on_decomposition(bool quiet);



};

template <class T>
void Decomposition::collect_indices(std::vector<std::vector<T>> const indices, std::vector<std::vector<T>> &global_indices, bool global_root) const{

	if(decomp_.parallel_mpi){
		//! This two variables are used to keep track of our place in the global_indices and indices arrays because the
		//! likelihood is that we are not involved in all frequencies so do not have an global indices or indices vector that
		//! can be indexed using f from the f loop below.
		int my_indices_index = 0;
		bool freq_used = false;
		int global_freq_index = 0;
		for(int f=0; f<decomp_.number_of_frequencies; f++){
			int receiver_rank;
			if(global_root){
				receiver_rank = decomp_.global_comm.root_id();
			}else{
				receiver_rank = decomp_.frequencies[f].global_owner;
			}
			if(decomp_.frequencies[f].in_this_frequency or (global_root and decomp_.global_comm.is_root())){
				if((global_root and decomp_.global_comm.is_root()) or (not global_root and decomp_.frequencies[f].global_owner == decomp_.global_comm.rank())){
					int time_block_index = 0;
					for(int t=0; t<decomp_.frequencies[f].number_of_time_blocks; t++){
						//! freq_index is used so when not running with global_root we can track the frequencies this process owns
						//! so the global_indices are correct for this local mode.
						int freq_index = f;
						if(not global_root){
							freq_index = global_freq_index;
						}
						//! If this is data for the root freq process then just copy it to the new data structure, do not send.
						if(decomp_.frequencies[f].time_blocks[t].global_owner == decomp_.global_comm.rank()){

							global_indices[freq_index][t] = indices[my_indices_index][time_block_index];
							time_block_index++;
							freq_used = true;
							//! Otherwise, receive from the owning process
						}else{
							t_uint tag = 0;
							int temp_size;
							decomp_.global_comm.recv_single(&temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
							global_indices[freq_index][t] = T(temp_size);
							decomp_.global_comm.recv_eigen(global_indices[freq_index][t], decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						}
					}
					if(not global_root){
						global_freq_index++;
					}
					//! If I am not the root process then send the data I have.
				}else{
					int time_block_index = 0;
					for(int t = 0; t < decomp_.frequencies[f].number_of_time_blocks; t++){
						if(decomp_.frequencies[f].time_blocks[t].global_owner == decomp_.global_comm.rank()){
							t_uint tag = 0;
							int temp_size;
							temp_size = indices[my_indices_index][time_block_index].size();
							decomp_.global_comm.send_single(temp_size, receiver_rank, tag);
							decomp_.global_comm.send_eigen(indices[my_indices_index][time_block_index], receiver_rank, tag);
							time_block_index++;
							freq_used = true;
						}
					}
				}
				if(freq_used){
					my_indices_index++;
					freq_used = false;
				}
			}
		}
	}else{
		for(int f = 0; f < global_indices.size(); f++){
			for(int t = 0; t < global_indices[f].size(); t++){
				global_indices[f][t] = indices[f][t];
			}
		}
	}

	return;
}


template <class T>
void Decomposition::distribute_fourier_data(std::vector<std::vector<Eigen::SparseMatrix<T>>> &local_sparse, std::vector<std::vector<Eigen::SparseMatrix<T>>> &global_sparse) const{

	bool nonblocking = true;

	if(decomp_.parallel_mpi){
		int my_local_index = 0;
		int my_global_index = 0;
		for(int f = 0; f < decomp_.number_of_frequencies; f++){
			if(decomp_.frequencies[f].in_this_frequency or decomp_.global_comm.is_root()){
				if(decomp_.global_comm.is_root()){
					int my_index = 0;
					// Strictly speaking these aren't required unless nonblocking is being used, but currently it is
					// simpler to define them whatever.
					int shapes[decomp_.frequencies[f].number_of_time_blocks][3];
					MPI_Request requests[decomp_.frequencies[f].number_of_time_blocks*4];
					// Initialise the requests to MPI_REQUEST_NULL so the wait all works for blocks that are owned by root.
					for(int t = 0; t < decomp_.frequencies[f].number_of_time_blocks*4; t++){
						requests[t] = MPI_REQUEST_NULL;
					}
					for(int t = 0; t < decomp_.frequencies[f].number_of_time_blocks; t++){
						//! If this is data for the root process then just copy it to the new data structure, do not send.
						if(decomp_.frequencies[f].time_blocks[t].global_owner == decomp_.global_comm.rank()){
							local_sparse[my_local_index][my_index] = global_sparse[my_global_index][t];
							local_sparse[my_local_index][my_index].makeCompressed();
							my_index++;
							//! Otherwise, send to the owning process
						}else{
							if(nonblocking){
								shapes[t][0] = global_sparse[my_global_index][t].rows();
								shapes[t][1] = global_sparse[my_global_index][t].cols();
								shapes[t][2] = global_sparse[my_global_index][t].nonZeros();
								t_uint tag = 0;
								decomp_.global_comm.nonblocking_send_sparse_eigen(global_sparse[my_global_index][t], decomp_.frequencies[0].time_blocks[t].global_owner, tag, shapes[t], &requests[t*4]);
							}else{
								t_uint tag = 0;
								decomp_.global_comm.send_sparse_eigen(global_sparse[my_global_index][t], decomp_.frequencies[f].time_blocks[t].global_owner, tag);
							}
						}
					}
					if(nonblocking){
						decomp_.global_comm.wait_on_all(requests, 4*decomp_.frequencies[f].number_of_time_blocks);
					}
					my_global_index++;
					//! If I am not the root process for this frequency then wait to receive the data I am expecting.
				}else{
					for(int t = 0; t < my_decomp_.frequencies[f].number_of_time_blocks; t++){
						t_uint tag = 0;
						my_decomp_.global_comm.recv_sparse_eigen(local_sparse[my_local_index][t], my_decomp_.global_comm.root_id(), tag);
					}

				}
				my_local_index++;
			}
		}

	}else{
		for(int f = 0; f < global_sparse.size(); f++){
			for(int t = 0; t < global_sparse[f].size(); t++){
				local_sparse[f][t] = global_sparse[f][t];
				local_sparse[f][t].makeCompressed();
			}
		}
	}


	return;
}

// This routine receives an individual fourier data set for a process. It is designed to be used with non-blocking communications to enable the construction of the
// fourier data to be mixed with the sending and receiving of the fourier data.
template <class T>
void Decomposition::receive_fourier_data(std::vector<Eigen::SparseMatrix<T>> &local_sparse, int freq, int my_freq, bool global_root) const{

	bool nonblocking = true;

	if(decomp_.parallel_mpi){
		int sender_id;
		if(global_root){
			sender_id = decomp_.global_comm.root_id();
		}else{
			sender_id = my_decomp_.frequencies[my_freq].global_owner;
		}
		if(((not global_root) and my_decomp_.frequencies[my_freq].global_owner != decomp_.global_comm.rank()) or (global_root and not decomp_.global_comm.is_root())){
			for(int k=0; k<my_decomp_.frequencies[my_freq].number_of_time_blocks; k++){
				t_uint tag = 0;
				decomp_.global_comm.recv_sparse_eigen(local_sparse[k], sender_id, tag);
			}
		}else{
			PSI_ERROR("recv_fourier_data should only ever be called the process that isn't a global owner of a given frequency/global root (depending on how called), not any of the other processes. An error has occurred");

		}
	}

	return;
}

// This sends an individual fourier data set to a process. It is designed to be used with non-blocking communications to enable the construction of the
// fourier data to be mixed with the sending and receiving of the fourier data.
template <class T>
void Decomposition::send_fourier_data(std::vector<Eigen::SparseMatrix<T>> &local_sparse, std::vector<Eigen::SparseMatrix<T>> &global_sparse, int *shapes, int block, int &my_index, int freq, bool &used_this_freq, bool global_root) {


	if(decomp_.parallel_mpi){
		int global_freq = my_decomp_.frequencies[freq].freq_number;
		if((not global_root and decomp_.frequencies[global_freq].global_owner == decomp_.global_comm.rank()) or (global_root and decomp_.global_comm.is_root())){
			//! If this is data for the root process then just copy it to the new data structure, do not send.
			if((global_root and (decomp_.frequencies[global_freq].time_blocks[block].global_owner == decomp_.global_comm.rank())) or ((not global_root) and (decomp_.frequencies[global_freq].time_blocks[block].global_owner == decomp_.global_comm.rank()))){
				local_sparse[my_index] = global_sparse[block];
				local_sparse[my_index].makeCompressed();
				my_index++;
				used_this_freq = true;
				//! Otherwise, send to the owning process
			}else{
				shapes[0] = global_sparse[block].rows();
				shapes[1] = global_sparse[block].cols();
				shapes[2] = global_sparse[block].nonZeros();
				t_uint tag = 0;
				if(not decomp_.frequencies[freq].requests_initialised){
					PSI_ERROR("Going to call an nonblocking routine when the requests have not been initialised. An error has occurred");
				}
				int owner = decomp_.frequencies[global_freq].time_blocks[block].global_owner;
				decomp_.global_comm.nonblocking_send_sparse_eigen(global_sparse[block], owner, tag, shapes, &decomp_.frequencies[freq].requests[block*4]);
			}

		}else{
			PSI_ERROR("send_fourier_data should only ever be called by the global owner of a given frequency/global root (depending on how it is called), not any of the other processes. An error has occurred");
		}

	}

	return;
}


template <class T>
void Decomposition::collect_residual_norms(Vector<Vector<T>> const residual_norms, Vector<Vector<T>> &total_residual_norms) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			for(int j = 0; j < decomp_.number_of_frequencies; j++){
				int my_index = 0;
				for(int k = 0; k < decomp_.frequencies[j].number_of_time_blocks; k++){
					//! If this is data for the root process then just copy it to the new data structure, do not send.
					if(decomp_.frequencies[j].time_blocks[k].global_owner == decomp_.global_comm.rank()){
						total_residual_norms[j][k] = residual_norms[j][my_index];
						my_index++;
						//! Otherwise, send to the owning process
					}else{
						t_uint tag = decomp_.frequencies[j].freq_number;
						decomp_.global_comm.recv_single(&total_residual_norms[j][k], decomp_.frequencies[j].time_blocks[k].global_owner, tag);
					}
				}
			}
			//! If I am not the root process then wait to receive the data I am expecting.
		}else{
			for(int j = 0; j < my_decomp_.number_of_frequencies; j++){
				for(int k = 0; k < my_decomp_.frequencies[j].number_of_time_blocks; k++){
					t_uint tag = my_decomp_.frequencies[j].freq_number;
					decomp_.global_comm.send_single(residual_norms[j][k], decomp_.global_comm.root_id(), tag);
				}
			}
		}

	}else{

		for(int j = 0; j < decomp_.number_of_frequencies; j++){
			for(int k = 0; k < residual_norms.size(); k++){
				total_residual_norms[j][k] = residual_norms[j][k];
			}
		}
	}


	return;
}


template <class T>
void Decomposition::collect_residual_sizes(std::vector<std::vector<T>> const residual, std::vector<std::vector<int>> &sizes) const{
	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_index = 0;
			for(int j = 0; j < decomp_.number_of_frequencies; j++){
				for(int k = 0; k < decomp_.frequencies[j].number_of_time_blocks; k++){
					//! If this is data for the root process then just copy it to the new data structure, do not send.
					if(decomp_.frequencies[j].time_blocks[k].global_owner == decomp_.global_comm.rank()){
						sizes[j][k] = residual[my_index].size();
						my_index++;
						//! Otherwise, send to the owning process
					}else{
						t_uint tag = decomp_.frequencies[j].freq_number;;
						int temp_size;
						decomp_.global_comm.recv_single(&temp_size, decomp_.frequencies[j].time_blocks[k].global_owner, tag);
						sizes[j][k] = temp_size;
					}
				}
			}
			//! If I am not the root process then wait to receive the data I am expecting.
		}else{
			for(int j = 0; j < my_decomp_.number_of_frequencies; j++){
				for(int k = 0; k < my_decomp_.frequencies[j].number_of_time_blocks; k++){
					t_uint tag = decomp_.frequencies[j].freq_number;
					int temp_size = residual[j][k].size();
					decomp_.global_comm.send_single(temp_size, decomp_.global_comm.root_id(), tag);
				}
			}
		}

	}else{

		for(int j = 0; j < decomp_.number_of_frequencies; j++){
			for(int k = 0; k < residual.size(); k++){
				sizes[j][k] = residual[j][k].size();
			}
		}
	}
	return;
}

template <class T>
void Decomposition::collect_residuals(std::vector<T> const residual, std::vector<T> &total_residuals) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_index = 0;
			for(int k = 0; k < decomp_.frequencies[0].number_of_time_blocks; k++){
				//! If this is data for the root process then just copy it to the new data structure, do not send.
				if(decomp_.frequencies[0].time_blocks[k].global_owner == decomp_.global_comm.rank()){
					total_residuals[k] = residual[my_index];
					my_index++;
					//! Otherwise, send to the owning process
				}else{
					t_uint tag = 0;
					decomp_.global_comm.recv_eigen(total_residuals[k], decomp_.frequencies[0].time_blocks[k].global_owner, tag);
				}
			}
			//! If I am not the root process then wait to receive the data I am expecting.
		}else{
			for(int j = 0; j < my_decomp_.number_of_frequencies; j++){
				for(int k = 0; k < my_decomp_.frequencies[j].number_of_time_blocks; k++){
					t_uint tag = 0;
					decomp_.global_comm.send_eigen(residual[k], decomp_.global_comm.root_id(), tag);
				}
			}
		}

	}else{
		for(int k = 0; k < residual.size(); k++){
			total_residuals[k] = residual[k];
		}
	}


	return;
}

template <class T>
void Decomposition::collect_epsilons(T const epsilons, T &total_epsilons) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_time_index = 0;
			for(int k = 0; k < decomp_.frequencies[0].number_of_time_blocks; k++){
				//! If this is data for the root process then just copy it to the new data structure, do not send.
				if(decomp_.frequencies[0].time_blocks[k].global_owner == decomp_.global_comm.rank()){
					total_epsilons[k] = epsilons[my_time_index];
					my_time_index++;
					//! Otherwise, send to the owning process
				}else{
					t_uint tag = decomp_.frequencies[0].freq_number;
					decomp_.global_comm.recv_single(&total_epsilons[k], decomp_.frequencies[0].time_blocks[k].global_owner, tag);
				}
			}
			//! If I am not the root process then wait to receive the data I am expecting.
		}else{
			for(int k = 0; k < my_decomp_.frequencies[0].number_of_time_blocks; k++){
				t_uint tag = decomp_.frequencies[0].freq_number;
				decomp_.global_comm.send_single(epsilons[k], decomp_.global_comm.root_id(), tag);
			}
		}

	}else{
		for(int k = 0; k < epsilons.size(); k++){
			total_epsilons[k] = epsilons[k];
		}
	}


	return;
}


template <class T>
void Decomposition::distribute_epsilons(T &epsilons, T const total_epsilons) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_index = 0;
			for(int k = 0; k < decomp_.frequencies[0].number_of_time_blocks; k++){
				//! If this is data for the root process then just copy it to the new data structure, do not send.
				if(decomp_.frequencies[0].time_blocks[k].global_owner == decomp_.global_comm.rank()){
					epsilons[my_index] = total_epsilons[k];
					my_index++;
					//! Otherwise, send to the owning process
				}else{
					t_uint tag = 0;
					decomp_.global_comm.send_single(total_epsilons[k], decomp_.frequencies[0].time_blocks[k].global_owner, tag);
				}
			}
			//! If I am not the root process then wait to receive the data I am expecting.
		}else{
			for(int j = 0; j < my_decomp_.number_of_frequencies; j++){
				for(int k = 0; k < my_decomp_.frequencies[j].number_of_time_blocks; k++){
					t_uint tag = 0;
					decomp_.global_comm.recv_single(&epsilons[k], decomp_.global_comm.root_id(), tag);
				}
			}
		}

	}else{
		for(int k = 0; k < epsilons.size(); k++){
			epsilons[k] = total_epsilons[k];
		}
	}


	return;
}

template <class T>
void Decomposition::collect_epsilons_wideband_blocking(T const epsilons, T &total_epsilons) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int j = 0; j < decomp_.number_of_frequencies; j++){
				int my_time_index = 0;
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				for(int k = 0; k < decomp_.frequencies[j].number_of_time_blocks; k++){
					//! If this is data for the root process then just copy it to the new data structure, do not send.
					if(decomp_.frequencies[j].time_blocks[k].global_owner == decomp_.global_comm.rank()){
						total_epsilons[j][k] = epsilons[my_freq_index][my_time_index];
						my_time_index++;
						my_freq_used = true;
						//! Otherwise, send to the owning process
					}else{
						t_uint tag = decomp_.frequencies[j].freq_number;
						decomp_.global_comm.recv_single(&total_epsilons[j][k], decomp_.frequencies[j].time_blocks[k].global_owner, tag);
					}
				}
			}
			//! If I am not the root process then send my data to the root.
		}else{
			for(int j = 0; j < my_decomp_.number_of_frequencies; j++){
				for(int k = 0; k < my_decomp_.frequencies[j].number_of_time_blocks; k++){
					t_uint tag = my_decomp_.frequencies[j].freq_number;
					decomp_.global_comm.send_single(epsilons[j][k], decomp_.global_comm.root_id(), tag);
				}
			}
		}

	}else{
		for(int j = 0; j < epsilons.size(); j++){
			for(int k = 0; k < epsilons[j].size(); k++){
				total_epsilons[j][k] = epsilons[j][k];
			}
		}
	}

	return;
}


template <class T>
void Decomposition::distribute_epsilons_wideband_blocking(T &epsilons, T const total_epsilons) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int j = 0; j < decomp_.number_of_frequencies; j++){
				int my_time_index = 0;
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				for(int k = 0; k < decomp_.frequencies[j].number_of_time_blocks; k++){
					//! If this is data for the root process then just copy it to the new data structure, do not send.
					if(decomp_.frequencies[j].time_blocks[k].global_owner == decomp_.global_comm.rank()){
						epsilons[my_freq_index][my_time_index] = total_epsilons[j][k];
						my_time_index++;
						my_freq_used = true;
						//! Otherwise, send to the owning process
					}else{
						t_uint tag = decomp_.frequencies[j].freq_number;
						decomp_.global_comm.send_single(total_epsilons[j][k], decomp_.frequencies[j].time_blocks[k].global_owner, tag);
					}
				}
			}
			//! If I am not the root process then wait to receive the data I am expecting.
		}else{
			for(int j = 0; j < my_decomp_.number_of_frequencies; j++){
				for(int k = 0; k < my_decomp_.frequencies[j].number_of_time_blocks; k++){
					t_uint tag = my_decomp_.frequencies[j].freq_number;
					decomp_.global_comm.recv_single(&epsilons[j][k], decomp_.global_comm.root_id(), tag);
				}
			}
		}

	}else{
		for(int j = 0; j < epsilons.size(); j++){
			for(int k = 0; k < epsilons[j].size(); k++){
				epsilons[j][k] = total_epsilons[j][k];
			}
		}
	}

	return;
}

template <class T>
void Decomposition::collect_frequency_root_data(Matrix<T> const frequency_data, Matrix<T> &total_data) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f=0; f<decomp_.number_of_frequencies; f++){
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				//! If this is data for the root process then just copy it to the new data structure, do not send.
				if(decomp_.frequencies[f].global_owner == decomp_.global_comm.rank()){
					for(int i=0; i<frequency_data.rows(); i++){
						total_data(i,f) = frequency_data(i,my_freq_index);
					}
					my_freq_used = true;
					//! Otherwise, send to the owning process
				}else{
					//! TODO: Optimise the receive below so the temp_vector is not needed
					t_uint tag = decomp_.frequencies[f].freq_number;
					int data_size;
					decomp_.global_comm.recv_single(&data_size, decomp_.frequencies[f].global_owner, tag);
					Vector<T> temp_vector(data_size);
					decomp_.global_comm.recv_eigen(temp_vector, decomp_.frequencies[f].global_owner, tag);
					total_data.col(f) = temp_vector;
				}

			}
			//! If I am not the root process but a frequency owner send my data to the root.
		}else{
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f = 0; f < my_decomp_.number_of_frequencies; f++){
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				if(my_decomp_.frequencies[f].global_owner == my_decomp_.global_comm.rank()){
					//! TODO: Optimise the send below so the temp_vector is not needed
					t_uint tag = my_decomp_.frequencies[f].freq_number;
					Vector<T> temp_vector = frequency_data.col(my_freq_index);
					int data_size = temp_vector.size();
					my_decomp_.global_comm.send_single(data_size, my_decomp_.global_comm.root_id(), tag);
					my_decomp_.global_comm.send_eigen(temp_vector, my_decomp_.global_comm.root_id(), tag);
					my_freq_used = true;
				}
			}
		}

	}else{
		for(int f = 0; f < total_data.size(); f++){
			total_data(f) = frequency_data(f);
		}
	}

	return;
}

template <class T1, class T2>
void Decomposition::distribute_frequency_data(T1 &frequency_data, T1 const total_data, const bool freq_root_only) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f = 0; f < decomp_.number_of_frequencies; f++){
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				//! If this is data for the root process then just copy it to the new data structure, do not send.
				if(decomp_.frequencies[f].global_owner == decomp_.global_comm.rank()){
					for(int i=0; i<frequency_data.rows(); i++){
						frequency_data(i,my_freq_index) = total_data(i,f);
					}
					my_freq_used = true;
					if(not freq_root_only){
						Vector<T2> temp_vector = total_data.col(f);
						int data_size = temp_vector.size();
						data_size = my_decomp_.frequencies[my_freq_index].freq_comm.broadcast(data_size, my_decomp_.frequencies[my_freq_index].freq_comm.root_id());
						my_decomp_.frequencies[my_freq_index].freq_comm.broadcast(temp_vector, my_decomp_.frequencies[my_freq_index].freq_comm.root_id());
					}
					//! Otherwise, send to the owning process
				}else{
					//! TODO: Optimise the send below so the temp_vector is not needed
					t_uint tag = decomp_.frequencies[f].freq_number;
					Vector<T2> temp_vector = total_data.col(f);
					int data_size = temp_vector.size();
					decomp_.global_comm.send_single(data_size, decomp_.frequencies[f].global_owner, tag);
					decomp_.global_comm.send_eigen(temp_vector, decomp_.frequencies[f].global_owner, tag);
				}

			}
			//! If I am not the root process then wait to receive the data I am expecting.
		}else{
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f = 0; f < my_decomp_.number_of_frequencies; f++){
				if(my_decomp_.frequencies[f].global_owner == my_decomp_.global_comm.rank()){
					if(my_freq_used){
						my_freq_index++;
						my_freq_used = false;
					}
					//! TODO: Optimise the receive below so the temp_vector is not needed
					int data_size;
					t_uint tag = my_decomp_.frequencies[f].freq_number;
					my_decomp_.global_comm.recv_single(&data_size, my_decomp_.global_comm.root_id(), tag);
					Vector<T2> temp_vector(data_size);
					my_decomp_.global_comm.recv_eigen(temp_vector, my_decomp_.global_comm.root_id(), tag);
					frequency_data.col(my_freq_index) = temp_vector;
					my_freq_used = true;
					if(not freq_root_only){
						data_size = my_decomp_.frequencies[my_freq_index].freq_comm.broadcast(data_size, my_decomp_.frequencies[my_freq_index].freq_comm.root_id());
						temp_vector = my_decomp_.frequencies[my_freq_index].freq_comm.broadcast(temp_vector, my_decomp_.frequencies[my_freq_index].freq_comm.root_id());
					}
				}else{
					if(not freq_root_only){
						int data_size;
						data_size = my_decomp_.frequencies[my_freq_index].freq_comm.broadcast(data_size, my_decomp_.frequencies[my_freq_index].freq_comm.root_id());
						Vector<T2> temp_vector(data_size);
						temp_vector = my_decomp_.frequencies[my_freq_index].freq_comm.broadcast(temp_vector, my_decomp_.frequencies[my_freq_index].freq_comm.root_id());
						frequency_data.col(f) = temp_vector;
					}
				}
			}
		}

	}else{
		for(int f = 0; f < frequency_data.size(); f++){
			frequency_data(f) = total_data(f);
		}
	}

	return;
}

template <class T>
void Decomposition::collect_svd_data(Matrix<T> const local_VT, Matrix<T> &total_VT, Matrix<T> const local_data_svd, Matrix<T> &total_data_svd) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f=0; f<decomp_.number_of_frequencies; f++){
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				//! If this is data for the root process then just copy it to the new data structure, do not send.
				if(decomp_.frequencies[f].global_owner == decomp_.global_comm.rank()){
					for(int i=0; i<local_VT.rows(); i++){
						total_VT(i,f) = local_VT(i,my_freq_index);
					}
					for(int i=0; i<local_data_svd.rows(); i++){
						total_data_svd(i,f) = local_data_svd(i,my_freq_index);
					}
					my_freq_used = true;
					//! Otherwise, send to the owning process
				}else{
					//! TODO: Optimise the receive below so the temp_vector is not needed
					t_uint tag = decomp_.frequencies[f].freq_number;
					int data_size;
					decomp_.global_comm.recv_single(&data_size, decomp_.frequencies[f].global_owner, tag);
					Vector<T> temp_vector(data_size);
					decomp_.global_comm.recv_eigen(temp_vector, decomp_.frequencies[f].global_owner, tag);
					total_VT.col(f) = temp_vector;
					decomp_.global_comm.recv_single(&data_size, decomp_.frequencies[f].global_owner, tag);
					Vector<T> temp_vector_2(data_size);
					decomp_.global_comm.recv_eigen(temp_vector_2, decomp_.frequencies[f].global_owner, tag);
					total_data_svd.col(f) = temp_vector_2;
				}

			}
			//! If I am not the root process but a frequency owner send my data to the root.
		}else{
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f = 0; f < my_decomp_.number_of_frequencies; f++){
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				if(my_decomp_.frequencies[f].global_owner == my_decomp_.global_comm.rank()){
					//! TODO: Optimise the send below so the temp_vector is not needed
					t_uint tag = my_decomp_.frequencies[f].freq_number;
					Vector<T> temp_vector = local_VT.col(my_freq_index);
					int data_size = temp_vector.size();
					my_decomp_.global_comm.send_single(data_size, my_decomp_.global_comm.root_id(), tag);
					my_decomp_.global_comm.send_eigen(temp_vector, my_decomp_.global_comm.root_id(), tag);
					Vector<T> temp_vector_2 = local_data_svd.col(my_freq_index);
					data_size = temp_vector_2.size();
					my_decomp_.global_comm.send_single(data_size, my_decomp_.global_comm.root_id(), tag);
					my_decomp_.global_comm.send_eigen(temp_vector_2, my_decomp_.global_comm.root_id(), tag);
					my_freq_used = true;
				}
			}
		}

	}else{
		for(int f = 0; f < total_VT.size(); f++){
			total_VT(f) = local_VT(f);
		}
		for(int f = 0; f < total_data_svd.size(); f++){
			total_data_svd(f) = local_data_svd(f);
		}
	}

	return;
}

template <class T1, class T2>
void Decomposition::distribute_svd_data(T1 &local_VT, T1 const total_VT, T1 &local_data_svd, Vector<T2> total_data_svd, int const image_size) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			Eigen::Map<T1> total_data_svd_matrix(total_data_svd.data(), image_size, decomp_.number_of_frequencies);

			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f = 0; f < decomp_.number_of_frequencies; f++){
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				//! If this is data for the root process then just copy it to the new data structure, do not send.
				if(decomp_.frequencies[f].global_owner == decomp_.global_comm.rank()){
					for(int i=0; i<total_VT.rows(); i++){
						local_VT(i, my_freq_index) = total_VT(i, f);
					}
					for(int i=0; i<total_data_svd_matrix.rows(); i++){
						local_data_svd(i, my_freq_index) = total_data_svd_matrix(i, f);
					}
					my_freq_used = true;
					//! Otherwise, send to the owning process
				}else{
					//! TODO: Optimise the send below so the temp_vector is not needed
					t_uint tag = decomp_.frequencies[f].freq_number;
					Vector<T2> temp_vector = total_VT.col(f);
					int data_size = temp_vector.size();
					decomp_.global_comm.send_single(data_size, decomp_.frequencies[f].global_owner, tag);
					decomp_.global_comm.send_eigen(temp_vector, decomp_.frequencies[f].global_owner, tag);
					Vector<T2> temp_vector_2 = total_data_svd_matrix.col(f);
					data_size = temp_vector_2.size();
					decomp_.global_comm.send_single(data_size, decomp_.frequencies[f].global_owner, tag);
					decomp_.global_comm.send_eigen(temp_vector_2, decomp_.frequencies[f].global_owner, tag);
				}

			}
			//! If I am not the root process then wait to receive the data I am expecting.
		}else{
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f = 0; f < my_decomp_.number_of_frequencies; f++){
				if(my_decomp_.frequencies[f].global_owner == my_decomp_.global_comm.rank()){
					if(my_freq_used){
						my_freq_index++;
						my_freq_used = false;
					}
					//! TODO: Optimise the receive below so the temp_vector is not needed
					int data_size;
					t_uint tag = my_decomp_.frequencies[f].freq_number;
					my_decomp_.global_comm.recv_single(&data_size, my_decomp_.global_comm.root_id(), tag);
					Vector<T2> temp_vector(data_size);
					my_decomp_.global_comm.recv_eigen(temp_vector, my_decomp_.global_comm.root_id(), tag);
					local_VT.col(my_freq_index) = temp_vector;
					my_decomp_.global_comm.recv_single(&data_size, my_decomp_.global_comm.root_id(), tag);
					Vector<T2> temp_vector_2(data_size);
					my_decomp_.global_comm.recv_eigen(temp_vector_2, my_decomp_.global_comm.root_id(), tag);
					local_data_svd.col(my_freq_index) = temp_vector_2;
					my_freq_used = true;
				}
			}
		}

	}else{
		for(int f = 0; f < total_VT.size(); f++){
			local_VT(f) = total_VT(f);
		}
		for(int f = 0; f < total_data_svd.size(); f++){
			local_data_svd(f) = total_data_svd(f);
		}
	}

	return;
}

template <class T1, class T2>
void Decomposition::collect_svd_result_data(Matrix<T1> const local_p, Matrix<T2> &total_p) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f=0; f<decomp_.number_of_frequencies; f++){
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				//! If this is data for the root process then just copy it to the new data structure, do not send.
				if(decomp_.frequencies[f].global_owner == decomp_.global_comm.rank()){
					for(int i=0; i<local_p.rows(); i++){
						total_p(i,f).real(local_p(i,my_freq_index));
					}
					my_freq_used = true;
					//! Otherwise, send to the owning process
				}else{
					//! TODO: Optimise the receive below so the temp_vector is not needed
					t_uint tag = decomp_.frequencies[f].freq_number;
					int data_size;
					decomp_.global_comm.recv_single(&data_size, decomp_.frequencies[f].global_owner, tag);
					Vector<T1> temp_vector(data_size);
					decomp_.global_comm.recv_eigen(temp_vector, decomp_.frequencies[f].global_owner, tag);
					for(int i=0; i<temp_vector.size(); i++){
						total_p(i, f).real(temp_vector(i));
					}
				}

			}
			//! If I am not the root process but a frequency owner send my data to the root.
		}else{
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f = 0; f < my_decomp_.number_of_frequencies; f++){
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				if(my_decomp_.frequencies[f].global_owner == my_decomp_.global_comm.rank()){
					//! TODO: Optimise the send below so the temp_vector is not needed
					t_uint tag = my_decomp_.frequencies[f].freq_number;
					Vector<T1> temp_vector = local_p.col(my_freq_index);
					int data_size = temp_vector.size();
					my_decomp_.global_comm.send_single(data_size, my_decomp_.global_comm.root_id(), tag);
					my_decomp_.global_comm.send_eigen(temp_vector, my_decomp_.global_comm.root_id(), tag);
					my_freq_used = true;
				}
			}
		}

	}else{
		for(int f = 0; f < total_p.size(); f++){
			total_p(f).real(local_p(f));
		}
	}

	return;
}

template <class T1, class T2>
void Decomposition::distribute_svd_result_data(Matrix<T1> &local_p, Matrix<T2> const total_p) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){

			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f = 0; f < decomp_.number_of_frequencies; f++){
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				//! If this is data for the root process then just copy it to the new data structure, do not send.
				if(decomp_.frequencies[f].global_owner == decomp_.global_comm.rank()){
					for(int i=0; i<total_p.rows(); i++){
						local_p(i, my_freq_index) = total_p(i, f).real();
					}
					my_freq_used = true;
					//! Otherwise, send to the owning process
				}else{
					//! TODO: Optimise the send below so the temp_vector is not needed
					t_uint tag = decomp_.frequencies[f].freq_number;
					Vector<T1> temp_vector = total_p.col(f).real();
					int data_size = temp_vector.size();
					decomp_.global_comm.send_single(data_size, decomp_.frequencies[f].global_owner, tag);
					decomp_.global_comm.send_eigen(temp_vector, decomp_.frequencies[f].global_owner, tag);
				}

			}
			//! If I am not the root process then wait to receive the data I am expecting.
		}else{
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f = 0; f < my_decomp_.number_of_frequencies; f++){
				if(my_decomp_.frequencies[f].global_owner == my_decomp_.global_comm.rank()){
					if(my_freq_used){
						my_freq_index++;
						my_freq_used = false;
					}
					//! TODO: Optimise the receive below so the temp_vector is not needed
					int data_size;
					t_uint tag = my_decomp_.frequencies[f].freq_number;
					my_decomp_.global_comm.recv_single(&data_size, my_decomp_.global_comm.root_id(), tag);
					Vector<T1> temp_vector(data_size);
					my_decomp_.global_comm.recv_eigen(temp_vector, my_decomp_.global_comm.root_id(), tag);
					local_p.col(my_freq_index) = temp_vector;
					my_freq_used = true;
				}
			}
		}

	}else{
		for(int f = 0; f < total_p.size(); f++){
			local_p(f) = total_p(f).real();
		}
	}

	return;
}


template <class T>
void Decomposition::collect_wavelet_root_data(Matrix<T> const wavelet_data, Matrix<T> &total_data, int const image_size) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f=0; f<decomp_.number_of_frequencies; f++){
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				//! If this is data for the root process then just copy it to the new data structure, do not send.
				if(decomp_.frequencies[f].global_owner == decomp_.global_comm.rank()){
					int my_wavelet_index = 0;
					bool my_wavelet_used =  false;
					for(int i=0; i<decomp_.number_of_root_wavelets; i++){
						if(my_wavelet_used){
							my_wavelet_index++;
							my_wavelet_used = false;
						}
						if(decomp_.root_wavelets[i].global_owner == decomp_.global_comm.rank()){
							total_data.block(image_size*i, f, image_size, 1) = wavelet_data.block(image_size*my_wavelet_index, my_freq_index, image_size, 1);
							my_wavelet_used = true;
						}
					}
					my_freq_used = true;
					//! Otherwise, send to the owning process
				}else{
					//! TODO: Optimise the receive below so the temp_vector is not needed
					t_uint tag = decomp_.frequencies[f].freq_number;
					int offset;
					decomp_.global_comm.recv_single(&offset, decomp_.frequencies[f].global_owner, tag);
					int data_size;
					decomp_.global_comm.recv_single(&data_size, decomp_.frequencies[f].global_owner, tag);
					Vector<T> temp_vector(data_size);
					decomp_.global_comm.recv_eigen(temp_vector, decomp_.frequencies[f].global_owner, tag);
					total_data.block(offset, f, data_size, 1) = temp_vector;
				}

			}
			//! If I am not the root process but a frequency owner send my data to the root.
		}else{
			int my_freq_index = 0;
			bool my_freq_used = false;
			for(int f = 0; f < my_decomp_.number_of_frequencies; f++){
				if(my_freq_used){
					my_freq_index++;
					my_freq_used = false;
				}
				if(my_decomp_.frequencies[f].global_owner == my_decomp_.global_comm.rank()){
					//! TODO: Optimise the send below so the temp_vector is not needed
					t_uint tag = my_decomp_.frequencies[f].freq_number;
					Vector<T> temp_vector = wavelet_data.col(my_freq_index);
					int offset = -1;
					for(int i=0; i<decomp_.number_of_root_wavelets; i++){
						if(decomp_.root_wavelets[i].global_owner == decomp_.global_comm.rank()){
							offset = i;
							break;
						}
					}
					offset = offset * image_size;
					my_decomp_.global_comm.send_single(offset, my_decomp_.global_comm.root_id(), tag);
					int data_size = temp_vector.size();
					my_decomp_.global_comm.send_single(data_size, my_decomp_.global_comm.root_id(), tag);
					my_decomp_.global_comm.send_eigen(temp_vector, my_decomp_.global_comm.root_id(), tag);
					my_freq_used = true;
				}
			}
		}

	}else{
		for(int f = 0; f < total_data.size(); f++){
			total_data(f) = wavelet_data(f);
		}
	}

	return;
}



template <class T>
void Decomposition::collect_l1_weights(T const l1_weights, T &total_l1_weights, t_uint image_size) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_index = 0;
			for(int k = 0; k < decomp_.frequencies[0].number_of_wavelets; k++){
				//! If this is data is from the root process then just copy it to the new data structure/
				if(decomp_.frequencies[0].wavelets[k].global_owner == decomp_.global_comm.rank()){
					for(int j = 0; j < image_size; j++){
						total_l1_weights[(k*image_size)+j] = l1_weights[my_index];
						my_index++;
					}
					//! Otherwise, recv from the owning process
				}else{
					t_uint tag = 0;
					T temp(image_size);
					decomp_.global_comm.recv_eigen(temp, decomp_.frequencies[0].wavelets[k].global_owner, tag);
					total_l1_weights.segment(k*image_size,image_size) = temp;
				}
			}
			//! If I am not the root process then wait to send the data I have.
		}else{
			for(int j = 0; j < my_decomp_.number_of_frequencies; j++){
				for(int k = 0; k < my_decomp_.frequencies[j].number_of_wavelets; k++){
					t_uint tag = 0;
					decomp_.global_comm.send_eigen(l1_weights.segment(k*image_size,image_size), decomp_.global_comm.root_id(), tag);
				}
			}
		}

	}else{
		for(int k = 0; k < total_l1_weights.size(); k++){
			total_l1_weights[k] = l1_weights[k];
		}
	}

	return;
}

template <class T>
void Decomposition::distribute_l1_weights(T &l1_weights, T const total_l1_weights, t_uint image_size) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_index = 0;
			for(int k = 0; k < decomp_.frequencies[0].number_of_wavelets; k++){
				//! If this is data for the root process then just copy it to the new data structure, do not send.
				if(decomp_.frequencies[0].wavelets[k].global_owner == decomp_.global_comm.rank()){
					for(int j = 0; j < image_size; j++){
						l1_weights[my_index] = total_l1_weights[(k*image_size)+j];
						my_index++;
					}
					//! Otherwise, send to the owning process
				}else{
					t_uint tag = 0;
					decomp_.global_comm.send_eigen(total_l1_weights.segment(k*image_size,image_size), decomp_.frequencies[0].wavelets[k].global_owner, tag);
				}
			}
			//! If I am not the root process then wait to receive the data I am expecting.
		}else{
			for(int j = 0; j < my_decomp_.number_of_frequencies; j++){
				for(int k = 0; k < my_decomp_.frequencies[j].number_of_wavelets; k++){
					t_uint tag = 0;
					T temp(image_size);
					decomp_.global_comm.recv_eigen(temp, decomp_.global_comm.root_id(), tag);
					l1_weights.segment(k*image_size,image_size) = temp;
				}
			}
		}

	}else{
		for(int k = 0; k < total_l1_weights.size(); k++){
			l1_weights[k] = total_l1_weights[k];
		}
	}

	return;
}

//! We are assuming the global root is psi_root owner
template <class T>
void Decomposition::collect_l21_weights(T const l21_weights, T &total_l21_weights, t_uint image_size) const{


	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_index = 0;
			for(int k = 0; k < decomp_.number_of_root_wavelets; k++){
				//! If this is data is from the root process then just copy it to the new data structure/
				if(decomp_.root_wavelets[k].global_owner == decomp_.global_comm.rank()){
					for(int j = 0; j < image_size; j++){
						total_l21_weights[(k*image_size)+j] = l21_weights[my_index];
						my_index++;
					}
					//! Otherwise, recv from the owning process
				}else{
					t_uint tag = 0;
					T temp(image_size);
					decomp_.global_comm.recv_eigen(temp, decomp_.root_wavelets[k].global_owner, tag);
					total_l21_weights.segment(k*image_size,image_size) = temp;
				}
			}
			//! If I am not the root process then wait to send the data I have.
		}else{
			for(int k = 0; k < my_decomp_.number_of_root_wavelets; k++){
				t_uint tag = 0;
				decomp_.global_comm.send_eigen(l21_weights.segment(k*image_size,image_size), decomp_.global_comm.root_id(), tag);
			}
		}

	}else{
		for(int k = 0; k < total_l21_weights.size(); k++){
			total_l21_weights[k] = l21_weights[k];
		}
	}


	return;
}


template <class T>
void Decomposition::distribute_l21_weights(T &l21_weights, T const total_l21_weights, t_uint image_size) const{

	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){
			int my_index = 0;
			for(int k = 0; k < decomp_.number_of_root_wavelets; k++){
				//! If this is data for the root process then just copy it to the new data structure, do not send.
				if(decomp_.root_wavelets[k].global_owner == decomp_.global_comm.rank()){
					for(int j = 0; j < image_size; j++){
						l21_weights[my_index] = total_l21_weights[(k*image_size)+j];
						my_index++;
					}
					//! Otherwise, send to the owning process
				}else{
					t_uint tag = 0;
					decomp_.global_comm.send_eigen(total_l21_weights.segment(k*image_size,image_size), decomp_.root_wavelets[k].global_owner, tag);
				}
			}
			//! If I am not the root process then wait to receive the data I am expecting.
		}else{
			for(int k = 0; k < my_decomp_.number_of_root_wavelets; k++){
				t_uint tag = 0;
				T temp(image_size);
				decomp_.global_comm.recv_eigen(temp, decomp_.global_comm.root_id(), tag);
				l21_weights.segment(k*image_size,image_size) = temp;
			}
		}

	}else{
		for(int k = 0; k < total_l21_weights.size(); k++){
			l21_weights[k] = total_l21_weights[k];
		}
	}

	return;

}

} // namespace Decomposition
} // namespace psi
#endif /* ifndef PSI_MPI_DECOMPOSITION */
