#include <exception>
#include <mpi.h>
#include "psi/mpi/decomposition.h"
#include "psi/logging.h"
#include <iostream>
#ifdef PSI_OPENMP
#include <omp.h>
#endif

namespace psi {
namespace mpi {
//! Constuctor for the class
//! Initialise with an empty MPI communicator by default so this can be called when run in serial (done in header).
Decomposition::Decomposition(const bool parallel,  Communicator comm_world){
	decomp_ = Decomp();
	decomp_.parallel_mpi = parallel;
	if(parallel){
		decomp_.global_comm = comm_world;
	}
	my_decomp_ = Decomp();
	my_decomp_.parallel_mpi = parallel;
	if(parallel){
		my_decomp_.global_comm = comm_world;
	}
}

//! Calculate the global decomposition of the problem (mapping of data to each process) for distributed memory computations (i.e. MPI).
//! We have four potential dimensions to distribute over:
//! - Measurement Operator:
//! -  - Frequencies
//! -  - Time blocks per frequency
//! -  - Sub-block per time block
//! - Bases
//! -  - Wavelets
//! The wavelet decomposition is independent of the other three. For the measurement operator decompositions the general hierarchy
//! that is followed is that frequency decomposition is preferable over Time block decomposition, which in term is preferable over
//! sub-block decomposition. Therefore, by default the algorithm will try to decompose over frequencies first, then time blocks, then
//! sub-blocks, stopping when available processes are exhausted. However, the user can override this functionality by specifying whether
//! any given decomposition is enabled (through the boolean variables passed to this routine).
void Decomposition::decompose_primal_dual(bool const freq_decomp, bool const time_decomp, bool const subblock_decomp, bool const wavelet_decomp,  bool const wavelet_root_decomp, t_int const input_frequencies, std::vector<t_int> input_wavelet_levels, std::vector<t_int> input_time_blocks, std::vector<std::vector<t_int>> input_sub_blocks, bool quiet){

	int padding_processes = decomp_.global_comm.number_of_padding_processes();

	decomp_.wavelet_decomp = wavelet_decomp;
	decomp_.freq_decomp = freq_decomp;
	decomp_.time_decomp = time_decomp;
	decomp_.sub_decomp = subblock_decomp;
	decomp_.root_wavelet_decomp = wavelet_root_decomp;

	if(input_frequencies > 1 && time_decomp && !freq_decomp){
		decomp_.global_comm.abort("Multiple frequencies present but we are parallelising over time blocks rather than frequencies. This is not currently supported.");
	}

	if(subblock_decomp){
		decomp_.global_comm.abort("Sub-block decomposition has be requested. This is not currently supported.");
	}

	auto const mpi_size = decomp_.global_comm.size();
	auto const mpi_rank = decomp_.global_comm.rank();

	decomp_.number_of_frequencies = input_frequencies;
	decomp_.lower_freq = 0;
	decomp_.upper_freq = input_frequencies;

	int idle;
	int wavelet_processes;
	int lower_block_bound;
	int upper_block_bound;

	if(freq_decomp){

		decompose_frequencies(input_frequencies, padding_processes, mpi_size-1, time_decomp, subblock_decomp, wavelet_decomp);

	}else{

		//! If we are not doing a frequency decomposition then the number of frequencies should be 1.
		if(input_frequencies != 1){
			decomp_.global_comm.abort("Multiple frequencies present but are not parallelising over frequencies. This is not currently supported.");
		}
		decomp_.frequencies = std::vector<FreqDecomp>(decomp_.number_of_frequencies);

		//! Setup the time block for the single frequency we have.
		lower_block_bound = 0;
		upper_block_bound = 0;
		decomp_.frequencies[0].freq_number = 0;
		decomp_.frequencies[0].number_of_time_blocks = input_time_blocks[0];
		decomp_.frequencies[0].global_owner = decomp_.global_comm.root_id();
		decomp_.frequencies[0].process_ids = std::vector<t_uint>(mpi_size);
		for(int j=padding_processes; j<mpi_size; ++j){
			decomp_.frequencies[0].process_ids[j] = j;
		}
		decomp_.frequencies[0].lower_process = padding_processes;
		decomp_.frequencies[0].upper_process = mpi_size-1;

	}

	for(int i=0; i<input_frequencies; i++){
		decompose_time_blocks(i, input_time_blocks[i], time_decomp);
	}

	for(int i=0; i<input_frequencies; i++){
		decompose_wavelets(i, input_wavelet_levels[i], wavelet_decomp);
	}

	decompose_root_wavelets(input_wavelet_levels[0], decomp_.root_wavelet_decomp);

	//! Now we have finished calculating the overall decomposition build the local decomposition of this process so it
	//! knows what parts of the calculation it is involved with.
	build_my_decomposition();

	report_on_decomposition(quiet);


}

//! Calculate what frequencies a processor is assigned.
int Decomposition::decompose_frequencies(t_uint number_of_frequencies, t_uint lower_process, t_uint upper_process, bool time_decomp, bool subblock_decomp, bool wavelet_decomp){

	//! We add one on here to account for the calculation being inclusive, i.e. including the top number, so if we have 8 processes then the
	//! top number is 7, the lower number is 0, which would mean the number_of_processes would equal 7 if we didn't add one.
	t_uint number_of_processes = (upper_process - lower_process) + 1;

	//! We only have a single frequency here
	//! Time block decomp data, i.e. time blocks this process is involved with from this frequency
	decomp_.frequencies = std::vector<FreqDecomp>(number_of_frequencies);

	t_uint available_processes = number_of_processes;
	t_uint large_block_size = 0;
	t_uint small_block_size = 0;
	t_uint large_block_limit = 0;

	//! If we have less processes available than frequencies then processes will need to own more than one frequencies.
	if(available_processes <= number_of_frequencies){

		small_block_size = number_of_frequencies/available_processes;
		large_block_size = small_block_size;
		large_block_limit = -1;
		//! We do this to make sure that the decomposition can handle when the number of frequencies does not divide evenly between processes
		//! The approach below rounds up so that
		if(number_of_frequencies % available_processes != 0){
			large_block_size = large_block_size + 1;
			for(int i = 0; i < available_processes; i++){
				int potential_size = (i+1)*large_block_size + ((available_processes - (i+1))*small_block_size);
				if(potential_size == number_of_frequencies or potential_size == number_of_frequencies - 1 or potential_size == number_of_frequencies + 1){
					large_block_limit = i;
					break;
				}
			}
		}else{
			//! Every block is a large (or small) block as they are both the same size
			large_block_limit = available_processes;
		}
		//! If our calculation loop above went wrong then default to large blocks all the way. This is
		//! likely to leave idle processes.
		if(large_block_limit == -1){
			large_block_limit = available_processes;
			PSI_HIGH_LOG("Problem calculating frequency decomposition. It is likely to be sub-optimal and you would be advised to try with a different number of processes");
		}
		int current_frequency = 0;
		t_uint loop_limit = 0;
		for(int i=0; i<available_processes; ++i){
			//! If this is the last process then give it all the remaining frequencies so we
			//! do not leave frequencies unassigned (this takes account of the +-1 in the potential_size loop above)
			if(i == available_processes - 1){
				loop_limit = number_of_frequencies;
			}else{
				//! Set the maximum number of frequencies for this process
				if(i <= large_block_limit){
					loop_limit = loop_limit + large_block_size;
				}else{
					loop_limit = loop_limit + small_block_size;
				}
			}
			//! Assign the frequencies to the current process
			for(int j = current_frequency; j < std::min(loop_limit,number_of_frequencies); j++){
				decomp_.frequencies[current_frequency].freq_number = current_frequency;
				decomp_.frequencies[current_frequency].global_owner = i + lower_process;
				decomp_.frequencies[current_frequency].lower_process = i + lower_process;
				decomp_.frequencies[current_frequency].upper_process = i + lower_process;
				current_frequency = current_frequency + 1;
			}
		}

		available_processes = 0;
		if(current_frequency != number_of_frequencies){
			decomp_.global_comm.abort("An error has occurred in the frequency decomposition functionality. We have not given out all the frequencies but have run out of processes.");
		}

	}else{

		//! If there are more processes than frequencies then we try and give an equal chunk of processes to each frequency
		//! This assumes that wavelet, time block, or sub block decompositions are being undertaken to use these extra processes
		//! in each frequency, otherwise there will be idle processes

		if(not time_decomp and not subblock_decomp and not wavelet_decomp){
			PSI_HIGH_LOG("More processes than frequencies, but no other parallelisation enabled so there will be idle processes. This is likely to be sub-optimal and you would be advised to try with a different number of processes");
		}
		//! The number of processes exactly divides the number of frequencies so we can give the same number of processes to each frequency
		if(available_processes % number_of_frequencies  == 0){
			large_block_size = available_processes/number_of_frequencies;
			small_block_size = available_processes/number_of_frequencies;
		}else{
			//! The number of processes does not exactly divide the number of frequencies so we need to give out blocks of processes to frequencies of
			//! varied sizes
			large_block_size = (available_processes/number_of_frequencies)+1;
			small_block_size = available_processes/number_of_frequencies;
			//! Work out where the large block size finishes. This is done naively but going through each frequency and assuming it
			//! Will be a large block frequency (i.e. have the higher number of processes assigned to it), and checking when we get to the
			//! point where all frequencies above that would be full using small blocks (+- 1). The  +-1 is to ensure we match situations where the calculation is
			//! not exact. We assume that having the block of processes for the last frequency as slightly smaller or larger than the other blocks is ok in terms of
			//! load balance.
			for(int i = 0; i < number_of_frequencies; i++){
				int potential_size = (i+1)*large_block_size + ((number_of_frequencies - (i+1))*small_block_size);
				if(potential_size == available_processes or potential_size == available_processes - 1 or potential_size == available_processes + 1 or potential_size > available_processes + 1){
					large_block_limit = i;
					break;
				}
			}

		}

		available_processes = 0;
		int current_lower_process = lower_process;
		//! extra_processes_to_add is used to add in the padding processes to the first frequency to ensuring rank 0 has the space for
		//! more memory. It is set to padding processes at the beginning of the frequency loop, but then reset to zero for the rest of the frequencies
		//! as they simply depend on the value of upper_process for the preceding frequency, which will already include the padding.
		for(int k=0; k<number_of_frequencies; ++k){
			decomp_.frequencies[k].freq_number = k;
			decomp_.frequencies[k].lower_process = current_lower_process;
			//! If we are at the point where we have gone beyond the large blocks then the assigned processes
			//! are of the small block size
			if(k > large_block_limit){
				decomp_.frequencies[k].upper_process = current_lower_process + small_block_size;
				decomp_.frequencies[k].upper_process--;
			}else{
				decomp_.frequencies[k].upper_process = current_lower_process + large_block_size;
				decomp_.frequencies[k].upper_process--;
			}
			assert(decomp_.frequencies[k].upper_process >= decomp_.frequencies[k].lower_process);
			current_lower_process = decomp_.frequencies[k].upper_process + 1;
			//! Make sure that, if we're on the last frequency, the upper process is the largest process rank we have
			//! This deals with cases where the small and large block decomposition does not exactly match the
			//! number of frequencies we have, so here we are rounding up or down on the last frequency decomposition
			//! to make sure we're using the actual number of processes we have.
			if(k == number_of_frequencies-1){
				decomp_.frequencies[k].upper_process = upper_process;
			}
			//! We choose the global owner as the lowest rank process (in the global communicator) that is
			//! involved with this frequency.
			decomp_.frequencies[k].global_owner = decomp_.frequencies[k].lower_process;
		}

	}

	for(int k=0; k<number_of_frequencies; k++){
		//! The +1 in this is because the total number of processes is inclusive of the upper process, not exclusive. This means,
		//! for example, if we have lower process as 5 and upper process as 8, then we have 4 processes in total (5,6,7,8) but
		//! 8-5 = 3; Hence the +1
		decomp_.frequencies[k].process_ids = std::vector<t_uint>((decomp_.frequencies[k].upper_process-decomp_.frequencies[k].lower_process)+1);
		int j = 0;
		for(int l=decomp_.frequencies[k].lower_process;l<=decomp_.frequencies[k].upper_process;l++){
			decomp_.frequencies[k].process_ids[j] = l;
			j++;
		}
	}

	return(available_processes);
}

//! Calculate what processes own the time blocks associated with a given frequency.
int Decomposition::decompose_time_blocks(t_uint freq_number, t_uint number_of_blocks, bool time_decomp){

	decomp_.frequencies[freq_number].number_of_time_blocks = number_of_blocks;

	t_uint number_of_processes;

	if(time_decomp){
		//! 1 is added here to account for the inclusive nature of the lower and upper process indices (i.e. the upper process is included
		//! in the set of processes used
		number_of_processes = (decomp_.frequencies[freq_number].upper_process - decomp_.frequencies[freq_number].lower_process) + 1;
	}else{
		number_of_processes = 1;
	}

	//! We only have a single frequency here
	//! Time block decomp data, i.e. time blocks this process is involved with from this frequency
	decomp_.frequencies[freq_number].time_blocks = std::vector<TimeDecomp>(number_of_blocks);

	t_uint available_processes = number_of_processes;
	t_uint large_block_size = 0;
	t_uint small_block_size = 0;
	t_uint large_block_limit = 0;
	//! If we have less processes available than blocks then processes will need to own more than one block.
	if(available_processes <= number_of_blocks){
		small_block_size = number_of_blocks/available_processes;
		large_block_size = small_block_size;
		large_block_limit = -1;
		//! We do this to make sure that the decomposition can handle when the number of blocks does not divide evenly between processes
		//! The approach below rounds up so that
		if(number_of_blocks % available_processes != 0){
			large_block_size = large_block_size + 1;
			for(int i = 0; i < available_processes; i++){
				int potential_size = (i+1)*large_block_size + ((available_processes - (i+1))*small_block_size);
				if(potential_size == number_of_blocks or potential_size == number_of_blocks - 1 or potential_size == number_of_blocks + 1){
					large_block_limit = i;
					break;
				}
			}
		}else{
			//! Every block is a large (or small) block as they are both the same size
			large_block_limit = available_processes;
		}
		//! If our calculation loop above went wrong then default to large blocks all the way. This is
		//! likely to leave idle processes.
		if(large_block_limit == -1){
			large_block_limit = available_processes;
			PSI_HIGH_LOG("Problem calculating time block decomposition. It is likely to be sub-optimal and you would be advised to try with a different number of processes");
		}
		int current_block = 0;
		t_uint loop_limit = 0;
		for(int i=0; i<available_processes; ++i){
			//! If this is the last process then give it all the remaining blocks so we
			//! do not leave blocks unassigned (this takes account of the +-1 in the potential_size loop above)
			if(i == available_processes - 1){
				loop_limit = number_of_blocks;
			}else{
				//! Set the maximum number of blocks for this process (this i)
				if(i <= large_block_limit){
					loop_limit = loop_limit + large_block_size;
				}else{
					loop_limit = loop_limit + small_block_size;
				}
			}
			//! Assign the blocks to the current process (this i)
			for(int j = current_block; j < std::min(loop_limit,number_of_blocks); j++){
				decomp_.frequencies[freq_number].time_blocks[j].time_block_number = current_block;
				decomp_.frequencies[freq_number].time_blocks[j].global_owner = i + decomp_.frequencies[freq_number].lower_process;
				current_block = current_block + 1;
			}
		}

		available_processes = 0;
		if(current_block != number_of_blocks){
			decomp_.global_comm.abort("An error has occurred in the time block decomposition functionality. We have not given out all the blocks but have run out of processes.");
		}

	}else{
		large_block_size = 1;
		small_block_size = 1;
		available_processes = available_processes - number_of_blocks;
		for(int k=0; k<number_of_blocks; ++k){
			decomp_.frequencies[freq_number].time_blocks[k].time_block_number = k;
			decomp_.frequencies[freq_number].time_blocks[k].global_owner = k +  decomp_.frequencies[freq_number].lower_process;
		}
	}


	return(available_processes);
}



//! Calculate what processes own the wavelets associated with a given frequency.
int Decomposition::decompose_wavelets(t_uint freq_number, t_uint number_of_wavelets, bool wavelet_decomp){

	decomp_.frequencies[freq_number].number_of_wavelets = number_of_wavelets;

	t_uint number_of_processes;
	if(wavelet_decomp){
		//! 1 is added here to account for the inclusive nature of the calculation (i.e. upper process is contained with in the processes assigned to this frequency).
		number_of_processes = (decomp_.frequencies[freq_number].upper_process - decomp_.frequencies[freq_number].lower_process) + 1;
	}else{
		number_of_processes = 1;
	}
	//! We only have a  frequency here
	//! Wavelet decomp data, i.e. wavelets this process is involved with from this frequency
	decomp_.frequencies[freq_number].wavelets = std::vector<WaveletDecomp>(number_of_wavelets);

	t_uint available_processes = number_of_processes;
	t_uint large_block_size = 0;
	t_uint small_block_size = 0;
	t_uint large_block_limit = 0;
	//! If we have less processes available than wavelets then processes will need to own more than one wavelets.
	if(available_processes < number_of_wavelets){
		small_block_size = number_of_wavelets/available_processes;
		large_block_size = small_block_size;
		large_block_limit = -1;
		//! We do this to make sure that the decomposition can handle when the number of wavelets does not divide evenly between processes
		//! The approach below rounds up so that
		if(number_of_wavelets % available_processes != 0){
			large_block_size = large_block_size + 1;
			for(int i = 0; i < available_processes; i++){
				int potential_size = (i+1)*large_block_size + ((available_processes - (i+1))*small_block_size);
				if(potential_size == number_of_wavelets or potential_size == number_of_wavelets - 1 or potential_size == number_of_wavelets + 1){
					large_block_limit = i;
					break;
				}
			}
		}else{
			//! Every block is a large (or small) block as they are both the same size
			large_block_limit = available_processes;
		}
		//! If our calculation loop above went wrong then default to large blocks all the way. This is
		//! likely to leave idle processes.
		if(large_block_limit == -1){
			large_block_limit = available_processes;
			PSI_HIGH_LOG("Problem calculating wavelet decomposition. It is likely to be sub-optimal and you would be advised to try with a different number of processes");
		}
		int current_wavelet = 0;
		t_uint loop_limit = 0;
		for(int i=0; i<available_processes; ++i){
			//! If this is the last process then give it all the remaining blocks so we
			//! do not leave blocks unassigned (this takes account of the +-1 in the potential_size loop above)
			if(i == available_processes - 1){
				loop_limit = number_of_wavelets;
			}else{
				//! Set the maximum number of blocks for this process (this i)
				if(i <= large_block_limit){
					loop_limit = loop_limit + large_block_size;
				}else{
					loop_limit = loop_limit + small_block_size;
				}
			}
			//! Assign the blocks to the current process (this i)
			for(int j = current_wavelet; j < std::min(loop_limit,number_of_wavelets); j++){
				decomp_.frequencies[freq_number].wavelets[j].wavelet_number = current_wavelet;
				decomp_.frequencies[freq_number].wavelets[j].global_owner = i + decomp_.frequencies[freq_number].lower_process;
				current_wavelet = current_wavelet + 1;
			}
		}

		available_processes = 0;
		if(current_wavelet != number_of_wavelets){
			decomp_.global_comm.abort("An error has occurred in the wavelet decomposition functionality. We have not given out all the wavelets but have run out of processes.");
		}
		decomp_.frequencies[freq_number].global_wavelet_owner = decomp_.frequencies[freq_number].wavelets[0].global_owner;

	}else{
		large_block_size = 1;
		small_block_size = 1;
		available_processes = available_processes - number_of_wavelets;
		for(int k=0; k<number_of_wavelets; ++k){
			decomp_.frequencies[freq_number].wavelets[k].wavelet_number = k;
			decomp_.frequencies[freq_number].wavelets[k].global_owner = k + decomp_.frequencies[freq_number].lower_process;
		}
		decomp_.frequencies[freq_number].global_wavelet_owner = decomp_.frequencies[freq_number].wavelets[0].global_owner;
	}

	return(available_processes);
}


//! Calculate what processes own the wavelets associated with the root process specific work (i.e. wavelet regularisation and similar)
void Decomposition::decompose_root_wavelets(int number_of_wavelets, bool root_wavelet_decomp){

	t_uint number_of_processes;

	decomp_.number_of_root_wavelets = number_of_wavelets;


	if(!root_wavelet_decomp){
		number_of_processes = 1;
	}else{
		//! the decomposition zero frequency rank here is used to deal with process padding
		number_of_processes = std::min(int(decomp_.global_comm.size() - decomp_.frequencies[0].lower_process), number_of_wavelets);
	}
	//! Wavelet decomp data, i.e. wavelets for the root process work
	decomp_.root_wavelets = std::vector<WaveletDecomp>(number_of_wavelets);

	t_uint available_processes = number_of_processes;
	t_uint large_block_size = 0;
	t_uint small_block_size = 0;
	t_uint large_block_limit = 0;
	//! If we have less processes available than wavelets then processes will need to own more than one wavelets.
	if(available_processes < number_of_wavelets){
		small_block_size = number_of_wavelets/available_processes;
		large_block_size = small_block_size;
		large_block_limit = -1;
		//! We do this to make sure that the decomposition can handle when the number of wavelets does not divide evenly between processes
		//! The approach below rounds up so that
		if(number_of_wavelets % available_processes != 0){
			large_block_size = large_block_size + 1;
			for(int i = 0; i < available_processes; i++){
				int potential_size = (i+1)*large_block_size + ((available_processes - (i+1))*small_block_size);
				if(potential_size == number_of_wavelets or potential_size == number_of_wavelets - 1 or potential_size == number_of_wavelets + 1){
					large_block_limit = i;
					break;
				}
			}
		}else{
			//! Every block is a large (or small) block as they are both the same size
			large_block_limit = available_processes;
		}
		//! If our calculation loop above went wrong then default to large blocks all the way. This is
		//! likely to leave idle processes.
		if(large_block_limit == -1){
			large_block_limit = available_processes;
			PSI_HIGH_LOG("Problem calculating root wavelet decomposition. It is likely to be sub-optimal and you would be advised to try with a different number of processes");
		}
		int current_wavelet = 0;
		int loop_limit = 0;
		for(int i=0; i<available_processes; ++i){
			//! If this is the last process then give it all the remaining blocks so we
			//! do not leave blocks unassigned (this takes account of the +-1 in the potential_size loop above)
			if(i == available_processes - 1){
				loop_limit = number_of_wavelets;
			}else{
				//! Set the maximum number of blocks for this process (this i)
				if(i <= large_block_limit){
					loop_limit = loop_limit + large_block_size;
				}else{
					loop_limit = loop_limit + small_block_size;
				}
			}
			//! Assign the blocks to the current process (this i)
			for(int j = current_wavelet; j < std::min(loop_limit,number_of_wavelets); j++){
				decomp_.root_wavelets[j].wavelet_number = j;
				decomp_.root_wavelets[j].global_owner = i + decomp_.frequencies[0].lower_process;
				current_wavelet = current_wavelet + 1;
			}
		}

		available_processes = 0;
		if(current_wavelet != number_of_wavelets){
			decomp_.global_comm.abort("An error has occurred in the root wavelet decomposition functionality. We have not given out all the wavelets but have run out of processes.");
		}

	}else{
		large_block_size = 1;
		small_block_size = 1;
		available_processes = available_processes - number_of_wavelets;
		for(int k=0; k<number_of_wavelets; ++k){
			decomp_.root_wavelets[k].wavelet_number = k;
			decomp_.root_wavelets[k].global_owner = k + decomp_.frequencies[0].lower_process;
		}

	}

	return;
}


//! Iterate through the global decomposition that has been constructed and build the decomposition of the owning process.
//! This contains all the frequencies, time blocks, and sub blocks this process is involved with and the associated communicators.
//! We use the same Decomposition struct to hold this data, but use the variable my_decomp_ rather than decomp_. This means
//! we can still hold the full decomposition (which will not have initialised communicators) for reference as required
//! (i.e. for root processes for any of the individual parts).
void Decomposition::build_my_decomposition(){

	t_uint my_rank = decomp_.global_comm.rank();
	my_decomp_.frequencies = std::vector<FreqDecomp>();
	t_uint number_of_my_frequencies = 0;
	int number_of_my_wavelets = 0;
	int lower_wavelet_number = 0;
	int my_current_frequency = 0;

	int number_of_root_wavelets;
	int lower_root_wavelet_number;

	std::tie(number_of_root_wavelets, lower_root_wavelet_number) = number_and_start_of_local_wavelets(decomp_.number_of_root_wavelets, decomp_.root_wavelets);

	int in_root_wavelets = 0;

	my_decomp_.number_of_root_wavelets = number_of_root_wavelets;
	my_decomp_.lower_root_wavelet = lower_root_wavelet_number;
	if(my_decomp_.number_of_root_wavelets > 0){
		my_decomp_.root_wavelets = std::vector<WaveletDecomp>();
		for(int j=0; j<decomp_.frequencies[0].number_of_wavelets; j++){
			my_decomp_.root_wavelets.push_back(decomp_.frequencies[0].wavelets[j]);
		}
		in_root_wavelets = 1;
	}

	if(in_root_wavelets == 1){
		my_decomp_.root_wavelet_comm = Communicator(decomp_.global_comm.split(in_root_wavelets));
	}else{
		//Split for those not in the root wavelets. This should automatically deallocate the communicator when the routine finishes as it will go out of scope.
		Communicator root_wavelet_comm = Communicator(decomp_.global_comm.split(in_root_wavelets));
	}

	for(int i=0; i<decomp_.number_of_frequencies; i++){
		int in_this_frequency = 0;
		int in_this_wavelet = 0;
		std::tie(number_of_my_wavelets, lower_wavelet_number) = number_and_start_of_local_wavelets(decomp_.frequencies[i].number_of_wavelets, decomp_.frequencies[i].wavelets);
		if(number_of_my_wavelets != 0){
			in_this_wavelet = 1;
		}
		if(in_this_wavelet or (std::find(decomp_.frequencies[i].process_ids.begin(), decomp_.frequencies[i].process_ids.end(), my_rank) != decomp_.frequencies[i].process_ids.end())){
			number_of_my_frequencies++;
			my_decomp_.frequencies.push_back(decomp_.frequencies[i]);
			my_decomp_.frequencies[number_of_my_frequencies-1].time_blocks = std::vector<TimeDecomp>();
			t_uint number_of_my_time_blocks = 0;
			for(int j=0; j<decomp_.frequencies[i].number_of_time_blocks; j++){
				if(decomp_.frequencies[i].time_blocks[j].global_owner == my_rank){
					my_decomp_.frequencies[number_of_my_frequencies-1].time_blocks.push_back(decomp_.frequencies[i].time_blocks[j]);
					number_of_my_time_blocks++;
				}
			}
			my_decomp_.frequencies[number_of_my_frequencies-1].wavelets = std::vector<WaveletDecomp>();
			for(int j=0; j<decomp_.frequencies[i].number_of_wavelets; j++){
				my_decomp_.frequencies[number_of_my_frequencies-1].wavelets.push_back(decomp_.frequencies[i].wavelets[j]);
			}
			my_decomp_.frequencies[number_of_my_frequencies-1].number_of_wavelets = number_of_my_wavelets;
			my_decomp_.frequencies[number_of_my_frequencies-1].lower_wavelet = lower_wavelet_number;
			my_decomp_.frequencies[number_of_my_frequencies-1].number_of_time_blocks = number_of_my_time_blocks;
			in_this_frequency = 1;
		}
		if(decomp_.parallel_mpi){
			if(in_this_frequency == 1){
				decomp_.frequencies[i].in_this_frequency = true;
				my_decomp_.frequencies[number_of_my_frequencies-1].in_this_frequency = true;
				my_decomp_.frequencies[number_of_my_frequencies-1].freq_comm = Communicator(decomp_.global_comm.split(in_this_frequency));
				my_decomp_.frequencies[number_of_my_frequencies-1].local_owner = my_decomp_.frequencies[number_of_my_frequencies-1].freq_comm.root_id();
			}else{
				//Split for those not in this frequency. This should automatically deallocate the communicator when the routine finishes as it will go out of scope.
				Communicator freq_comm = Communicator(decomp_.global_comm.split(in_this_frequency));
				decomp_.frequencies[i].in_this_frequency = false;
			}
			if(in_this_wavelet == 1){
				my_decomp_.frequencies[number_of_my_frequencies-1].wavelet_comm = Communicator(decomp_.global_comm.split(in_this_wavelet));
				if(my_decomp_.frequencies[number_of_my_frequencies-1].wavelet_comm.is_root()){
					if(my_rank != my_decomp_.frequencies[number_of_my_frequencies-1].global_wavelet_owner){
						decomp_.global_comm.abort("An error has occurred in the wavelet decomposition functionality. Wavelet global owner is not the same as the wavelet_comm root.");
					}
				}
			}else{
				//Split for those not in the wavelet. This should automatically deallocate the communicator when the routine finishes as it will go out of scope.
				Communicator wavelet_comm = Communicator(decomp_.global_comm.split(in_this_wavelet));
			}
		}

	}

	my_decomp_.number_of_frequencies = number_of_my_frequencies;

	//! The following code creates communicators to go across frequency wavelets.
	//! The idea is that each wavelet owner should be in the communicator for all frequencies, i.e. there will be single communicator
	//! per wavelet containing all the processes that own that wavelet across all frequencies. This is required for the l21 calculation.
	//! This code assumes all frequencies have the same number of wavelets. It should be a safe assumption, but we put in an assert just in case.
	std::vector<t_uint> wavelets;
	if(my_decomp_.number_of_frequencies != 0){
		wavelets = std::vector<t_uint>(decomp_.frequencies[0].number_of_wavelets, 0);
		for(int i=0; i<my_decomp_.number_of_frequencies; i++){
			// Check the wavelets array is the correct size;
			assert(decomp_.frequencies[i].number_of_wavelets == decomp_.frequencies[0].number_of_wavelets);
			for(int j=0; j<my_decomp_.frequencies[i].number_of_wavelets; j++){
				if(my_decomp_.frequencies[i].wavelets[j].global_owner == my_rank){
					wavelets[j] = 1;
				}
			}
		}


		//! This code creates the communicators based on the information calculated above
		my_decomp_.wavelet_comms = std::vector<Communicator>(decomp_.frequencies[0].number_of_wavelets);
		my_decomp_.wavelet_comms_involvement = std::vector<bool>(decomp_.frequencies[0].number_of_wavelets);
	}

	for(int i=0; i<decomp_.frequencies[0].number_of_wavelets; i++){
		if(my_decomp_.number_of_frequencies != 0 and wavelets[i] == 1){
			my_decomp_.wavelet_comms[i] = Communicator(decomp_.global_comm.split(wavelets[i]));
			my_decomp_.wavelet_comms_involvement[i] = true;
		}else{
			//Split for those not in the wavelet on any frequency. This should automatically deallocate the communicator when the routine finishes as it will go out of scope.
			Communicator temp_comm = Communicator(decomp_.global_comm.split(0));
			if(my_decomp_.number_of_frequencies != 0){
				my_decomp_.wavelet_comms_involvement[i] = false;
			}
		}
	}


	my_decomp_.checkpointing = decomp_.checkpointing;
	my_decomp_.checkpointing_frequency = decomp_.checkpointing_frequency;

}

//! Print out summary statistics on the decomposition.
void Decomposition::report_on_decomposition(bool quiet){


	if(decomp_.parallel_mpi){
		if(decomp_.global_comm.is_root()){

			PSI_HIGH_LOG("Using {} MPI processes",decomp_.global_comm.size());
#ifdef PSI_OPENMP
			int num_threads = 0;
#pragma omp parallel shared(num_threads)
			{
				num_threads = omp_get_num_threads();
			}
			PSI_HIGH_LOG("Each MPI process has {} OpenMP threads", num_threads);
#endif

			if(not quiet){

				if(decomp_.freq_decomp){

					int min_size = 0;
					int max_size = 0;
					for(int fr=0; fr<decomp_.number_of_frequencies; fr++){
						int freq_size = decomp_.frequencies[fr].process_ids.size();
						if(freq_size < min_size or min_size == 0){
							min_size = freq_size;
						}
						if(freq_size > max_size){
							max_size = freq_size;
						}
					}

					//! Calculate the relative difference between the biggest and smallest blocks assigned to processes or blocks of processes. This
					//! gives us a crude measure of load imbalance as it ignore how many blocks there are

					float unbalance;
					if(max_size == 0){
						unbalance = 1.0;
					}else{
						unbalance = min_size/(float)max_size;
					}
					unbalance = 100*(1 - unbalance);
					PSI_HIGH_LOG("Number of frequencies: {}",decomp_.number_of_frequencies);
					PSI_HIGH_LOG("Largest block size: {} Smallest block size: {}",max_size,min_size);
					PSI_HIGH_LOG("Percentage difference largest and smallest number of blocks {}%",unbalance);
				}else{
					PSI_HIGH_LOG("Number of frequencies: {}",decomp_.number_of_frequencies);
					PSI_HIGH_LOG("Frequencies have not been parallelised, but time blocks and wavelets may have");
				}
				for(int fr=0; fr<decomp_.number_of_frequencies; fr++){
					PSI_HIGH_LOG("Assignment of processes for frequency[{}]:",decomp_.frequencies[fr].freq_number);
					for(int p=0; p<decomp_.frequencies[fr].process_ids.size();p++){
						PSI_HIGH_LOG("{}",decomp_.frequencies[fr].process_ids[p]);
					}
				}


				if(decomp_.root_wavelet_decomp){
					int num_wavelets = decomp_.number_of_root_wavelets;
					PSI_HIGH_LOG("Number of root wavelets: {}",num_wavelets);
					PSI_HIGH_LOG("Assignment of processes for root wavelets:");
					for(int wave=0; wave<num_wavelets; wave++){
						PSI_HIGH_LOG("{} ", decomp_.root_wavelets[wave].global_owner);
					}
				}else{
					int num_wavelets = decomp_.number_of_root_wavelets;
					PSI_HIGH_LOG("Number of root wavelets: {}",num_wavelets);
					PSI_HIGH_LOG("All root wavelets are being calculated by a single process");
				}

				for(int fr=0; fr<decomp_.number_of_frequencies; fr++){
					if(decomp_.time_decomp){
						t_int time_block_min_size = 0;
						t_int time_block_max_size = 0;
						t_int time_block_number_of_splits = 1;
						t_int time_block_current_total = 0;
						t_int time_block_current_process;
						time_block_current_process = (int)decomp_.frequencies[fr].time_blocks[0].global_owner;
						for(int k=0; k<decomp_.frequencies[fr].time_blocks.size(); k++){
							if((int)decomp_.frequencies[fr].time_blocks[k].global_owner == time_block_current_process){
								time_block_current_total++;
							}else{
								if(time_block_current_total < time_block_min_size or time_block_min_size == 0){
									time_block_min_size = time_block_current_total;
								}
								if(time_block_current_total > time_block_max_size){
									time_block_max_size = time_block_current_total;
								}
								time_block_current_process = (int)decomp_.frequencies[fr].time_blocks[k].global_owner;
								time_block_number_of_splits++;
								time_block_current_total = 1;
							}
						}
						//! Check if the last block is the smallest or biggest we've encountered
						if(time_block_current_total < time_block_min_size or time_block_min_size == 0){
							time_block_min_size = time_block_current_total;
						}
						if(time_block_current_total > time_block_max_size){
							time_block_max_size = time_block_current_total;
						}

						//! Calculate the relative difference between the biggest and smallest blocks assigned to processes. This
						//! gives us a crude measure of load imbalance (currently ignoring the actual size of each block).

						float time_block_unbalance;
						if(time_block_max_size == 0){
							time_block_unbalance = 1.0;
						}else{
							time_block_unbalance = time_block_min_size/(float)time_block_max_size;
						}
						time_block_unbalance = 100*(1 - time_block_unbalance);
						int block_size = decomp_.frequencies[fr].time_blocks.size();
						PSI_HIGH_LOG("Number of time blocks for frequency[{}]: {}",fr,block_size);
						PSI_HIGH_LOG("Largest block size: {} Smallest block size: {} Number of processes used: {}",time_block_max_size,time_block_min_size,time_block_number_of_splits);
						PSI_HIGH_LOG("Percentage difference largest and smallest number of blocks assigned to a process {}%",time_block_unbalance);
						if(time_block_current_process != (t_int)decomp_.frequencies[fr].upper_process){
							int idle = decomp_.frequencies[fr].upper_process - time_block_current_process;
							PSI_HIGH_LOG("{} processes idle for this set of time blocks", idle);
						}

					}else{
						int block_size = decomp_.frequencies[fr].time_blocks.size();
						PSI_HIGH_LOG("Number of time blocks for frequency[{}]: {}",fr,block_size);
						PSI_HIGH_LOG("All time blocks for this frequency are being calculated by a single process");
					}
					if(decomp_.wavelet_decomp){
						t_int wavelet_min_size = 0;
						t_int wavelet_max_size = 0;
						t_int wavelet_number_of_splits = 1;
						t_int wavelet_current_total = 0;
						t_int wavelet_current_process;
						wavelet_current_process = decomp_.frequencies[fr].wavelets[0].global_owner;
						for(int k=0; k<decomp_.frequencies[fr].number_of_wavelets; k++){
							if(decomp_.frequencies[fr].wavelets[k].global_owner == wavelet_current_process){
								wavelet_current_total++;
							}else{
								if(wavelet_current_total < wavelet_min_size or wavelet_min_size == 0){
									wavelet_min_size = wavelet_current_total;
								}
								if(wavelet_current_total > wavelet_max_size){
									wavelet_max_size = wavelet_current_total;
								}
								wavelet_current_process = decomp_.frequencies[fr].wavelets[k].global_owner;
								wavelet_number_of_splits++;
								wavelet_current_total = 1;
							}
						}
						//! Check if the last block is the smallest or biggest we've encountered
						if(wavelet_current_total < wavelet_min_size or wavelet_min_size == 0){
							wavelet_min_size = wavelet_current_total;
						}
						if(wavelet_current_total > wavelet_max_size){
							wavelet_max_size = wavelet_current_total;
						}

						//! Calculate the relative difference between the biggest and smallest blocks assigned to processes. This
						//! gives us a crude measure of load imbalance.
						float wavelet_unbalance;
						if(wavelet_max_size == 0){
							wavelet_unbalance = 1.0;
						}else{
							wavelet_unbalance = wavelet_min_size/(float)wavelet_max_size;
						}
						wavelet_unbalance = 100*(1 - wavelet_unbalance);
						int num_wavelets = decomp_.frequencies[fr].number_of_wavelets;
						PSI_HIGH_LOG("Number of wavelets for frequency[{}]: {}",fr,num_wavelets);
						PSI_HIGH_LOG("Largest block of wavelets: {} Smallest of wavelets: {} Number of processes used: {}",wavelet_max_size,wavelet_min_size,wavelet_number_of_splits);
						PSI_HIGH_LOG("Percentage difference largest and smallest number of wavelets assigned to a process {}%",wavelet_unbalance);
					}else{
						int num_wavelets = decomp_.frequencies[fr].number_of_wavelets;
						PSI_HIGH_LOG("Number of wavelets for frequency[{}]: {}",fr,num_wavelets);
						PSI_HIGH_LOG("All wavelets for this frequency are being calculated by a single process");
					}
				}
			}
		}
	}

}

//! Should checkpointing be carried out for this iteration (niters)
bool Decomposition::checkpoint_now(t_uint niters, t_uint maxiters) const{

	bool frequency = false;

	if(not decomp_.checkpointing){
		return(false);
	}

	if(decomp_.checkpointing_frequency != 0){
		//! Checkpoint if this is the first iteration, if we have had decomp_.checkpointing_frequency iterations since the last checkpoint, or
		//! we have reached the end of the iterations for the algorithm run.
		//! The +1 on the checkpointing_frequency here is to deal with the fact that iterations (niters) counts from 0 rather than 1.
		frequency = ((niters == 0) or (niters != 0  and (((niters+1)%(decomp_.checkpointing_frequency) == 0) or ((niters + 1) == maxiters))));
	}else{
		//! Checkpoint at the end of the algorithm is checkpointing_frequency set to 0.
		frequency = ((niters + 1) == maxiters);
	}
	return(frequency);
}

bool Decomposition::restore_checkpoint() const{

	return(decomp_.restoring and not decomp_.checkpoint_restored);

}

// Set variable so the program knows a restore has been done (checkpoint_restored) to ensure
// it isn't re-restored on the next reweighting iteration. Also set a variable so the program
// knows that the restore has happened on this reweighting iteration (this_reweighting_iteration)
// so that it doesn't overwrite epsilon and l1 weight values that have been restored this iteration but
// does for future iterations.
void Decomposition::restore_complete() {
	decomp_.checkpoint_restored = true;
}

void Decomposition::initialise_requests(int freq, int count){
	if(decomp_.frequencies[freq].requests_initialised){
		PSI_ERROR("Attempting to initialise the decomposition requests array when it already has been initialised. This is a mistake.");
	}else{
		// Initialise the requests to MPI_REQUEST_NULL so the wait all works for blocks that are owned by root.
		decomp_.frequencies[freq].requests = (MPI_Request *) malloc(count*sizeof(MPI_Request));
		for(int k = 0; k < count; k++){
			decomp_.frequencies[freq].requests[k] = MPI_REQUEST_NULL;
		}
		decomp_.frequencies[freq].requests_initialised = true;
	}
	return;
}

void Decomposition::cleanup_requests(int freq){

	if(not decomp_.frequencies[freq].requests_initialised){
		PSI_ERROR("Attempting to cleanup the decomposition requests array when it has not been initialised. This is a mistake.");
	}else{
		//TODO Check no requests are still outstanding. If this is called and requests are still outstanding the
		// application could deadlock.
		free(decomp_.frequencies[freq].requests);
		// Record that the requests array is not active
		decomp_.frequencies[freq].requests_initialised = false;
	}
	return;
}

void Decomposition::wait_on_requests(int freq, int count){

	if(not decomp_.frequencies[freq].requests_initialised){
		PSI_ERROR("Attempting to wait the decomposition requests array when it has not been initialised. This is a mistake.");
	}else if(not decomp_.global_comm.is_root() and not decomp_.frequencies[freq].freq_comm.is_root()){
		PSI_ERROR("Wait called by wrong process, should only be called by the frequency root");
	}else{
		decomp_.global_comm.wait_on_all(decomp_.frequencies[freq].requests, count);
	}
	return;
}


//! How many wavelets does this process have in the decomposition for a given frequency.
std::pair<int, int> Decomposition::number_and_start_of_local_wavelets(int number_of_wavelets, std::vector<WaveletDecomp> wavelets){
	int my_number_of_wavelets = 0;
	int start_wavelet = -1;
	for(int i = 0; i<number_of_wavelets; i++){
		if(wavelets[i].global_owner == decomp_.global_comm.rank()){
			my_number_of_wavelets++;
			if(i < start_wavelet or start_wavelet == -1){
				start_wavelet = i;
			}
		}
	}
	return(std::make_pair(my_number_of_wavelets, start_wavelet));
}

bool Decomposition::own_this_frequency(int frequency){
	if(decomp_.frequencies[frequency].global_owner == decomp_.global_comm.rank()){
		return(true);
	}else{
		return(false);
	}
}


} /* psi::decomposition */
}/* psi  */
