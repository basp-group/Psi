#include "psi/config.h"
#include "sara.h"

namespace psi {
namespace wavelets {
SARA distribute_sara(SARA const &sara, t_uint const sara_start, t_uint const sara_count, t_uint const total_wavelets) {
	t_uint total_size = sara.size();
	t_uint local_start = sara_start;
	t_uint local_end = std::min(total_size,sara_start + sara_count);
	//!
	if(sara_count == 0){
		local_start = total_size;
		local_end = total_size;
	}
	return SARA(sara.begin() + local_start, sara.begin() + local_end, total_wavelets);

}
}
}
