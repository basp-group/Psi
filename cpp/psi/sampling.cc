#include "psi/sampling.h"

namespace psi {
Sampling::Sampling(t_uint size, t_uint samples) : indices_(size), size(size) {
  std::iota(indices_.begin(), indices_.end(), 0);
  std::shuffle(indices_.begin(), indices_.end(), std::mt19937(std::random_device()()));
  indices_.resize(samples);
}
} /* psi  */
