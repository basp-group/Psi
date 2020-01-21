#ifndef PSI_TIFF_WRAPPER_H
#define PSI_TIFF_WRAPPER_H

#include "psi/config.h"
#include <Eigen/Core>
#include "psi/types.h"

namespace psi {
namespace notinstalled {
//! Reads tiff image from psi data directory if it exists
psi::Image<> read_standard_tiff(std::string const &name);
}
} /* psi::notinstalled */
#endif
