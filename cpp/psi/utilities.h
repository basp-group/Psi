#ifndef PSI_UTILITIES_H
#define PSI_UTILITIES_H

#include "psi/config.h"
#include <Eigen/Core>
#include "psi/types.h"

namespace psi {
namespace utilities {
//! Reads tiff image
psi::Image<> read_tiff(std::string const &name);
//! Writes a tiff greyscale file
void write_tiff(Image<> const &image, std::string const &filename);
}
} /* psi::utilities */
#endif

