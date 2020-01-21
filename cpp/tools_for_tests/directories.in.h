#ifndef PSI_DATA_DIR_H
#define PSI_DATA_DIR_H

#include "psi/config.h"
#include <string>

namespace psi {
namespace notinstalled {
//! Holds images and such
inline std::string data_directory() { return "@PROJECT_SOURCE_DIR@/images"; }
//! Output artefacts from tests
inline std::string output_directory() { return "@PROJECT_BINARY_DIR@/outputs"; }
}
} /* psi::notinstalled */
#endif
