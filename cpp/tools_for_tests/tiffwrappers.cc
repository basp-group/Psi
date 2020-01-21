#include <fstream>

#include "psi/types.h"
#include "psi/utilities.h"
#include "tools_for_tests/directories.h"
#include "tools_for_tests/tiffwrappers.h"

namespace psi {
namespace notinstalled {
Image<> read_standard_tiff(std::string const &name) {
  std::string const stdname = notinstalled::data_directory() + "/" + name + ".tiff";
  bool const is_std = std::ifstream(stdname).good();
  return utilities::read_tiff(is_std ? stdname : name);
}
}
} /* psi::notinstalled  */
