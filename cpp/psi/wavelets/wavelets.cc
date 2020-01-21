#include "psi/wavelets/wavelets.h"
#include "psi/config.h"
#include <exception>
#include "psi/logging.h"

namespace psi {
namespace wavelets {

Wavelet factory(std::string name, t_uint nlevels, t_uint start_level) {
  if(name == "dirac" or name == "Dirac") {
    PSI_MEDIUM_LOG("Creating Dirac Wavelet");
    return Wavelet(daubechies_data(1), (t_uint)0);
  }

  if(name.substr(0, 2) == "DB" or name.substr(0, 2) == "db") {
    std::istringstream sstr(name.substr(2, name.size() - 2));
    t_uint l(0);
    sstr >> l;
    PSI_MEDIUM_LOG("Creating Daubechies Wavelet {}, level {}, start level {}", l, nlevels, start_level);
    return Wavelet(daubechies_data(l+start_level), nlevels);
  }
  // Unknown input wavelet
  throw std::exception();
}
}
}
