#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include <tools_for_tests/directories.h>
#include <tools_for_tests/tiffwrappers.h>
#include <psi/logging.h>
#include <psi/utilities.h>

// \min_{x} ||\Psi^\dagger x||_1 \quad \mbox{s.t.} \quad ||y - x||_2 < \epsilon and x \geq 0
int main(int argc, char const **argv) {
  if(argc != 3) {
    std::cout << "Expects two arguments:\n"
                 "- path to the image to clean (or name of standard PSI image)\n"
                 "- path to output image\n";
    exit(0);
  }

  // Initializes and sets logger (if compiled with logging)
  // See set_level function for levels.
  psi::logging::initialize();

  // Read input file
  auto const image = psi::notinstalled::read_standard_tiff(argv[1]);
  psi::utilities::write_tiff(image, argv[2]);

  auto const reread = psi::utilities::read_tiff(argv[2]);

  return 0;
}
