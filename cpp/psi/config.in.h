#ifndef PSI_CPP_CONFIG_H
#define PSI_CPP_CONFIG_H

//! Problems with using and constructors
#cmakedefine PSI_HAS_USING
#ifndef PSI_HAS_USING
#define PSI_HAS_NOT_USING
#endif

//! True if using OPENMP
#cmakedefine PSI_OPENMP

//! Macro to start logging or not
#cmakedefine PSI_DO_LOGGING

//! True if using MPI
#cmakedefine PSI_MPI

//! True if found MKL
#cmakedefine PSI_MKL

// Whether Eigen will use MKL (if MKL was found and PSI_EIGEN_MKL is enabled in CMake)
#cmakedefine PSI_EIGEN_MKL
#cmakedefine EIGEN_USE_MKL_ALL

//! True if found BLAS
#cmakedefine PSI_BLAS

// Whether Eigen will use BLAS (if BLAS was found and PSI_EIGEN_BLAS is enabled in CMake)
#cmakedefine PSI_EIGEN_BLAS
#cmakedefine EIGEN_USE_BLAS 

// Whether we will use the SVD from Scalapack instead of Eigen
#cmakedefine PSI_SCALAPACK
// Whether the scalapack is from the MKL library
#cmakedefine PSI_SCALAPACK_MKL

// figures out available basic types
#cmakedefine PSI_CHAR_ARCH
#cmakedefine PSI_LONG_ARCH
#cmakedefine PSI_ULONG_ARCH


#include <string>
#include <tuple>

namespace psi {
//! Returns library version
inline std::string version() { return "@PSI_VERSION@"; }
//! Returns library version
inline std::tuple<uint8_t, uint8_t, uint8_t> version_tuple() {
  return std::tuple<uint8_t, uint8_t, uint8_t>(
      @PSI_VERSION_MAJOR@, @PSI_VERSION_MINOR@, @PSI_VERSION_PATCH@);
}
//! Returns library git reference, if known
inline std::string gitref() { return "@PSI_GITREF@"; }
//! Default logging level
inline std::string default_logging_level() { return "@PSI_TEST_LOG_LEVEL@"; }
//! Default logger name
inline std::string default_logger_name() { return "@PSI_LOGGER_NAME@"; }
//! Wether to add color to the logger
inline constexpr bool color_logger() { return @PSI_COLOR_LOGGING@; }
# ifdef PSI_OPENMP
//! Number of threads used during testing
inline constexpr std::size_t number_of_threads_in_tests() { return @PSI_DEFAULT_OPENMP_THREADS@; }
# endif
}

#endif
