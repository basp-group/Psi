#include(PackageLookup)  # check for existence, or install external projects

find_package(Eigen3 REQUIRED)


if(logging)
  find_package(spdlog REQUIRED)
endif()

find_package(TIFF)
if(examples)
  if(NOT TIFF_FOUND)
    message(FATAL_ERROR "Examples and regressions require TIFF")
  endif()
endif()


if(openmp)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(PSI_DEFAULT_OPENMP_THREADS 4 CACHE STRING "Number of threads used in testing")
    set(PSI_OPENMP TRUE)
    add_library(openmp::openmp INTERFACE IMPORTED GLOBAL)
    set_target_properties(openmp::openmp PROPERTIES
      INTERFACE_COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}"
      INTERFACE_LINK_LIBRARIES  "${OpenMP_CXX_FLAGS}")
  else()
    message(STATUS "Could not find OpenMP. Compiling without.")
    set(PSI_OPENMP FALSE)
  endif()
endif()

set(PSI_MPI FALSE)
if(mpi)
        find_package(MPI)
endif()
set(PSI_MPI ${MPI_FOUND})

set(PSI_EIGEN_MKL FALSE)
# Find MKL
find_package(MKL)

if(MKL_FOUND AND mkl)
    set(PSI_EIGEN_MKL 1) # This will go into config.h
    set(EIGEN_USE_MKL_ALL 1) # This will go into config.h - it makes Eigen use MKL
    include_directories(${MKL_INCLUDE_DIR})
else()
    set(PSI_EIGEN_MKL 0)
    set(EIGEN_USE_MKL_ALL 0)
endif()
