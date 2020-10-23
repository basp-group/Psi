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
    set(PSI_EIGEN_BLAS 1) # This will go into config.h
else()
    set(PSI_EIGEN_MKL 0)
    set(EIGEN_USE_MKL_ALL 0)
    set(PSI_EIGEN_BLAS FALSE)
    find_package(BLAS)
    if(BLAS_FOUND AND blas)
       set(PSI_EIGEN_BLAS 1)
       set(EIGEN_USE_BLAS 1)
    else()
       set(PSI_EIGEN_BLAS 0)
       set(EIGEN_USE_BLAS 0)
    endif()
    set(PSI_BLAS ${BLAS_FOUND})
endif()
set(PSI_EIGEN_MKL ${MKL_FOUND})

set(PSI_SCALAPACK FALSE)

# Find Scalapack
find_package(Scalapack)

if(SCALAPACK_FOUND AND scalapack)
    set(PSI_SCALAPACK 1) # This will go into config.h
    if(MKL_FOUND)
    	set(PSI_SCALAPACK_MKL 1)# This will go into config.h
    endif()
elseif(NOT SCALAPACK_FOUND AND scalapack)
    message(FATAL_ERROR "Scalapack selected but not found")
else()
    set(PSI_SCALAPACK 0)
    set(PSI_SCALAPACK_MKL 0)
endif()
