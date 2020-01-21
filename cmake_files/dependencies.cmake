include(PackageLookup)  # check for existence, or install external projects

lookup_package(Eigen3 REQUIRED DOWNLOAD_BY_DEFAULT ARGUMENTS URL "https://bitbucket.org/eigen/eigen/get/3.3.5.tar.gz" MD5 "ee48cafede2f51fe33984ff5c9f48026")

find_package(Eigen3 REQUIRED)

if(logging)
  lookup_package(spdlog REQUIRED)
endif()

find_package(TIFF)
if(examples OR regression)
  if(NOT TIFF_FOUND)
    message(FATAL_ERROR "Examples and regressions require TIFF")
  endif()
endif()

if(regressions)
  find_package(FFTW3 REQUIRED DOUBLE)
  set(REGRESSION_ORACLE_ID "last_of_c"
    CACHE STRING "Commmit/tag/branch againts which to run regressions")

  lookup_package(Psi
    REQUIRED DOWNLOAD_BY_DEFAULT
    PATHS "${EXTERNAL_ROOT}"
    NO_DEFAULT_PATH
    ARGUMENTS
      GIT_REPOSITORY "https://www.github.com/basp-group/Psi.git"
      GIT_TAG ${REGRESSION_ORACLE_ID}
      BUILD_TYPE Release
  )
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
