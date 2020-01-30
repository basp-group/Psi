cmake_minimum_required(VERSION 2.8)
project(PSI CXX)

# Location of extra cmake includes for the project
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_files)
# Downloads and installs GreatCMakeCookOff
# It contains a number of cmake recipes

# Version and git hash id
include(VersionAndGitRef)
set_version(1.0.0)
get_gitref()

option(tests          "Enable testing"                         on)
option(benchmarks     "Enable benchmarking"                    off)
option(examples       "Enable Examples"                        on)
option(logging        "Enable logging"                         on)
option(openmp         "Enable OpenMP"                          on)
option(mpi 	      "Enable MPI" 			       on)
option(mkl            "Use MKL for Eigen acceleration"         on)

if(tests)
  enable_testing()
endif()

include(AddCPP11Flags)
include(CheckCXX11Features)

cxx11_feature_check(REQUIRED unique_ptr nullptr override constructor_delegate)
include(compilations)

# search/install dependencies
include(dependencies)
# sets rpath policy explicitly
if(CMAKE_VERSION VERSION_LESS 3.0)
    set_property(GLOBAL PROPERTY MACOSX_RPATH ON)
else()
    cmake_policy(SET CMP0042 NEW)
endif()

if(PSI_MPI)
  include(DetectIntegerArchitecture)
  DetectIntegerArchitecture(PSI)
endif()


if(tests OR examples)
  enable_testing()
endif()

if(tests)
  include(AddCatchTest)
endif()
if(examples)
  include(AddExample)
endif()

if(benchmarks)
  include(AddBenchmark)
endif()


add_subdirectory(cpp)

# Exports all Psi so other packages can access it
include(export_psi)
