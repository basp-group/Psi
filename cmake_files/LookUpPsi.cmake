# Looks up [Psi](http://basp-group.github.io/Psi/)
#
# - GIT_REPOSITORY: defaults to https://github.com/basp-group/Psi.git
# - GIT_TAG: defaults to master
# - BUILD_TYPE: defaults to Release
#
if(PSI_ARGUMENTS)
    cmake_parse_arguments(Psi "" "GIT_REPOSITORY;GIT_TAG;BUILD_TYPE" ""
        ${PSI_ARGUMENTS})
endif()
if(NOT PSI_GIT_REPOSITORY)
    set(PSI_GIT_REPOSITORY https://github.com/basp-group/Psi.git)
endif()
if(NOT PSI_GIT_TAG)
    set(PSI_GIT_TAG master)
endif()
if(NOT PSI_BUILD_TYPE)
    set(PSI_BUILD_TYPE Release)
endif()

# write subset of variables to cache for psi to use
include(PassonVariables)
passon_variables(Lookup-Psi
  FILENAME "${EXTERNAL_ROOT}/src/PsiVariables.cmake"
  PATTERNS
      "CMAKE_[^_]*_R?PATH" "CMAKE_C_.*"
      "BLAS_.*" "FFTW3_.*" "TIFF_.*"
  ALSOADD
      "\nset(CMAKE_INSTALL_PREFIX \"${EXTERNAL_ROOT}\" CACHE STRING \"\")\n"
)
ExternalProject_Add(
    Lookup-Psi
    PREFIX ${EXTERNAL_ROOT}
    GIT_REPOSITORY ${PSI_GIT_REPOSITORY}
    GIT_TAG ${PSI_GIT_TAG}
    CMAKE_ARGS
      -C "${EXTERNAL_ROOT}/src/PsiVariables.cmake"
      -DBUILD_SHARED_LIBS=OFF
      -DCMAKE_BUILD_TYPE=${PSI_BUILD_TYPE}
      -DNOEXPORT=TRUE
      -Dtests=FALSE
      -Dexamples=FALSE
      -Dlogging=FALSE
      -Dregressions=FALSE
      -Dopenmp=FALSE
      -Dcpp=FALSE
      -DCLIBS_ONLY=FALSE
    INSTALL_DIR ${EXTERNAL_ROOT}
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
)
add_recursive_cmake_step(Lookup-Psi DEPENDEES install)

foreach(dep Eigen3 spdlog)
  lookup_package(${dep})
  if(TARGET ${dep})
    add_dependencies(Lookup-Psi ${dep})
  endif()
endforeach()
