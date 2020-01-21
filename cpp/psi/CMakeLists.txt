# list of headers
set(headers
  primal_dual_blocking_wideband_serial.h
  primal_dual.h primal_dual_wideband_blocking.h primal_dual_time_blocking.h primal_dual_time_blocking_debug.h logging.disabled.h maths.h proximal.h relative_variation.h
  reweighted.h reweighted_wideband.h reweighted_time_blocking.h types.h wrapper.h exception.h linear_transform.h linear_transform_operations.h logging.h positive_quadrant.h
  real_type.h sampling.h power_method.h power_method_wideband.h power_method_blocking.h preconditioned_primal_dual.h primal_dual_time_blocking.h
  serial_primal_dual_time_blocking.h ${PROJECT_BINARY_DIR}/include/psi/config.h
)
set(wavelet_headers
  wavelets/direct.h wavelets/indirect.h wavelets/innards.impl.h wavelets/sara.h
  wavelets/wavelet_data.h wavelets/wavelets.h
)

set(sources wavelets/wavelets.cc wavelets/wavelet_data.cc)
if(TIFF_FOUND)
  list(APPEND sources utilities.cc)
  list(APPEND headers utilities.h)
endif()

if(PSI_MPI)
  list(APPEND sources mpi/decomposition.cc mpi/communicator.cc mpi/types.cc wavelets/sara.cc)
  list(APPEND mpi_headers mpi/decomposition.h mpi/communicator.h mpi/types.h mpi/utilities.h wavelets.sara.h)
endif()

add_library(psi SHARED ${sources})
add_dependencies(psi lookup_dependencies)
set(version "${PSI_VERSION_MAJOR}.${PSI_VERSION_MINOR}.${PSI_VERSION_PATCH}")
set(soversion "${PSI_VERSION_MAJOR}.${PSI_VERSION_MINOR}")
set_target_properties(psi PROPERTIES VERSION ${version} SOVERSION ${soversion})

target_include_directories(psi PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include/>)

if(TIFF_FOUND)
  target_link_libraries(psi ${TIFF_LIBRARY})
  target_include_directories(psi SYSTEM PUBLIC ${TIFF_INCLUDE_DIR})
endif()
# Add spdlog as direct dependency if not downloaded
if(SPDLOG_INCLUDE_DIR AND NOT spdlog_BUILT_AS_EXTERNAL_PROJECT)
  target_include_directories(psi PUBLIC $<BUILD_INTERFACE:${SPDLOG_INCLUDE_DIR}>)
elseif(SPDLOG_INCLUDE_DIR AND spdlog_BUILT_AS_EXTERNAL_PROJECT)
  target_include_directories(psi SYSTEM PUBLIC $<BUILD_INTERFACE:${SPDLOG_INCLUDE_DIR}>)
endif()
# Add eigen as direct dependency if not downloaded
if(EIGEN3_INCLUDE_DIR AND NOT Eigen3_BUILT_AS_EXTERNAL_PROJECT)
  target_include_directories(psi PUBLIC $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>)
elseif(EIGEN3_INCLUDE_DIR AND Eigen3_BUILT_AS_EXTERNAL_PROJECT)
  target_include_directories(psi SYSTEM PUBLIC $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>)
endif()
if(TARGET openmp::openmp)
  target_link_libraries(psi openmp::openmp)
endif()

if(PSI_MPI)
  target_link_libraries(psi ${MPI_CXX_LIBRARIES})
  target_include_directories(psi SYSTEM PUBLIC ${MPI_CXX_INCLUDE_PATH})
endif()

install(FILES ${headers} DESTINATION include/psi)
install(FILES ${wavelet_headers} DESTINATION include/psi/wavelets)
install(FILES ${mpi_headers} DESTINATION include/psi/mpi)
install(TARGETS psi
  EXPORT PsiCPPTargets
  DESTINATION share/cmake/psi
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include
)
