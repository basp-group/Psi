if(logging)
  add_library(common_catch_main_object OBJECT "common_catch_main.cc")
  if(SPDLOG_INCLUDE_DIR)
    target_include_directories(common_catch_main_object SYSTEM PUBLIC ${SPDLOG_INCLUDE_DIR})
  endif()
  if(CATCH_INCLUDE_DIR)
    target_include_directories(common_catch_main_object SYSTEM PUBLIC ${CATCH_INCLUDE_DIR})
  endif()
  target_include_directories(common_catch_main_object PUBLIC
    "${PROJECT_BINARY_DIR}/include/"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
  )
  add_dependencies(common_catch_main_object lookup_dependencies)
endif()

add_catch_test(bisection_method LIBRARIES psi)
add_catch_test(chained_operators LIBRARIES psi)
add_catch_test(wavelets LIBRARIES psi)
add_catch_test(sara LIBRARIES psi)
add_catch_test(maths LIBRARIES psi)
add_catch_test(wrapper LIBRARIES psi)
add_catch_test(conjugate_gradient LIBRARIES psi)
add_catch_test(linear_transform LIBRARIES psi)
add_catch_test(proximal LIBRARIES psi)
add_catch_test_with_seed(seeded_proximal test_proximal 1449580491)
add_catch_test(reweighted LIBRARIES psi)
add_catch_test(power_method LIBRARIES psi)
add_catch_test(power_method_blocking LIBRARIES psi)
add_catch_test(primal_dual LIBRARIES psi)
add_catch_test(forward_backward LIBRARIES psi)
add_catch_test(preconditioned_primal_dual LIBRARIES psi)
add_catch_test(sort_utils LIBRARIES psi)

if(PSI_MPI)
  add_library(common_mpi_catch_main_object OBJECT "common_mpi_catch_main.cc")
  add_dependencies(common_mpi_catch_main_object lookup_dependencies)
  target_include_directories(common_mpi_catch_main_object
    PUBLIC ${PROJECT_SOURCE_DIR}/cpp ${PROJECT_BINARY_DIR}/include ${MPI_CXX_INCLUDE_PATH} ${EIGEN3_INCLUDE_DIR})
  if(SPDLOG_INCLUDE_DIR)
    target_include_directories(common_mpi_catch_main_object SYSTEM PUBLIC ${SPDLOG_INCLUDE_DIR})
  endif()
  if(CATCH_INCLUDE_DIR)
    target_include_directories(common_mpi_catch_main_object SYSTEM PUBLIC ${CATCH_INCLUDE_DIR})
  endif()

  function(add_mpi_test_from_test testname)
    unset(arguments)
    if(CATCH_JUNIT)
      set(arguments -r junit -o ${PROJECT_BINARY_DIR}/Testing/${testname}.xml)
    endif()
    if(NOT MPIEXEC_MAX_NUMPROCS)
      set(MPIEXEC_MAX_NUMPROCS 4)
    endif()
    add_test(NAME ${testname}
      COMMAND
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:test_${testname}> ${arguments})
    set_tests_properties(${testname} PROPERTIES LABELS "catch;mpi")
  endfunction()

  function(add_mpi_test testname)
    add_catch_test(${testname} COMMON_MAIN common_mpi_catch_main_object NOTEST ${ARGN})
    add_mpi_test_from_test(${testname})
  endfunction()

  add_mpi_test(mpi LIBRARIES psi)
  add_mpi_test(decomposition LIBRARIES psi)
  add_mpi_test(io LIBRARIES psi)
  add_mpi_test(scalapack LIBRARIES psi)
endif()

