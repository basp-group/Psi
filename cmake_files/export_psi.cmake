# Exports Psi so other packages can access it
export(TARGETS psi FILE "${PROJECT_BINARY_DIR}/PsiCPPTargets.cmake")

# Avoids creating an entry in the cmake registry.
if(NOT NOEXPORT)
    export(PACKAGE Psi)
endif()

# First in binary dir
set(ALL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/cpp" "${PROJECT_BINARY_DIR}/include")
configure_File(cmake_files/PsiConfig.in.cmake
    "${PROJECT_BINARY_DIR}/PsiConfig.cmake" @ONLY
)
configure_File(cmake_files/PsiConfigVersion.in.cmake
    "${PROJECT_BINARY_DIR}/PsiConfigVersion.cmake" @ONLY
)

# Then for installation tree
file(RELATIVE_PATH REL_INCLUDE_DIR
    "${CMAKE_INSTALL_PREFIX}/share/cmake/psi"
    "${CMAKE_INSTALL_PREFIX}/include"
)
set(ALL_INCLUDE_DIRS "\${PSI_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(cmake_files/PsiConfig.in.cmake
    "${PROJECT_BINARY_DIR}/CMakeFiles/PsiConfig.cmake" @ONLY
)

# Finally install all files
install(FILES
    "${PROJECT_BINARY_DIR}/CMakeFiles/PsiConfig.cmake"
    "${PROJECT_BINARY_DIR}/PsiConfigVersion.cmake"
    DESTINATION share/cmake/psi
    COMPONENT dev
)

install(EXPORT PsiCPPTargets DESTINATION share/cmake/psi COMPONENT dev)
