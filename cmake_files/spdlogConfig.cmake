find_package(Threads REQUIRED)

set(SPDLOG_FMT_EXTERNAL @SPDLOG_FMT_EXTERNAL@)
set(config_targets_file @config_targets_file@)

if(SPDLOG_FMT_EXTERNAL)
    include(CMakeFindDependencyMacro)
    find_dependency(fmt CONFIG)
endif()


include("${CMAKE_CURRENT_LIST_DIR}/${config_targets_file}")
