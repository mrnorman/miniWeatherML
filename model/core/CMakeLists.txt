
add_library(core coupler_globals.cpp)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(core)

