cmake_minimum_required(VERSION 3.0)
project(miniCloudML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

set(YAKL_HOME ${CMAKE_CURRENT_SOURCE_DIR}/../external/YAKL)
set(YAKL_BIN  ${CMAKE_CURRENT_BINARY_DIR}/yakl)
add_subdirectory(${YAKL_HOME} ${YAKL_BIN})
include_directories(${YAKL_BIN})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(coupler coupler)

add_executable(driver driver.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(driver)
target_link_libraries(driver coupler "${NCFLAGS} -lyaml-cpp")

