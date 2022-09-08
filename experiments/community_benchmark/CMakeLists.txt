cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${MINIWEATHER_ML_HOME}/model model)

add_executable(driver driver.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(driver)
target_link_libraries(driver model "${MW_LINK_FLAGS}")

