cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${MINIWEATHER_ML_HOME}/model model)

add_executable(train_sgs train_sgs.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(train_sgs)
target_link_libraries(train_sgs model "${MW_LINK_FLAGS}" model)

