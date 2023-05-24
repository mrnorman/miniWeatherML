cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${MINIWEATHER_ML_HOME}/model model)

add_executable(train_weno train_weno.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(train_weno)
target_link_libraries(train_weno model "${MW_LINK_FLAGS}" model)
target_include_directories(train_weno PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

