cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${MINIWEATHER_ML_HOME}/model model)

add_executable(global_llf global_llf.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(global_llf)
target_link_libraries(global_llf model "${MW_LINK_FLAGS}")
target_include_directories(global_llf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

