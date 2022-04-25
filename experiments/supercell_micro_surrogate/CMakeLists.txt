cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../model model)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/custom_modules custom_modules)

include(${YAKL_HOME}/yakl_utils.cmake)

add_executable(generate_micro_data generate_micro_data.cpp)
yakl_process_target(generate_micro_data)
target_link_libraries(generate_micro_data model custom_modules "${NCFLAGS} -lyaml-cpp")

add_executable(gather_statistics gather_statistics.cpp)
yakl_process_target(gather_statistics)
target_link_libraries(gather_statistics model custom_modules "${NCFLAGS} -lyaml-cpp")

add_executable(inference_pinni inference_pinni.cpp)
yakl_process_target(inference_pinni)
target_link_libraries(inference_pinni model custom_modules "${NCFLAGS} -lyaml-cpp")

