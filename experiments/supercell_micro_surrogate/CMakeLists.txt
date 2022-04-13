cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

# ====PyTorch====
find_package(Torch REQUIRED)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../model model)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/custom_modules custom_modules)

add_executable(generate_micro_data generate_micro_data.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(generate_micro_data)
target_link_libraries(generate_micro_data model custom_modules "${NCFLAGS} -lyaml-cpp")

add_executable(gather_statistics gather_statistics.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(gather_statistics)
target_link_libraries(gather_statistics model custom_modules "${NCFLAGS} -lyaml-cpp")

add_executable(torch_inference_micro torch_inference_micro.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(torch_inference_micro)
# ====Update with Torch libraries====
target_link_libraries(torch_inference_micro model custom_modules "${NCFLAGS} -lyaml-cpp" ${TORCH_LIBRARIES})
set_property(TARGET torch_inference_micro PROPERTY CXX_STANDARD 14)


