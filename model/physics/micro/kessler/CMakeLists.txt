
set(MICRO_SRC kessler.f90)

add_library(micro ${MICRO_SRC})
target_include_directories(micro PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

