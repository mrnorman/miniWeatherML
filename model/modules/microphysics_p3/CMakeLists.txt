
set(P3_SRC scream_types.F90
           micro_p3_utils.F90
           scream_abortutils.F90
           wv_sat_scream.F90
           micro_p3.F90)

add_library(microphysics_p3 ${P3_SRC})
target_include_directories(microphysics_p3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(microphysics_p3 PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(microphysics_p3 core)
set_source_files_properties(${P3_SRC} PROPERTIES COMPILE_FLAGS "${YAKL_F90_FLAGS}")

