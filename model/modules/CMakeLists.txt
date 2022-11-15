
add_subdirectory(microphysics_p3 microphysics_p3)

add_library(modules INTERFACE)
target_link_libraries(modules INTERFACE microphysics_p3 core)
target_include_directories(modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/microphysics_p3)
target_include_directories(modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/dynamics_compr_nh_rho_theta)

