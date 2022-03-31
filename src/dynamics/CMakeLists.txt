
add_library(dynamics INTERFACE)
target_link_libraries(dynamics INTERFACE core)
target_include_directories(dynamics INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(dynamics INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

