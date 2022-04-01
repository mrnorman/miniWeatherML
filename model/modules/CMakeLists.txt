
add_library(modules INTERFACE)
target_link_libraries(modules INTERFACE core)
target_include_directories(modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/helpers)

