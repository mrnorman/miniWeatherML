
add_library(custom_modules INTERFACE)
target_link_libraries(custom_modules INTERFACE core)
target_include_directories(custom_modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

