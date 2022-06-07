
set(YAKL_HOME ${CMAKE_CURRENT_SOURCE_DIR}/../external/YAKL)
set(YAKL_HOME ${CMAKE_CURRENT_SOURCE_DIR}/../external/YAKL PARENT_SCOPE)
set(YAKL_BIN  ${CMAKE_CURRENT_BINARY_DIR}/yakl)
set(YAKL_BIN  ${CMAKE_CURRENT_BINARY_DIR}/yakl PARENT_SCOPE)
add_subdirectory(${YAKL_HOME} ${YAKL_BIN})
include_directories(${YAKL_BIN})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../external/ponni ponni)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(core     core   )
add_subdirectory(modules  modules)

add_library(model INTERFACE)
target_link_libraries(model INTERFACE modules core ponni)
target_include_directories(model INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

