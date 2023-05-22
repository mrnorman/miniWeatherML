
set(YAKL_HOME ${CMAKE_CURRENT_SOURCE_DIR}/../external/YAKL)
set(YAKL_HOME ${CMAKE_CURRENT_SOURCE_DIR}/../external/YAKL PARENT_SCOPE)
set(YAKL_BIN  ${CMAKE_CURRENT_BINARY_DIR}/yakl)
set(YAKL_BIN  ${CMAKE_CURRENT_BINARY_DIR}/yakl PARENT_SCOPE)
add_subdirectory(${YAKL_HOME} ${YAKL_BIN})
include_directories(${YAKL_BIN})

set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../external/yaml-cpp ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../external/ponni ponni)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(core     core   )
add_subdirectory(modules  modules)

add_library(model INTERFACE)
target_link_libraries(model INTERFACE modules core ponni yaml-cpp)
target_include_directories(model INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../external/yaml-cpp/include)

