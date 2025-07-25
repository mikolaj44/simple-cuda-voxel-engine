# Parse version information from version.h:
unset(_CUB_VERSION_INCLUDE_DIR CACHE) # Clear old result to force search

# Find CMAKE_INSTALL_INCLUDEDIR=include/rapids directory"
set(from_install_prefix "../../../../")

find_path(_CUB_VERSION_INCLUDE_DIR cub/version.cuh
  REQUIRED
  NO_CMAKE_FIND_ROOT_PATH # Don't allow CMake to re-root the search
  NO_DEFAULT_PATH # Only search explicit paths below:
  PATHS
    "${CMAKE_CURRENT_LIST_DIR}/${from_install_prefix}/include/rapids"
)
set_property(CACHE _CUB_VERSION_INCLUDE_DIR PROPERTY TYPE INTERNAL)
