# Parse version information from version header:
unset(_libcudacxx_VERSION_INCLUDE_DIR CACHE) # Clear old result to force search

# Find CMAKE_INSTALL_INCLUDEDIR=include/rapids/libcudacxx directory"
set(from_install_prefix "../../../../")

find_path(_libcudacxx_VERSION_INCLUDE_DIR cuda/std/detail/__config
  REQUIRED
  NO_CMAKE_FIND_ROOT_PATH # Don't allow CMake to re-root the search
  NO_DEFAULT_PATH # Only search explicit paths below:
  PATHS
    "${CMAKE_CURRENT_LIST_DIR}/${from_install_prefix}/include/rapids/libcudacxx" # Install tree
)
set_property(CACHE _libcudacxx_VERSION_INCLUDE_DIR PROPERTY TYPE INTERNAL)
