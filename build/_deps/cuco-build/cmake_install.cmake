# Install script for directory: /home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/cuco-src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/mikolaj/Desktop/cuda-voxel-engine/out/install/Configure preset using toolchain file")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/rapids-cmake-build/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/rapids" TYPE DIRECTORY FILES "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/cccl-src/cub/cub" FILES_MATCHING REGEX "/[^/]*\\.cuh$" REGEX "/[^/]*\\.hpp$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/rapids/cmake/" TYPE DIRECTORY FILES "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/cccl-src/lib/cmake/cub" REGEX ".*header-search.cmake.*" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/rapids/cmake/cub" TYPE FILE FILES "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/cccl-build/cub-header-search.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/rapids" TYPE DIRECTORY FILES "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/cccl-src/thrust/thrust" FILES_MATCHING REGEX "/[^/]*\\.h$" REGEX "/[^/]*\\.inl$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/rapids/cmake/" TYPE DIRECTORY FILES "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/cccl-src/lib/cmake/thrust" REGEX ".*header-search.cmake.*" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/rapids/cmake/thrust" TYPE FILE FILES "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/cccl-build/thrust-header-search.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/rapids/libcudacxx" TYPE DIRECTORY FILES "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/cccl-src/libcudacxx/include/cuda" FILES_MATCHING REGEX "/[^/]*$" REGEX "/CMakeLists\\.txt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/rapids/libcudacxx" TYPE DIRECTORY FILES "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/cccl-src/libcudacxx/include/nv" FILES_MATCHING REGEX "/[^/]*$" REGEX "/CMakeLists\\.txt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/rapids/cmake/" TYPE DIRECTORY FILES "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/cccl-src/lib/cmake/libcudacxx" REGEX ".*header-search.cmake.*" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/rapids/cmake/libcudacxx" TYPE FILE FILES "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/cccl-build/libcudacxx-header-search.cmake")
endif()

