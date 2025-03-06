# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/rapids-cmake-src")
  file(MAKE_DIRECTORY "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/rapids-cmake-src")
endif()
file(MAKE_DIRECTORY
  "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/rapids-cmake-build"
  "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix"
  "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/tmp"
  "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp"
  "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src"
  "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
