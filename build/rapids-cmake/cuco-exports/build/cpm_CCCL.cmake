#=============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================


# CPM Search for CCCL
#
# Make sure we search for a build-dir config module for the CPM project
set(possible_package_dir "/home/mikolaj/Desktop/cuda-voxel-engine/build/_deps/cccl-build")
if(possible_package_dir AND NOT DEFINED CCCL_DIR)
  set(CCCL_DIR "${possible_package_dir}")
endif()

CPMFindPackage(
  "NAME;CCCL;VERSION;2.7.0;FIND_PACKAGE_ARGUMENTS;EXACT;GIT_REPOSITORY;https://github.com/NVIDIA/cccl.git;GIT_TAG;v2.7.0;GIT_SHALLOW;OFF;PATCH_COMMAND;/usr/bin/cmake;-P;/home/mikolaj/Desktop/cuda-voxel-engine/build/rapids-cmake/patches/CCCL/patch.cmake;EXCLUDE_FROM_ALL;OFF;OPTIONS;CCCL_TOPLEVEL_PROJECT OFF;CCCL_ENABLE_INSTALL_RULES ON"
  )

if(possible_package_dir)
  unset(possible_package_dir)
endif()
#=============================================================================
