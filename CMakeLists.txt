cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(cuda-voxel-engineLANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

include(cmake/CPM.cmake)

CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY NVIDIA/cccl
    GIT_TAG main
)

CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_ARCHITECTURES "75")

add_executable(${PROJECT_NAME} src/main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "engine")

#target_link_libraries(${PROJECT_NAME} CCCL:CCCL)
target_link_libraries(${PROJECT_NAME} cuco)