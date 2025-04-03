cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(cuda-voxel-engineLANGUAGES CXX CUDA)
# cmake_policy(SET CMP00)

file(GLOB SOURCES "src/*.cpp" "src/*.cu")
file(GLOB HEADERS "include/*.h" "include/*.hpp" "include/*.cuh")

find_package(CUDAToolkit REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${SDL2_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})

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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -rdc=true -use_fast_math -w -Xcompiler -w")
set(CMAKE_CUDA_ARCHITECTURES "75")

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Xptxas -O3")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -Xptxas -O2")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "engine")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(${PROJECT_NAME} PRIVATE include)

#target_link_libraries(${PROJECT_NAME} CCCL:CCCL)
target_link_libraries(${PROJECT_NAME} PRIVATE cuco)
target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2 SDL2_ttf::SDL2_ttf)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw GLEW::GLEW GL GLUT::GLUT)