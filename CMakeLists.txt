cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(cuda-voxel-engine LANGUAGES CXX CUDA)
# cmake_policy(SET CMP00)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")
file(GLOB_RECURSE HEADERS "include/*.h" "include/*.hpp" "include/*.cuh")

find_package(CUDAToolkit REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${SDL2_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})

set(CMAKE_CUDA_ARCHITECTURES "75")

# set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Xptxas -O3")
# set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -Xptxas -O2")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-extended-lambda
    -use_fast_math
    -diag-suppress=550
    -diag-suppress=177
    -diag-suppress=1650
    -Xcompiler -w
  >
)

target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:
    -O3
  >
)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "engine")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(${PROJECT_NAME} PRIVATE include)

target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2 SDL2_ttf::SDL2_ttf)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw GLEW::GLEW GL GLUT::GLUT)