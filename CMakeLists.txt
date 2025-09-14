cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

cmake_policy(SET CMP0072 NEW)

project(simple-cuda-voxel-engine LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)



option(BUILD_EXAMPLES "Build scve examples" ON)

set(CUDA_ARCHITECTURE_NUM "75" CACHE STRING "Set the CUDA architecture, 75 by default")



find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(SDL2 REQUIRED)
find_package(glfw3 REQUIRED)



set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURE_NUM}-real;${CUDA_ARCHITECTURE_NUM}-virtual")



file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

#file(GLOB_RECURSE HEADERS "include/*.h" "include/*.hpp" "include/*.cuh" "src/*.h" "src/*.hpp" "src/*.cuh")

add_library(scve
  STATIC
  ${CPP_SOURCES}
  ${CUDA_SOURCES}
)

#set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

set_target_properties(scve 
  PROPERTIES
  CMAKE_CXX_STANDARD 20
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  OUTPUT_NAME "engine"
)

target_include_directories(scve
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)

target_link_libraries(scve
  PRIVATE
    CUDA::cudart
    SDL2::SDL2
    glfw
    GLEW::GLEW
    GL
)

install(TARGETS scve
  EXPORT scveTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/scve
)

install(DIRECTORY include/scve
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT scveTargets
  FILE scveTargets.cmake
  NAMESPACE scve::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scve
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/scveConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scve
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/scveConfig.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scve
)



target_compile_options(scve PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    --expt-extended-lambda
    -use_fast_math
    -diag-suppress=550
    -diag-suppress=177
    -diag-suppress=1650
    -diag-suppress=20044
    -Xcompiler -w
  >
)

target_compile_options(scve PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:
    -O3
  >
)

if(BUILD_EXAMPLES)
  file(GLOB EXAMPLES "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cu")

  foreach(example_file IN LISTS EXAMPLES)
    get_filename_component(example_name ${example_file} NAME_WE)

    add_executable(${example_name} ${example_file})

    set_source_files_properties(${example_file} PROPERTIES LANGUAGE CUDA)

    target_link_libraries(${example_name} PRIVATE scve CUDA::cudart)

    set_target_properties(${example_name} 
      PROPERTIES
      CMAKE_CXX_STANDARD 20
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_compile_options(${example_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-extended-lambda
        -use_fast_math
        -diag-suppress=550
        -diag-suppress=177
        -diag-suppress=1650
        -diag-suppress=20044
        -Xcompiler -w
      >
    )

    target_compile_options(${example_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:
        -O3
      >
    )
  endforeach()
endif()