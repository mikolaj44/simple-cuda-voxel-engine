cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

cmake_policy(SET CMP0072 NEW)

project(simple-cuda-voxel-engine LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)



option(BUILD_EXAMPLES "Build scve examples" OFF)

set(CUDA_ARCHITECTURE_NUM "75" CACHE STRING "Set the CUDA architecture, 75 by default")



if(MSVC)
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/RTC1" "" CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")
endif()

if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Forcing multi-config generator to only use Release")
    set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
elseif(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release FORCE)
endif()



find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)



find_package(GLEW CONFIG QUIET)

if(NOT GLEW_FOUND)
    find_package(GLEW REQUIRED)
endif()

find_package(SDL2 CONFIG QUIET)

if(NOT SDL2_FOUND)
    find_package(SDL2 REQUIRED)
endif()

find_package(glfw3 CONFIG QUIET)

if(NOT glfw3_FOUND)
    find_package(glfw3 REQUIRED)
endif()



set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURE_NUM}-real;${CUDA_ARCHITECTURE_NUM}-virtual")



file(GLOB_RECURSE CPP_SOURCES "include/scve/internal/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "include/scve/internal/*.cu")

add_library(scve
  STATIC
  ${CPP_SOURCES}
  ${CUDA_SOURCES}
)

set_target_properties(scve 
  PROPERTIES
  CMAKE_CXX_STANDARD 20
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE_NUM}
  OUTPUT_NAME "scve"
)

target_include_directories(scve
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)

if(TARGET GLEW::GLEW)
    target_link_libraries(scve PUBLIC GLEW::GLEW)
else()
    target_include_directories(scve PUBLIC ${GLEW_INCLUDE_DIRS})
    target_link_libraries(scve PUBLIC ${GLEW_LIBRARIES})
endif()

if(TARGET SDL2::SDL2)
    target_link_libraries(scve PUBLIC SDL2::SDL2)
else()
    target_include_directories(scve PUBLIC ${SDL2_INCLUDE_DIRS})
    target_link_libraries(scve PUBLIC ${SDL2_LIBRARIES})
endif()

if(TARGET glfw)
    target_link_libraries(scve PUBLIC glfw)
else()
    target_include_directories(scve PUBLIC ${GLFW_INCLUDE_DIRS})
    target_link_libraries(scve PUBLIC ${GLFW_LIBRARIES})
endif()

if(TARGET OpenGL::GL)
    target_link_libraries(scve PUBLIC OpenGL::GL)
elseif(WIN32)
    target_link_libraries(scve PUBLIC opengl32)
endif()

target_link_libraries(scve
  PUBLIC
    CUDA::cudart
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
    --disable-warnings
    $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler /W0>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Xcompiler -w>
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
    	--disable-warnings
    	$<$<CXX_COMPILER_ID:MSVC>:-Xcompiler /W0>
    	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Xcompiler -w>
  	>
    )

    target_compile_options(${example_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:
        -O3
      >
    )
  endforeach()
endif()