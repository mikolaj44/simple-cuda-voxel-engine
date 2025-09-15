Usage
=====

To use this library, you need to compile your project using the CUDA Toolkit, which also means that your source files need to be .cu, not .cpp. This is because the functors are declared as host-device as they need to be used on the GPU.

The library should was tested on both Linux (NVCC) and Windows (NVCC + MSVC). Only the **Release** build is available at the moment.

Installation
------------

You need to have the following packages installed:
The **CUDA Toolkit**, **SDL2**, **glfw3**, **GLEW**, **OpenGL**

You can clone the `repo <https://github.com/mikolaj44/simple-cuda-voxel-engine>`_ , build and install the library with CMake as shown below. The options passed in the last command are optional:

**BUILD_EXAMPLES** - builds the examples located in the examples directory, **OFF** by default

**CUDA_ARCHITECTURE_NUM** - the CUDA architecture, **75** by default

.. code-block:: bash

   cd path/to/simple-cuda-voxel-engine
   mkdir -p build
   cmake -S . -B build -DBUILD_EXAMPLES=OFF -DCUDA_ARCHITECTURE_NUM=75 -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   cmake --install build --config Release

The last command may require using sudo / running the command line as administrator on Windows.

On Windows, you can get **SDL2**, **glfw3** and **GLEW** via `vcpkg <https://github.com/microsoft/vcpkg>`_ and add this option to the third command: **-DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake**

You can also add the package with `CPM <https://github.com/cpm-cmake/CPM.cmake>`_ (CMake Package Manager) like so:

.. code-block:: cmake

   CPMAddPackage(
      NAME scve
      GITHUB_REPOSITORY mikolaj44/simple-cuda-voxel-engine
      OPTIONS
         "BUILD_EXAMPLES=OFF"
         "CUDA_ARCHITECTURE_NUM=75"
   )

It should work on both Linux and Windows, although I am in the process of verifying that.

Quickstart
----------

Check out this mandelbulb fractal example from the `repo <https://github.com/mikolaj44/simple-cuda-voxel-engine/tree/main/examples>`_ :

.. literalinclude:: ../examples/mandelbulb.cu
   :language: cuda
   :linenos:

CMakeLists.txt Example
----------------------

This is an example CMakeLists.txt that you can use with the installed library if you don't have the packages from the prerequisites installed.
It's for a project with a single **your_cuda_file.cu** file.

.. literalinclude:: ExampleCMakeLists.txt
   :language: cmake
   :linenos: