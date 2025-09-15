Usage
=====

To use this library, you need to compile your project using the CUDA Toolkit, which also means that your source files need to be .cu, not .cpp. This is because the functors are declared as host-device as they need to be used on the GPU.

Installation
------------

You can clone the `repo <https://github.com/mikolaj44/simple-cuda-voxel-engine>`_ , build and install the library with CMake as shown below. The options passed in the last command are optional:

**BUILD_EXAMPLES** - builds the examples located in the examples directory, **OFF** by default

**CUDA_ARCHITECTURE_NUM** - the CUDA architecture, **75** by default

.. code-block:: bash

   cd path/to/simple-cuda-voxel-engine
   mkdir -p build
   cmake -S . -B build -DBUILD_EXAMPLES=OFF -DCUDA_ARCHITECTURE_NUM=75
   cmake --build build
   sudo cmake --install build

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

Check out this mandelbulb fractal example from the **`repo <https://github.com/mikolaj44/simple-cuda-voxel-engine/tree/main/examples>`_**:

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