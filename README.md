# A simple, fully raytraced voxel engine written in CUDA, designed for ease of use

## Features:

Inserting voxel data using (x, y, z) -> blockId functors

Inserting voxel data from blockId arrays

Saving voxel data to blockId arrays

Two texturing modes, each for 127 block types: custom textures (any resolution) or custom color mapping (you can use the default blockId -> hue mapping or provide your own blockId -> Material functor)

(Work in progress) Phong reflection model based lighting with an ambient light and multiple light sources, all of which you can easily set.

An octree data structure with morton encoding, that's optimized for speed and space - it's entirely on the GPU, supports parallelized insertion and retrieval.

Camera movement (mouse + keyboard controls)

## Prerequisites

You need to have the following packages installed:\
The **CUDA Toolkit**, **SDL2**, **glfw3**, **GLEW**, **OpenGL**

You can check out the [Documentation](https://simple-cuda-voxel-engine.readthedocs.io) for a CMakeLists.txt example that installs this library and all of these packages besides the CUDA Toolkit.

## Installation

The library should was tested on both Linux (NVCC) and Windows (NVCC + MSVC). Only the **Release** build is available at the moment.

You can clone this repo, build and install the library with CMake as shown below. The options passed in the last command are optional:

**BUILD_EXAMPLES** - builds the examples located in the examples directory, **OFF** by default

**CUDA_ARCHITECTURE_NUM** - the CUDA architecture, **75** by default

```bash
    cd path/to/simple-cuda-voxel-engine
    mkdir -p build
    cmake -S . -B build -DBUILD_EXAMPLES=OFF -DCUDA_ARCHITECTURE_NUM=75 -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release
    cmake --install build --config Release
```

The last command may require using sudo / running the command line as administrator on Windows.

On Windows, you can get **SDL2**, **glfw3** and **GLEW** via **[vcpkg](https://github.com/microsoft/vcpkg)** and add this option to the third command:
**-DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake**

You can also add the package with **[CPM](https://github.com/cpm-cmake/CPM.cmake)** (CMake Package Manager) like so:

```
CPMAddPackage(
    NAME scve
    GITHUB_REPOSITORY mikolaj44/simple-cuda-voxel-engine
    OPTIONS
        "BUILD_EXAMPLES=OFF"
        "CUDA_ARCHITECTURE_NUM=75"
)
```

## Documentation: 
### https://simple-cuda-voxel-engine.readthedocs.io

## Usage

To use this library, you need to compile your project using the CUDA Toolkit, which also means that your source files need to be **.cu**, not **.cpp**. This is because the functors are declared as host-device as they need to be used on the GPU. Check the [Documentation](https://simple-cuda-voxel-engine.readthedocs.io) for more examples.

## Sources I used:

- I implemented the morton encoding (explained [here](https://forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/), you can also check out the [libmorton](https://github.com/Forceflow/libmorton) library) using an approach from [this](https://geidav.wordpress.com/2014/08/18/advanced-octrees-2-node-representations/) blog.
- I use a modified version of the [Revelles algorithm](https://www.ugr.es/~curena/publ/2000-wscg/revelles-wscg00.pdf) that I took from [this repository](https://github.com/BadGraphixD/Cuda-Voxel-Raytracing), which is licensed under the MIT License.

## Plans for future updates:

- Removing voxel data functionality
- Custom keyboard mapping in inputLoop()