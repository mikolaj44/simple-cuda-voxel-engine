# A simple, fully raytraced, CUDA-based voxel engine, designed for ease of use

## Features:

Inserting voxel data using (x, y, z) -> blockId functions

Inserting voxel data from blockId arrays

Saving voxel data to blockId arrays

Two texturing modes, each for 127 block types: custom textures (any resolution) or custom color mapping (you can use the default blockId -> hue mapping or provide your own blockId -> Material function)

(Work in progress) Phong reflection model based lighting with an ambient light and multiple light sources, all of which you can easily set.

An octree data structure with morton encoding, that's optimized for speed and space - it's entirely on the GPU, supports parallelized insertion and retrieval

Camera movement (mouse + keyboard controls)

## Prerequisites

You need to have the following packages installed:\
**The CUDA Toolkit**, **SDL2**, **glfw3**, **OpenGL**

You can check out the [Documentation](https://simple-cuda-voxel-engine.readthedocs.io) for a CMakeLists.txt example that installs all of these packages besides the CUDA Toolkit, which is large.

## Installation

You can clone this repo, build and install the library with CMake as shown below. The options passed in the last command are optional:

**BUILD_EXAMPLES** - builds the examples located in the examples directory, **OFF** by default

**CUDA_ARCHITECTURE_NUM** - the CUDA architecture, **75** by default

```bash
    cd path/to/simple-cuda-voxel-engine
    mkdir -p build
    cmake -S . -B build -DBUILD_EXAMPLES=OFF -DCUDA_ARCHITECTURE_NUM=75
    cmake --build build
    cmake --install build
```

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

## Sources I used:

- I implemented the morton encoding using an algorithm from [this blog post](https://forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/), you can also check out the [libmorton](https://github.com/Forceflow/libmorton) library.
- I use a modified version of the [Revelles algorithm](https://www.ugr.es/~curena/publ/2000-wscg/revelles-wscg00.pdf) that I temporarily took from [this repository](https://github.com/BadGraphixD/Cuda-Voxel-Raytracing), which is licensed under the MIT License.

## What will be added soon:

- Phong lighting math fix

## Plans for future updates:

- Removing voxel data functionality
- Custom keyboard mapping in inputLoop()