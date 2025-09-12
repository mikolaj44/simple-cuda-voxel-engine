# A simple, fully raytraced, CUDA-based voxel engine, designed for ease of use

# I'm currently working on turning this project into a library, working on documentation and adding the last couple of features

## Documentation (only API reference for now, will add an installation guide and examples soon):

https://simple-cuda-voxel-engine.readthedocs.io

## Sources I used:

- I implemented the morton encoding using an algorithm from [this blog post](https://forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/), you can also check out the [libmorton](https://github.com/Forceflow/libmorton) library.
- I use a modified version of the [Revelles algorithm](https://www.ugr.es/~curena/publ/2000-wscg/revelles-wscg00.pdf) that I temporarily took from [this repository](https://github.com/BadGraphixD/Cuda-Voxel-Raytracing), which is licensed under the MIT License.

## Features:

Inserting voxel data using (x, y, z) -> blockId functions

Inserting voxel data from blockId arrays

Saving voxel data to blockId arrays

Two texturing modes, each for 127 block types: custom textures (any resolution) or custom color mapping (you can use the default blockId -> hue mapping or provide your own blockId -> Material function)

(Work in progress) Phong reflection model based lighting with an ambient light and multiple light sources, all of which you can easily set.

An octree data structure with morton encoding, that's optimized for speed and space - it's entirely on the GPU, supports parallelized insertion and retrieval

Camera movement (mouse + keyboard controls)

## What will be added soon:

- Full documentation
- Phong lighting math fix
- Making this package a library that can be installed with CPM
- Better CUDA error handling


## Plans for future updates:

- Removing voxel data
- Keyboard mapping