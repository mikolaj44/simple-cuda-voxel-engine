# Work in progress, I'm currently refactoring the code so it's not ready to use

# Simple CUDA voxel engine

## Sources I used:

- I implemented the morton encoding using an algorithm from [this blog post](https://forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/) and I'm also using the [libmorton](https://github.com/Forceflow/libmorton) library.
- I use a modified version of the [Revelles algorithm](https://www.ugr.es/~curena/publ/2000-wscg/revelles-wscg00.pdf) that I temporarily took from [this repository](https://github.com/BadGraphixD/Cuda-Voxel-Raytracing), which is licensed under the MIT License.


## What the project is at the moment:

An easy to use voxel engine that allows for constructing custom shapes.

It supports two texturing modes: custom textures (any resolution) or color from hue (currently 128 colors). 

It uses an octree data structure with morton encoding, that's optimized for speed and space - it's entirely on the GPU, supports parallelized insertion.

Phong reflection model is used for the lighting, although it only supports single source right now.

## What I will add soon:

- Ability to pass a function defining custom materials for each block type

- Multiple light sources

- Examples and and a guide
