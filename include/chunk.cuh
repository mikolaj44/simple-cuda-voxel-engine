#pragma once
#include "block_variant.cuh"
#include "pixel_drawing.cuh"

#include <map>
#include <vector>

using namespace std;

class Chunk {

public:
   int x, y, z;

   Chunk() {};

   Chunk(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {};
};

extern map<pair<int, int>, bool> generatedChunks;