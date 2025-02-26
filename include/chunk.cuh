#pragma once
#include "block.cuh"
#include "pixeldrawing.cuh"

#include <map>
#include <vector>

using namespace std;

//struct Chunk {
//
//    int x;
//    int z;
//    Block blocks[CHUNK_W][CHUNK_H][CHUNK_W];
//
//    Chunk(int x_, int z_);
//
//    Chunk(int x_, int z_, Block blocks_[CHUNK_W][CHUNK_H][CHUNK_W]);
//
//    static int coordBlockToChunkSpace(int coord);
//};

extern map<pair<int, int>, bool> generatedChunks;