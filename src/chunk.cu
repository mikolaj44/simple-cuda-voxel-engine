#include "chunk.cuh"

using namespace std;

//Chunk::Chunk(int x_, int z_) {
//
//    x = x_;
//    z = z_;
//}
//
//Chunk::Chunk(int x_, int z_, Block blocks_[CHUNK_W][CHUNK_H][CHUNK_W]) {
//
//    x = x_;
//    z = z_;
//
//    for (int x = 0; x < CHUNK_W; x++)
//        for (int y = 0; y < CHUNK_H; y++)
//            for (int z = 0; z < CHUNK_W; z++) {
//                blocks[x][y][z].type = blocks_[x][y][z].type;
//                //blocks[x][y][z].lightLevel = blocks_[x][y][z].lightLevel;
//            }
//}
//
//int Chunk::coordBlockToChunkSpace(int coord) {
//
//    coord = coord % CHUNK_W;
//
//    if (coord < 0)
//        coord += CHUNK_W;
//
//    return coord;
//}

map<pair<int, int>, bool> generatedChunks;