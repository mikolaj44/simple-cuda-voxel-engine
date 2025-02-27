#include "chunk_generation.cuh"

#include <iostream>

#include "db_perlin.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void generateChunk(Octree* octree, int x_, int y_, int z_, unsigned int gridSize, unsigned int blockSize, int offsetX, int offsetY) {

    size_t numBlocks = CHUNK_W * CHUNK_W * CHUNK_H;

    thrust::device_vector<Block> chunkBlocks(numBlocks);

    for (int x = 0; x < CHUNK_W; x++) {
        for (int y = 0; y < CHUNK_H; y++) {
            for (int z = 0; z < CHUNK_W; z++) {

                float val = db::perlin((float(x) + float(x_) * CHUNK_W) / smoothing + offsetX, (float(z) + float(z_) * CHUNK_W) / smoothing + offsetY) * amplify;

                //octree->insert(x + x_ * CHUNK_W, y, z + z_ * CHUNK_W, 1, octree->root);
                //return;

                if (y >= val + 30) { //y >= val + 20

                    if(y <= val + 20.5 + 10){
                        chunkBlocks[x + y * CHUNK_W + z * CHUNK_W * CHUNK_H] = Block(x + x_ * CHUNK_W, y, z + z_ * CHUNK_W, 1); // TODO: change y to be relative too (cube chunks)
                    }

                    //     octree->insert(x + x_ * CHUNK_W, y, z + z_ * CHUNK_W, 1);
                    // else if(y <= val + 20.9 + 10)
                    //     octree->insert(x + x_ * CHUNK_W, y, z + z_ * CHUNK_W, 3);
                    // else if (y <= val + 22 + 10)
                    //     octree->insert(x + x_ * CHUNK_W, y, z + z_ * CHUNK_W, 2);
                    // else if (y <= val + 45 + 10)
                    //     octree->insert(x + x_ * CHUNK_W, y, z + z_ * CHUNK_W, 2);
                    //else
                    //    octree->insert(x + x_ * CHUNK_W, y, z + z_ * CHUNK_W, 5, octree->root);
                }
            }
        }
    }

    insert(octree, chunkBlocks, numBlocks, gridSize, blockSize);

    // TODO: use morton codes for hashing
    generatedChunks.insert(make_pair(make_pair(x_, z_), true));
}

void generateVisibleChunks(Octree* octree) {

    map<pair<int, int>, bool>::const_iterator got;

    got = generatedChunks.find(make_pair((int)cameraPos.x / CHUNK_W, (int)cameraPos.z / CHUNK_W));
    if (got == generatedChunks.end())
        generateChunk(octree, (int)cameraPos.x % CHUNK_W, 0, (int)cameraPos.z % CHUNK_W);

    float step = 0.5;

    float focalLengthX = FOCAL_LENGTH * cos(cameraAngle.x);

    float leftAngle = angleNormalize(cameraAngle.y + halfHorFOV);
    float rightAngle = angleNormalize(cameraAngle.y - halfHorFOV);

    //cout << sin(leftAngle) << " " << cos(leftAngle) << " " << sin(rightAngle) << " " << sin(rightAngle) << endl;

    float sX1 = sin(leftAngle) * step;
    float sZ1 = cos(leftAngle) * step;

    float sX2 = sin(rightAngle) * step;
    float sZ2 = cos(rightAngle) * step;

    float posX1 = cameraPos.x / (float)CHUNK_W, posZ1 = cameraPos.z / (float)CHUNK_W, posX2 = cameraPos.x / (float)CHUNK_W, posZ2 = cameraPos.z / (float)CHUNK_W;

    float distance = 0;

    //cout << sX1 << " " << sZ1 << " " << sX2 << " " << sZ2 << endl;

    while (distance < RENDER_DISTANCE) {

        posX1 += sX1;
        posZ1 += sZ1;

        posX2 += sX2;
        posZ2 += sZ2;

        vector<pair<int, int>> points = LinePoints((int)posX1, (int)posZ1, (int)posX2, (int)posZ2);

        /*for(int i = 0; i < points.size(); i++)
            cout << "(" << int(points[i].first) << " " << int(points[i].second) << ") ";
        cout << endl;*/

        for (int i = 0; i < points.size(); i++) {

            got = generatedChunks.find(make_pair(points[i].first, points[i].second));

            if (got == generatedChunks.end())
                generateChunk(octree, points[i].first, 0, points[i].second);
        }

        distance += step;
    }
}