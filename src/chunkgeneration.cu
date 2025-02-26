#include "chunkgeneration.cuh"
#include "db_perlin.hpp"
#include <iostream>

void GenerateChunk(int x_, int z_, Octree* octree, float offsetX, float offsetZ) {

    for (int x = 0; x < CHUNK_W; x++) {
        for (int y = 0; y < CHUNK_H; y++) {
            for (int z = 0; z < CHUNK_W; z++) {

                float val = db::perlin((float(x) + float(x_) * CHUNK_W) / smoothing+ offsetX, (float(z) + float(z_) * CHUNK_W) / smoothing + offsetZ) * amplify;
                //val = 5;

                //octree->insert(x,y,z, rand() % 5, octree->root);

                //cout << x << " " << y << " " << z << '\n';

                //if(y > 10)

                /*if (x == 1 && y == 2 && z == 3) {
                    octree->insert(x + x_ * CHUNK_W, y, z + z_ * CHUNK_W, rand() % 4 + 1, octree->root);
                    return;
                }
                continue;*/

                //octree->insert(x + x_ * CHUNK_W, y, z + z_ * CHUNK_W, 1, octree->root);
                //return;

                if (y >= val + 30) { //y >= val + 20

                    //cout << val + 20 << endl;

                    //int val1 = rand() % 5;

                    // if(y <= val + 20.5 + 10)
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

                //if(y >= 20)
                //    chunks[chunks.size() - 1].blocks[x][y][z].type = "grass";
                //else
                //    chunks[chunks.size() - 1].blocks[x][y][z].type = "air";

                //else if(y > val + 5)
                //    chunk.blocks[x][y][z].type = "stone";
            }
        }
    }

    generatedChunks.insert(make_pair(make_pair(x_, z_), true));
}

void GenerateVisibleChunks(Octree* octree) {

    map<pair<int, int>, bool>::const_iterator got;

    got = generatedChunks.find(make_pair((int)cameraPos.x / CHUNK_W, (int)cameraPos.z / CHUNK_W));
    if (got == generatedChunks.end())
        GenerateChunk((int)cameraPos.x % CHUNK_W, (int)cameraPos.z % CHUNK_W, octree);

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
                GenerateChunk(points[i].first, points[i].second, octree);
        }

        distance += step;
    }
}