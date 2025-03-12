#include <iostream>
#include <vector>
#include <cmath>
#include <map>

#include "renderer.cuh"
#include "chunk_generation.cuh"
#include "chunk.cuh"
#include "pixel_drawing.cuh"

using namespace std;

void calculateFOV() {

    halfHorFOV = atanf(SCREEN_WIDTH / (2.0 * FOCAL_LENGTH));
    halfVerFOV = atanf(SCREEN_HEIGHT / (2.0 * FOCAL_LENGTH));
}

__global__ void renderScreenCudaKernel(Octree* octree, int width, int height, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, unsigned char* pixels) {

    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index >= width * height)
        return;

    int pX = index % width;
    int pY = index / width;

    //if (!(pX == 0 && pY == 0))
    //    return;

    //printf("%f %f %f\n", oX, oY, oZ);

    float alpha;
    float polar;

    alpha = (atanf(-(pX - SCREEN_WIDTH  / 2) / FOCAL_LENGTH) - cameraAngleY + M_PI / 2); // horizontal angle
    polar = (atanf(-(pY - SCREEN_HEIGHT / 2) / FOCAL_LENGTH) + cameraAngleX + M_PI / 2); // vertical angle

    float sX = sin(polar) * cos(alpha);
    float sZ = sin(polar) * sin(alpha);
    float sY = cos(polar);

    performRaycast(octree, oX, oY, oZ, sX, sY, sZ, pX, pY, 1, pixels);
}

void renderScreenCuda(Octree* octree, int width, int height, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, unsigned char* pixels, unsigned int gridSize, unsigned int blockSize) {
    renderScreenCudaKernel<<<gridSize,blockSize>>>(octree, SCREEN_WIDTH, SCREEN_HEIGHT, cameraAngle.x, cameraAngle.y, cameraPos.x, cameraPos.y, cameraPos.z, pixels);
}

//void DrawVisibleScreenSection(int x1, int x2, int y1, int y2, Octree* octree) {
//
//    float alpha = -cameraAngle.y + M_PI / 2;
//    float polar = cameraAngle.x + M_PI / 2;
//
//    for (int pX = (x1 + x2) / 2; pX > -(x1 + x2) / 2; pX -= 1) {
//
//        alpha = -cameraAngle.y + M_PI / 2;
//
//        for (int pY = (y1 + y2) / 2; pY > -(y1 + y2) / 2; pY -= 1) {
//
//            float alpha = (atanf(pX / FOCAL_LENGTH) - cameraAngle.y + M_PI / 2); // horizontal angle
//            float polar = (atanf(pY / FOCAL_LENGTH) + cameraAngle.x + M_PI / 2); // vertical angle
//
//            float sX = sin(polar) * cos(alpha);
//            float sZ = sin(polar) * sin(alpha);
//            float sY = cos(polar);
//
//            Vector3 direction(sX,sY,sZ);
//
//            rayParameter(octree, cameraPos, direction, -pX + (x1 + x2) / 2, -pY + (y1 + y2) / 2);
//        }
//    }
//}

void DrawVisibleFaces(Octree* octree) {

    //DrawVisibleScreenSection(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT, octree);

    //vector<future<void>> results;

    //int y = 0;

    //for (int i = 0; i < MAX_THREADS_AMOUNT; i++) {

    //    y += SCREEN_HEIGHT / MAX_THREADS_AMOUNT;

    //    //results.emplace_back(
    //    threadPool.detach_task(
    //        [y, octree]
    //        {
    //            DrawVisibleScreenSection(0, SCREEN_WIDTH, 0, y, octree);
    //        });
    //    //);
    //}

    //for (auto&& result : results)
    //    result.get();

    ////threadPool.purge();
    //threadPool.wait();


    /*thread threads[MAX_THREADS_AMOUNT];

    int y = 0;

    for (int i = 0; i < MAX_THREADS_AMOUNT; i++) {

        y += SCREEN_HEIGHT / MAX_THREADS_AMOUNT;
        threads[i] = thread(DrawVisibleScreenSection, 0, SCREEN_WIDTH, 0, y);
    }

    for (int i = 0; i < MAX_THREADS_AMOUNT; i++) {

        threads[i].join();
    }*/
}