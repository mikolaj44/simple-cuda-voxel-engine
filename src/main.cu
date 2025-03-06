#define SDL_MAIN_HANDLED

#include <SDL.h>
#include <SDL_ttf.h>
#include <iostream>
#include <ctime>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <bitset>
#include <chrono>

#include "renderer.cuh"
#include "chunk.cuh"
#include "chunk_generation.cuh"
#include "octree.cuh"
#include "blocks_data.cuh"
#include "cuda_morton.cuh"

#define DB_PERLIN_IMPL
#include "db_perlin.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

SDL_Window* window;
SDL_Renderer* renderer;
SDL_Texture* texture;

BlockTexture** blockTextures = nullptr;

void initBlockTextures() {

    int blockAmount = 3;

    cudaMallocManaged(&blockTextures, size_t(blockAmount * sizeof(BlockTexture*)));

    for (int i = 1; i < blockAmount + 1; i++) {

        string currPathStr = pathStr + "/res/textures/" + to_string(i);

        cudaMallocManaged(&blockTextures[i], sizeof(BlockTexture));
        new (blockTextures[i]) BlockTexture(16, 16, currPathStr + "/top.png", currPathStr + "/bottom.png", currPathStr + "/left.png", currPathStr + "/right.png", currPathStr + "/front.png", currPathStr + "/back.png");
    }

    createBlocksData<<<1,1>>>(blockTextures);

    cudaDeviceSynchronize();
}

void handleCameraMovement(int mouseX, int mouseY, int& prevMouseX, int& prevMouseY) {

    mouseX -= SCREEN_WIDTH / 2;
    mouseY -= SCREEN_HEIGHT / 2;

    cameraAngle.y -= (prevMouseX - mouseX) * MOUSE_SENSITIVITY;
    cameraAngle.x += (prevMouseY - mouseY) * MOUSE_SENSITIVITY;

    if (cameraAngle.x < -M_PI / 2.0) {
        cameraAngle.x = -M_PI / 2.0;
    }
    else if (cameraAngle.x > M_PI / 2.0) {
        cameraAngle.x = M_PI / 2.0;
    }

    //cout << cameraAngle.x << endl;

    prevMouseX = mouseX;
    prevMouseY = mouseY;
}

Octree* octree;

dim3 blockSize(40,40,40);
dim3 gridSize(10,10,10);

inline void reinsertGeometry(){

    //clock_t start_ = clock();

    octree->clear();
    generateChunks(octree, cameraPos, blockSize, gridSize);
    //cudaDeviceSynchronize();

    //clock_t end_ = clock();

    //double elapsed = double(end_ - start_) / CLOCKS_PER_SEC;
    //std::cout << "Execution time: " << elapsed << " seconds" << std::endl;
}

int main(){
    
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    size_t sizeB = 64 * 1024 * 1024;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeB);
    cudaDeviceSetLimit(cudaLimitStackSize, 2048);
    
    cudaMallocManaged(&octree, sizeof(Octree));
    octree->createOctree();

    // cout << bitset<64>(octree_morton3D_64_encode(0,0,0,0,octree->xMin, octree->yMin, octree->zMin, octree->level)) << endl;
    // cout << bitset<64>(octree_morton3D_64_encode(1 << 17,0,0, 0,octree->xMin, octree->yMin, octree->zMin, octree->level)) << endl;
    // cout << bitset<64>(octree_morton3D_64_encode(0,1 << 17,0, 0,octree->xMin, octree->yMin, octree->zMin, octree->level)) << endl;
    // //cout << bitset<64>(octree_morton3D_64_encode(1 << 18 - 5,1 << 18 - 6,1 << 18 - 7,0,18)) << endl;
    // return 0;

    //octree = new Octree(-pow2neg, pow2, -pow2neg, pow2, -pow2neg, pow2);

    size_t size = SCREEN_WIDTH * SCREEN_HEIGHT * 4;

    unsigned char* pixels = new unsigned char[size]; // cpu
    unsigned char* pixels_gpu; // gpu

    cudaMalloc(&pixels_gpu, size * sizeof(unsigned char));

    const int threadsPerBlock = 700;
    const int blocksPerGrid = (SCREEN_WIDTH * SCREEN_HEIGHT + threadsPerBlock - 1) / threadsPerBlock;

    initBlockTextures();

    //cudaMemcpy(pixels, pixels_gpu, bytes, cudaMemcpyHostToDevice);

    window = SDL_CreateWindow("voxel engine", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, 0);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");

    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, SCREEN_HEIGHT);

    if (mouseControls) {
        SDL_SetRelativeMouseMode(SDL_TRUE);
    }

    int texture_pitch = 0;
    void* texture_pixels = NULL;

    if (SDL_LockTexture(texture, NULL, &texture_pixels, &texture_pitch) != 0) {
        SDL_Log("Unable to lock texture: %s", SDL_GetError());
    }
    else {
        memcpy(texture_pixels, pixels, texture_pitch * SCREEN_HEIGHT);
    }
    SDL_UnlockTexture(texture);

    TTF_Init();

    string fontPath = pathStr + "/res/fonts/Arial.ttf";

    TTF_Font* font = TTF_OpenFont(fontPath.c_str(), 28);

    SDL_Color whiteColor = { 255, 255, 255 };

    SDL_Surface* textSurface = TTF_RenderText_Blended(font, "text", whiteColor);

    SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);

    SDL_Rect textRect;
    textRect.x = 0;
    textRect.y = 0;
    textRect.w = textSurface->w;
    textRect.h = textSurface->h;

    SDL_FreeSurface(textSurface);

    bool quit = false;

    calculateFOV();

    int prevMouseX = 0, prevMouseY = 0;

    reinsertGeometry();

    while (!quit) {

        Uint64 start = SDL_GetPerformanceCounter();

        SDL_Event event_;

        float offsetX, offsetZ;
        int cubeSize;

        // if (doGravity) {

        //     unsigned char check = octree->get((int)cameraPos.x, (int)(cameraPos.y + 2), (int)cameraPos.z);

        //     cout << (int)check << endl;

        //     if (check == 0 || check == 255)
        //         cameraPos.y += 0.5;
        // }

        while (SDL_PollEvent(&event_)) {

            switch (event_.type) {

                int x, y, z;

                case SDL_MOUSEMOTION:
                    if (mouseControls) {
                        handleCameraMovement(event_.motion.x, event_.motion.y, prevMouseX, prevMouseY);
                    }
                    break;

                case SDL_WINDOWEVENT:

                    switch (event_.window.event) {

                        case SDL_WINDOWEVENT_CLOSE:   // exit game
                            goto end;
                            break;

                        default:
                            break;
                        }
                    break;

                case SDL_QUIT:
                    goto end;

                case SDL_KEYDOWN:

                    switch (event_.key.keysym.sym) {

                    case SDLK_1:

                        // for (int x = -START_RENDER_DISTANCE; x < START_RENDER_DISTANCE; x++)
                        //     for (int z = -START_RENDER_DISTANCE; z < START_RENDER_DISTANCE; z++)
                        //         if (x * x + z * z <= START_RENDER_DISTANCE * START_RENDER_DISTANCE) {
                        //             //generateChunk(octree, x, 0, z);
                        //         }

                        // break;

                    case SDLK_z:
                        PLAYER_SPEED /= 2;
                        break;
                    case SDLK_x:
                        PLAYER_SPEED *= 2;
                        break;
                    case SDLK_c:
                        doOldRendering = !doOldRendering;
                        break; 

                    case SDLK_2:

                        cubeSize = 100;

                        for (int x = -cubeSize / 2; x < cubeSize / 2; x++) {
                            for (int y = -cubeSize / 2; y < cubeSize / 2; y++) {
                                for (int z = -cubeSize / 2; z < cubeSize / 2; z++) {
                                    //octree->insert(x, y, z, rand() % 255 + 1);
                                }
                            }
                        }

                        break;

                    case SDLK_3:

                        cubeSize = 100;

                        // for (int x = -cubeSize / 2; x < cubeSize / 2; x++) {
                        //     for (int y = -cubeSize / 2; y < cubeSize / 2; y++) {
                        //         for (int z = -cubeSize / 2; z < cubeSize / 2; z++) {
                        //             if (rand() % 200 == 1)
                        //                 //octree->insert(x, y, z, rand() % 400 + 1);
                        //         }
                        //     }
                        // }

                        break;

                    case SDLK_4:

                        cubeSize = 50;

                        // for (int x = -cubeSize * 2; x < cubeSize * 2; x++) {
                        //     for (int y = -cubeSize * 2; y < cubeSize * 2; y++) {
                        //         for (int z = -cubeSize * 2; z < cubeSize * 2; z++) {
                        //             if (x * x + y * y + z * z <= cubeSize * cubeSize)
                        //                 //octree->insert(x, y, z, rand() % 700 + 1);
                        //         }
                        //     }
                        // }

                        break;

                    case SDLK_5:

                        break;

                        case SDLK_UP:
                            cameraPos.x += sin(cameraAngle.y) * PLAYER_SPEED;
                            cameraPos.z += cos(cameraAngle.y) * PLAYER_SPEED;
                            break;
                        case SDLK_DOWN:
                            cameraPos.x -= sin(cameraAngle.y) * PLAYER_SPEED;
                            cameraPos.z -= cos(cameraAngle.y) * PLAYER_SPEED;
                            break;

                        case SDLK_LEFT:
                            cameraPos.x -= sin(cameraAngle.y + M_PI/2) * PLAYER_SPEED;
                            cameraPos.z -= cos(cameraAngle.y + M_PI/2) * PLAYER_SPEED;
                            break;
                        case SDLK_RIGHT:
                            cameraPos.x += sin(cameraAngle.y + M_PI/2) * PLAYER_SPEED;
                            cameraPos.z += cos(cameraAngle.y + M_PI/2) * PLAYER_SPEED;
                            break;

                        case SDLK_s:
                            cameraPos.y += PLAYER_SPEED;
                            break;
                        case SDLK_w:
                            cameraPos.y -= PLAYER_SPEED;
                            break;

                        case SDLK_q:
                            cameraAngle.y -= PLAYER_TURN_Y_SPEED;
                            break;
                        case SDLK_e:
                            cameraAngle.y += PLAYER_TURN_Y_SPEED;
                            break;

                        case SDLK_r:
                            cameraAngle.x += 0.1;
                            if (cameraAngle.x > 2 * M_PI)
                                cameraAngle.x = 0;
                            break;

                        case SDLK_f:
                            cameraAngle.x -= 0.1;
                            if (cameraAngle.x < 0)
                                cameraAngle.x = 2 * M_PI;
                            break;

                        case SDLK_i:
                            x = rand() % 128 - 64;
                            y = rand() % 128 - 64;
                            z = rand() % 128 - 64;
                            cout << x << " " << y << " " << z << endl;
                            //octree->insert(x,y,z, rand() % 5);
                            break;

                        case SDLK_o:
                            x = rand() % 256 - 128;
                            y = rand() % 256 - 128;
                            z = rand() % 256 - 128;
                            cout << x << " " << y << " " << z << endl;
                            //octree->insert(x, y, z, rand() % 5);
                            break;

                        case SDLK_t:
                            doOldRendering = !doOldRendering;
                            break;

                        case SDLK_9:
                            shiftX++;
                            break;
                        case SDLK_0:
                            shiftX--;
                            break;

                        case SDLK_n:
                            shiftY++;
                            break;
                        case SDLK_m:
                            shiftY--;
                            break;

                        case SDLK_h:
                            shiftZ++;
                            break;
                        case SDLK_j:
                            shiftZ--;
                            break;

                        case SDLK_k:

                            //shift--;

                            offsetX = rand() % 100;
                            offsetZ = rand() % 100;

                            //for (int x = -START_RENDER_DISTANCE * 2; x < START_RENDER_DISTANCE * 2; x++)
                             //   for (int z = -START_RENDER_DISTANCE * 2; z < START_RENDER_DISTANCE * 2; z++)
                               //     if (x*x + z*z <= 4 * START_RENDER_DISTANCE * START_RENDER_DISTANCE)
                                        //generateChunk(octree, 0, 0, 0, offsetX, offsetZ);

                            break;

                        case SDLK_l:
                            shift++;
                            break;

                        case SDLK_b:
                            showBorder = !showBorder;
                            break;

                        default:
                            break;
                    }
            }
        }

        if (!doOldRendering) {

            // allocate space on device for the host octree (calculate the size)
            // copy it

            //-dc -G -lineinfo 

            renderScreenCuda(octree, SCREEN_WIDTH, SCREEN_HEIGHT, cameraAngle.x, cameraAngle.y, cameraPos.x, cameraPos.y, cameraPos.z, pixels_gpu, blocksPerGrid, threadsPerBlock);
            //cudaDeviceSynchronize();
            //DrawVisibleFaces(octree);

            bool showCudaErrors = true;

            if (showCudaErrors) {
                cudaError_t err = cudaGetLastError();

                //size_t freeMem, totalMem;
                //cudaMemGetInfo(&freeMem, &totalMem);
                //printf("Free memory: %zu bytes\n", freeMem);
                //printf("Total memory: %zu bytes\n", totalMem);

                if (err != cudaSuccess) {
                    printf("%s |\n", cudaGetErrorString(err));
                    return 0;
                }
            }

            /*size_t stackSize;
            cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
            printf("Stack size: %zu bytes\n", stackSize);*/
        }
        else {
            octree->display(pixels, showBorder);
        }

        //cout << cameraPos.x << " " << cameraPos.y << " " << cameraPos.z << endl;
        //cout << PLAYER_SPEED << endl;

        SDL_RenderClear(renderer);

        SDL_LockTexture(texture, NULL, &texture_pixels, &texture_pitch);

        if(!doOldRendering){
            cudaMemcpy(texture_pixels, pixels_gpu, size, cudaMemcpyDeviceToHost);
        }
        else{
            memcpy(texture_pixels, pixels, size * sizeof(unsigned char));
        }

        SDL_UnlockTexture(texture);

        memset(pixels, 0, SCREEN_HEIGHT * texture_pitch);
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        //system("pause");

        if (showFps) {

            Uint64 end = SDL_GetPerformanceCounter();
            float elapsed = (end - start) / (float)SDL_GetPerformanceFrequency();

            SDL_Surface* updatedSurface = TTF_RenderText_Blended(font, to_string(1.0 / elapsed).c_str(), whiteColor);
            SDL_UpdateTexture(textTexture, nullptr, updatedSurface->pixels, updatedSurface->pitch);

            SDL_RenderCopy(renderer, textTexture, nullptr, &textRect);

            /*cout << "FPS: " << 1.0 / elapsed << ", " << threadsPerBlock << " threads per block , " << blocksPerGrid << " blocks" << "\n";
            cout << blocksPerGrid * threadsPerBlock << " threads" << "\n";
            cout << "octree taken size (KB): " << octree->memoryTakenBytes / 1024 << "\n";
            cout << "octree available size (KB): " << octree->memoryAvailableBytes / 1024 << "\n\n";*/
            SDL_RenderPresent(renderer);
            SDL_FreeSurface(updatedSurface);
        }
        else {
            SDL_RenderPresent(renderer);
        }
    }

end:

    delete octree;

    SDL_FreeSurface(textSurface);
    SDL_DestroyTexture(textTexture);
    TTF_Quit();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    cudaFree(pixels_gpu);

    cudaDeviceReset();
    exit(0);

    delete[] pixels;
    delete[] (char*)texture_pixels;

    return 0;
}

//got = chunkPosIndex.find(make_pair((int)cameraPos.x / CHUNK_W + 1, (int)cameraPos.z / CHUNK_W + 1));
//if (got == chunkPosIndex.end())
//GenerateChunk((int)cameraPos.x / CHUNK_W + 1, (int)cameraPos.z / CHUNK_W + 1);
//
//got = chunkPosIndex.find(make_pair((int)cameraPos.x / CHUNK_W, (int)cameraPos.z / CHUNK_W + 1));
//if (got == chunkPosIndex.end())
//GenerateChunk((int)cameraPos.x / CHUNK_W, (int)cameraPos.z / CHUNK_W + 1);
//
//got = chunkPosIndex.find(make_pair((int)cameraPos.x / CHUNK_W + 1, (int)cameraPos.z / CHUNK_W));
//if (got == chunkPosIndex.end())
//GenerateChunk((int)cameraPos.x / CHUNK_W + 1, (int)cameraPos.z / CHUNK_W);
//
//got = chunkPosIndex.find(make_pair((int)cameraPos.x / CHUNK_W + 1, (int)cameraPos.z / CHUNK_W - 1));
//if (got == chunkPosIndex.end())
//GenerateChunk((int)cameraPos.x / CHUNK_W + 1, (int)cameraPos.z / CHUNK_W - 1);
//
//got = chunkPosIndex.find(make_pair((int)cameraPos.x / CHUNK_W - 1, (int)cameraPos.z / CHUNK_W + 1));
//if (got == chunkPosIndex.end())
//GenerateChunk((int)cameraPos.x / CHUNK_W - 1, (int)cameraPos.z / CHUNK_W + 1);
//
//got = chunkPosIndex.find(make_pair((int)cameraPos.x / CHUNK_W - 1, (int)cameraPos.z / CHUNK_W - 1));
//if (got == chunkPosIndex.end())
//GenerateChunk((int)cameraPos.x / CHUNK_W - 1, (int)cameraPos.z / CHUNK_W - 1);
//
//got = chunkPosIndex.find(make_pair((int)cameraPos.x / CHUNK_W - 1, (int)cameraPos.z / CHUNK_W));
//if (got == chunkPosIndex.end())
//GenerateChunk((int)cameraPos.x / CHUNK_W - 1, (int)cameraPos.z / CHUNK_W);
//
//got = chunkPosIndex.find(make_pair((int)cameraPos.x / CHUNK_W, (int)cameraPos.z / CHUNK_W - 1));
//if (got == chunkPosIndex.end())
//GenerateChunk((int)cameraPos.x / CHUNK_W, (int)cameraPos.z / CHUNK_W - 1);