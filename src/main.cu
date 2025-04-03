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

#include <GL/glew.h>

#include "renderer.cuh"
#include "chunk.cuh"
#include "chunk_generation.cuh"
#include "octree.cuh"
#include "blocks_data.cuh"
#include "cuda_morton.cuh"

#define DB_PERLIN_IMPL
#include "db_perlin.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
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

    //Uint64 start = SDL_GetPerformanceCounter();

    octree->clear();
    generateChunks(octree, Vector3(0,0,0), blockSize, gridSize);
    //cudaDeviceSynchronize();

    // Uint64 end = SDL_GetPerformanceCounter();
    // float elapsed = (end - start) / (float)SDL_GetPerformanceFrequency();
    // std::cout << "Execution time: " << elapsed << " seconds" << std::endl;
}

GLuint textureID;
cudaGraphicsResource *cudaResource;
Uint64 start, end;

void createCudaTexture() {
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    
    // Register texture with CUDA
    cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

void writeAndRenderTexture() {
    cudaArray *cudaArrayPtr;
    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&cudaArrayPtr, cudaResource, 0, 0);

    // Copy data to CUDA array
    uchar4 *devPtr;
    size_t pitch;
    cudaMallocPitch(&devPtr, &pitch, SCREEN_WIDTH * sizeof(uchar4), SCREEN_HEIGHT);

    //printf("launch\n");

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    //cudaEventRecord(start);
    renderScreenCuda(octree, SCREEN_WIDTH, SCREEN_HEIGHT, cameraAngle.x, cameraAngle.y, cameraPos.x, cameraPos.y, cameraPos.z, devPtr, 32768, 512);
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Kernel execution time: %f ms (%f fps)\n", milliseconds, 1.0 / (milliseconds / 1000.0));
    //cudaDeviceSynchronize();

    // Copy memory from CUDA device to OpenGL texture
    cudaMemcpy2DToArray(cudaArrayPtr, 0, 0, devPtr, pitch, SCREEN_WIDTH * sizeof(uchar4), SCREEN_HEIGHT, cudaMemcpyDeviceToDevice);
    cudaFree(devPtr);

    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    glBegin(GL_QUADS);
        glTexCoord2f(0, 1); glVertex2f(-1,  1);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 0); glVertex2f( 1, -1);
        glTexCoord2f(1, 1); glVertex2f( 1,  1);
    glEnd();
    
    glDisable(GL_TEXTURE_2D);
}


int main(){
    
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    size_t sizeB = 64 * 1024 * 1024;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeB);
    cudaDeviceSetLimit(cudaLimitStackSize, 2048);
    
    cudaMallocManaged(&octree, sizeof(Octree));
    octree->createOctree();

    const int threadsPerBlock = 600;
    const int blocksPerGrid = (SCREEN_WIDTH * SCREEN_HEIGHT + threadsPerBlock - 1) / threadsPerBlock;

    initBlockTextures();

    SDL_Init(SDL_INIT_VIDEO);

    window = SDL_CreateWindow("voxel engine", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_OPENGL);
    //renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    //SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");

    SDL_GLContext glContext = SDL_GL_CreateContext(window);
    glewInit();

    createCudaTexture();

    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, SCREEN_HEIGHT);

    if (mouseControls) {
        SDL_SetRelativeMouseMode(SDL_TRUE);
    }

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

    while (!quit) {

        reinsertGeometry();

        SDL_Event event_;

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
                            goto end_;
                            break;

                        default:
                            break;
                        }
                    break;

                case SDL_QUIT:
                    goto end_;

                case SDL_KEYDOWN:

                    switch (event_.key.keysym.sym) {

                        case SDLK_z:
                            PLAYER_SPEED /= 2;
                            break;
                        case SDLK_x:
                            PLAYER_SPEED *= 2;
                            break;
                        case SDLK_c:
                            doOldRendering = !doOldRendering;
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

                        case SDLK_t:
                            doOldRendering = !doOldRendering;
                            break;
                            
                        default:
                            break;
                    }
            }
        }

        if (!doOldRendering) {


            
            writeAndRenderTexture();
            //cudaDeviceSynchronize();

            SDL_GL_SwapWindow(window);

            //bool showCudaErrors = true;

            // if (showCudaErrors) {
            //     cudaError_t err = cudaGetLastError();

            //     //size_t freeMem, totalMem;
            //     //cudaMemGetInfo(&freeMem, &totalMem);
            //     //printf("Free memory: %zu bytes\n", freeMem);
            //     //printf("Total memory: %zu bytes\n", totalMem);

            //     if (err != cudaSuccess) {
            //         printf("%s |\n", cudaGetErrorString(err));
            //         return 0;
            //     }
            // }

            /*size_t stackSize;
            cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
            printf("Stack size: %zu bytes\n", stackSize);*/
        }
        // else {
        //     octree->display(pixels_cpu, showBorder);
        // }

        //cout << cameraPos.x << " " << cameraPos.y << " " << cameraPos.z << endl;
        //cout << PLAYER_SPEED << endl;

        //SDL_RenderClear(renderer);

        //if(!doOldRendering){
        //    cudaMemcpy(pixels_cpu, pixels_gpu, size, cudaMemcpyDeviceToHost);
        //}

        //memset(pixels, 0, SCREEN_HEIGHT * texture_pitch);

        // SDL_UpdateTexture(texture, NULL, pixels_cpu, 1);
        // SDL_RenderCopy(renderer, texture, NULL, NULL);

        //system("pause");

        if (showFps) {
            // float elapsed = (end - start) / (float)SDL_GetPerformanceFrequency();

            // cout << to_string(1.0 / elapsed).c_str() << endl;

            //SDL_Surface* updatedSurface = TTF_RenderText_Blended(font, to_string(1.0 / elapsed).c_str(), whiteColor);
            //SDL_UpdateTexture(textTexture, nullptr, updatedSurface->pixels, updatedSurface->pitch);

            // SDL_RenderCopy(renderer, textTexture, nullptr, &textRect);

            // //cout << "FPS: " << 1.0 / elapsed << ", " << threadsPerBlock << " threads per block , " << blocksPerGrid << " blocks" << "\n";
            // // cout << blocksPerGrid * threadsPerBlock << " threads" << "\n";
            // // cout << "octree taken size (KB): " << octree->memoryTakenBytes / 1024 << "\n";
            // // cout << "octree available size (KB): " << octree->memoryAvailableBytes / 1024 << "\n\n";
            // SDL_RenderPresent(renderer);
            // SDL_FreeSurface(updatedSurface);
        }
        else {
            //SDL_RenderPresent(renderer);
        }
    }

end_:

    delete octree;

    cudaGraphicsUnregisterResource(cudaResource);
    glDeleteTextures(1, &textureID);

    SDL_FreeSurface(textSurface);
    SDL_DestroyTexture(textTexture);
    TTF_Quit();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    cudaDeviceReset();
    exit(0);

    return 0;
}