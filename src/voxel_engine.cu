#include "voxel_engine.h"
#include "renderer.cuh"
#include "globals.cuh"
#include "blocks_data.cuh"

uint64_t VoxelEngine::frameCount = 0;

int VoxelEngine::prevMouseX = 0;
int VoxelEngine::prevMouseY = 0;

SDL_Event VoxelEngine::event_;

dim3 VoxelEngine::maxGridSize(32768,32768,32768);
dim3 VoxelEngine::blockSize(9,9,9);

BlockTexture** VoxelEngine::blockTextures = nullptr;

Octree* VoxelEngine::octree;

SDL_Window* VoxelEngine::window;
SDL_Renderer* VoxelEngine::renderer;
SDL_Texture* VoxelEngine::texture;

SDL_Surface* VoxelEngine::textSurface;
SDL_Texture* VoxelEngine::textTexture;

GLuint VoxelEngine::textureID;
cudaGraphicsResource* VoxelEngine::cudaResource;

bool VoxelEngine::wasInitialized = false;

void VoxelEngine::init(int SCREEN_WIDTH_, int SCREEN_HEIGHT_){
    if(wasInitialized){
        return;
    }

    cudaMemcpyToSymbol(SCREEN_WIDTH_DEVICE, &SCREEN_WIDTH_, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(SCREEN_HEIGHT_DEVICE, &SCREEN_HEIGHT_, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

    SCREEN_WIDTH_HOST = SCREEN_WIDTH_;
    SCREEN_HEIGHT_HOST = SCREEN_HEIGHT_;

    cudaMallocManaged(&octree, sizeof(Octree));
    octree->createOctree();

    const int threadsPerBlock = 600;
    const int blocksPerGrid = (SCREEN_WIDTH_HOST * SCREEN_HEIGHT_HOST + threadsPerBlock - 1) / threadsPerBlock;

    initBlockTextures();

    SDL_Init(SDL_INIT_VIDEO);

    window = SDL_CreateWindow("voxel engine", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH_HOST, SCREEN_HEIGHT_HOST, SDL_WINDOW_OPENGL);

    SDL_GLContext glContext = SDL_GL_CreateContext(window);
    glewInit();

    createCudaTexture();

    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH_HOST, SCREEN_HEIGHT_HOST);

    SDL_SetRelativeMouseMode(SDL_TRUE);

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

    //Renderer::calculateFOV();

    wasInitialized = true;
}

void VoxelEngine::createCudaTexture() {
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SCREEN_WIDTH_HOST, SCREEN_HEIGHT_HOST, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    
    // Register texture with CUDA
    cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

void VoxelEngine::writeAndRenderTexture() {
    cudaArray *cudaArrayPtr;
    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&cudaArrayPtr, cudaResource, 0, 0);

    // Copy data to CUDA array
    uchar4 *devPtr;
    size_t pitch;
    cudaMallocPitch(&devPtr, &pitch, SCREEN_WIDTH_HOST * sizeof(uchar4), SCREEN_HEIGHT_HOST);

    Renderer::renderScreenCuda(octree, cameraAngle.x, cameraAngle.y, cameraPos.x, cameraPos.y, cameraPos.z, devPtr, 4096, 512);

    // Copy memory from CUDA device to OpenGL texture
    cudaMemcpy2DToArray(cudaArrayPtr, 0, 0, devPtr, pitch, SCREEN_WIDTH_HOST * sizeof(uchar4), SCREEN_HEIGHT_HOST, cudaMemcpyDeviceToDevice);
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

void VoxelEngine::handleInput(){
    SDL_PollEvent(&event_);

    switch (event_.type) {

       int x, y, z;

       case SDL_MOUSEMOTION:
           if (mouseControls) {
               handleCameraMovement(event_.motion.x, event_.motion.y, prevMouseX, prevMouseY);
           }
           break;

       case SDL_WINDOWEVENT:
           switch (event_.window.event) {
               case SDL_WINDOWEVENT_CLOSE:
                   cleanup();
                   return;

               default:
                   break;
               }
           break;

       case SDL_QUIT:
            cleanup();
            return;

       case SDL_KEYDOWN:
           switch (event_.key.keysym.sym) {
               case SDLK_z:
                   PLAYER_SPEED /= 2;
                   break;

               case SDLK_x:
                   PLAYER_SPEED *= 2;
                   break;

               case SDLK_c:
                   octree->textureRenderingEnabled = !octree->textureRenderingEnabled;
                   //doOldRendering = !doOldRendering;
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

void VoxelEngine::displayFrame(){
    writeAndRenderTexture();
    SDL_GL_SwapWindow(window);
}

void VoxelEngine::cleanup(){
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
}

void VoxelEngine::clearVoxels(){
    octree->clear();
}

void VoxelEngine::initBlockTextures() {
    int blockAmount = 4;

    cudaMallocManaged(&blockTextures, size_t(blockAmount * sizeof(BlockTexture*)));

    for (int i = 1; i < blockAmount + 1; i++) {

        string currPathStr = pathStr + "/res/textures/" + to_string(i);

        cudaMallocManaged(&blockTextures[i], sizeof(BlockTexture));
        new (blockTextures[i]) BlockTexture(778, 748, currPathStr + "/top.png", currPathStr + "/bottom.png", currPathStr + "/left.png", currPathStr + "/right.png", currPathStr + "/front.png", currPathStr + "/back.png");
    }

    createBlocksData<<<1,1>>>(blockTextures);

    cudaDeviceSynchronize();
}

void VoxelEngine::handleCameraMovement(int mouseX, int mouseY, int& prevMouseX, int& prevMouseY) {

    mouseX -= SCREEN_WIDTH_HOST / 2;
    mouseY -= SCREEN_HEIGHT_HOST / 2;

    cameraAngle.y -= (prevMouseX - mouseX) * MOUSE_SENSITIVITY;
    cameraAngle.x += (prevMouseY - mouseY) * MOUSE_SENSITIVITY;

    if (cameraAngle.x < -M_PI / 2.0) {
        cameraAngle.x = -M_PI / 2.0;
    }
    else if (cameraAngle.x > M_PI / 2.0) {
        cameraAngle.x = M_PI / 2.0;
    }

    prevMouseX = mouseX;
    prevMouseY = mouseY;
}