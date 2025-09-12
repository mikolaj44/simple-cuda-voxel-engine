#include "renderer/cuda_renderer_utils.cuh"

#include <cuda_gl_interop.h>

namespace scve::cuda_renderer_utils {
    GLuint textureID;
    cudaGraphicsResource* cudaResource;

    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;

    SDL_Surface* textSurface;
    SDL_Texture* textTexture;

    namespace {
        cudaError_t createCUDATexture(int windowWidth, int windowHeight) {
            glGenTextures(1, &textureID);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            
            return cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        }
    }

    cudaError_t initSDL(int windowWidth, int windowHeight) {
        SDL_Init(SDL_INIT_VIDEO);

        window = SDL_CreateWindow("voxel engine", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, windowWidth, windowHeight, SDL_WINDOW_OPENGL);
    
        SDL_GLContext glContext = SDL_GL_CreateContext(window);
        glewInit();
    
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, windowWidth, windowHeight);
    
        SDL_SetRelativeMouseMode(SDL_TRUE);

        return createCUDATexture(windowWidth, windowHeight);
    }

    cudaError_t cleanupSDL() {
        cudaError_t error = cudaGraphicsUnregisterResource(cudaResource);

        if(error != cudaSuccess) {
            return error;
        }

        glDeleteTextures(1, &textureID);
    
        SDL_FreeSurface(textSurface);
        SDL_DestroyTexture(textTexture);
        // TTF_Quit();
    
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();

        return error;
    }
}