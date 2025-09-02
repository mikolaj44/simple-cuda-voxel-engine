#define SDL_MAIN_HANDLED

#include <GL/glew.h>
#include <SDL.h>

extern GLuint textureID;
extern cudaGraphicsResource *cudaResource;

extern SDL_Window* window;
extern SDL_Renderer* renderer;
extern SDL_Texture* texture;

extern SDL_Surface* textSurface;
extern SDL_Texture* textTexture;

namespace cuda_renderer_utils {
    cudaError_t initSDL(int windowWidth, int windowHeight);

    cudaError_t cleanupSDL();
}