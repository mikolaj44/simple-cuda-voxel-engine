#define SDL_MAIN_HANDLED

#include <GL/glew.h>

#ifdef _WIN32
    #include <Windows.h> 
    #include <GL/gl.h>
#else
    #include <GL/gl.h>
#endif

#include <SDL.h>

namespace scve {
namespace cuda_renderer_utils {
    extern GLuint textureID;
    extern cudaGraphicsResource *cudaResource;

    extern SDL_Window* window;
    extern SDL_Renderer* renderer;
    extern SDL_Texture* texture;

    extern SDL_Surface* textSurface;
    extern SDL_Texture* textTexture;

    cudaError_t initSDL(int windowWidth, int windowHeight);

    cudaError_t cleanupSDL();
}
}