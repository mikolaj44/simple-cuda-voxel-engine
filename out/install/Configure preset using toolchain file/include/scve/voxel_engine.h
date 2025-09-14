#pragma once


#include <string>
#include <cstdint>
#include <vector>

#include "../src/octree/octree.cuh"
#include "block_variant/block_variant_manager.cuh"

#include "cuda_math.cuh"
#include "material.cuh"
#include "light/point_light.cuh"

class Octree;

namespace scve {
/**
* @brief This is the main voxel engine class. It needs to be initialized with \ref init before use and cleaned up with \ref cleanup at the end.
Note: I use the term "host-allocated" in this documentation, which means CPU-side, so the RAM memory. Most resources are "device-allocated" internally by
the engine, so in the VRAM.
* */
class VoxelEngine {
public:

    /// @name Essential functions
    /// These are the main functions you will be using to interact with the engine. Some setters described in the next section are also very useful, so
    /// be sure to also check that out - you can see the examples section to get an idea of what you can do with all of these functions.
    /// @{

    /**
     * @details Initializes the engine - it can return a CUDA error if initialization fails, same with any other methods that return a cudaError_t, so it's recommended
     * to check these error codes, for example using \ref test. The created octree is centered by default (see \ref setOctreeCenter or \ref setOctreeMinPos to move it around)
     * Each block type for which you want to add a texture needs to be in a folder named from 1 to 127 and needs to contain the following files: \par
     * @code
     * 1/
     *  front.png
     *  back.png
     *  left.png
     *  right.png
     *  top.png
     *  bottom.png
     * @endcode
     * **Important:** these need to be .png files with 4 color channels, the resolution doesn't matter.
     * You can skip some numbers if you don't want to provide a texture for that block type. Note that if you enable texturing mode and a texture for a block id is not present, then it will be black.\par                    
     * @param windowWidth The width of the window
     * @param windowHeight The height of the window
     * @param texturesPath  The absolute path to a folder containing texture files. You can leave it as an empty string to not load any textures.
     * @param initialMaxOctreeDepth Initial maximum depth/level of the octree - this is further explained in \ref setOctreeMaxDepth. You can leave it as 0, but change to the value you want (from 0 to 10) before inserting voxels
     * @return CUDA error code
     * @throws std::runtime_error If called on an initialized engine, if the path to the textures folder does not exist or if one of the images in a folder like 1/ can't be loaded.
     * */
    static cudaError_t init(unsigned int windowWidth, unsigned int windowHeight, std::string texturesPath = "", unsigned int initialMaxOctreeDepth = 0);

    /**
    * @details An utility function to check error codes returned by functions like \ref init, \ref cleanup, \ref clearVoxels and others returning **cudaError_t**.
    * If the provided error is not **cudaSuccess**, then it prints the error message and calls **std::exit(EXIT_FAILURE)**.
    * @param error The CUDA error to be tested, for example from \ref init, \ref cleanup, or \ref setPointLights
    * */
    static void test(cudaError_t error);

    /**
    * @details Cleans up the resources used by the engine.
    * @return CUDA error code
    * */
    static cudaError_t cleanup();

    /**
     * @details Removes all voxels from the octree, so it becomes empty but stays the same size (depth/level doesn't change). Equivalent to calling
     * \ref setOctreeMaxDepth with \ref getOctreeMaxDepth as the argument
     * @return CUDA error code
     * */
    static cudaError_t clearVoxels();

    /**
    * @details Renders a single frame to the created window. \ref inputLoop also calls this by default after handling the keyboard input.
    * */
    static void displayFrame();

    /**
     * @details Handles keyboard input (if enabled, you can do so with \ref setKeyboardControlEnabled), handles mouse controls
     * (if enabled, you can do so with \ref setMouseControlEnabled), then calls \ref displayFrame (if **displayFrame** is set to **true**),
     * calls your function (**func**) if it's not **nullptr** after all of this - you can move the octree around here or do whatever you want.
     * @param func a function pointer to your custom function that gets called at the end
     * @param displayFrame will call \ref displayFrame if set to true, otherwise won't render a frame
     * */
    static void inputLoop(void (*func)() = nullptr, bool displayFrame = true);

    /**
    * @details Sets the max octree depth/level (one side has 2 ^ maxLevel voxels, max **depth** value is 10, min value is 0) and **clears the octree**.
    * It also allocates the memory which is 2 ^ (3 * maxLevel + 1) bytes, so the value of **10** is about 2 GB of VRAM used, while **9** is about 270 MB.
    * This basically allows you to insert more/less voxels and you need to set this **before inserting** them. \ref getMaxOctreeLevelByGPU is useful here.
    * @param depth The maximum octree depth that will be available (no more than 10)
    * @return CUDA error code
    * */
    static cudaError_t setOctreeMaxDepth(int depth);

    /**
     * @details Sets the point lights used for Phong lighting if \ref setPhongIlluminationEnabled is set to **true**.
     * @param pointLights The reference to a host-allocated vector containing the point lights
     * @return CUDA error code
     * */
    static cudaError_t setPointLights(const std::vector<PointLight>& pointLights);

    /**
    * @details Sets the point lights used for Phong lighting if \ref setPhongIlluminationEnabled is set to **true**.
    * @param pointLights The a host-allocated array containing the point lights
    * @param numLights The amount of point lights in the array
    * @return CUDA error code
    * */
    static cudaError_t setPointLights(PointLight* pointLights, unsigned int numLights);

    /**
    * @details Sets the materials for all block types (127 of them) using your own (**blockId**, **frameNumber**) to **material** mapping,
    * so you can decide the material a block type should have, maybe also dependent on the current frame number. If not called by the user,
    * the default mapping is as follows: \par
    * Material color: hue value for the color ofeach block type (in HSB, where saturation and brightness are set to max) \par
    * In detail: Material(hueToRGB((blockId) * 2.8346 / 360.0), 1.0, 0.0, 20.0)
    * @param func the device lambda that takes 2 parameters: **uint8_t blockId from 1 to 127** and **uint64_t frameNumber** and returns the **Material material** used by that block type.
    * */
    #ifdef __CUDACC__
        template<typename IdFrameToMaterialFunction>
        static void setMaterials(IdFrameToMaterialFunction func) {
            int numVariants = block_variant_manager::blockVariants->size();
            block_variant_manager::setBlocksVariantMaterialsKernel<<<1, numVariants>>>(func, frameNumber);
        }
    #endif

    /// @}

    /// @name Simple getters and setters
    /// These functions allow you to do things like moving the camera, the octree, enabling different texturing modes, setting the background light color
    /// and much more.
    /// @{

    /**
    * @return The camera position
    * */
    static Vector3<> getCameraPos();

    /**
    * @details Sets the camera position
    * @param pos The new camera position
    * */
    static void setCameraPos(Vector3<> pos);

    /**
     * @return The octree's minimal vertex position
     * */
    static Vector3<int> getOctreeMinPos();

    /**
    * @details Sets the "minimal" position of the octree, that is of the vertex that is in the minimal octant.
    * For example, an octree can start in (50, 50, 50) and span from that point 512 voxels to the max vertex: (562, 562, 562).
    * @param pos The new minimal vertex position
    * */
    static void setOctreeMinPos(Vector3<int> pos);

    /**
     * @return The octree's center position, (0, 0, 0) by default
     * */
    static Vector3<int> getOctreeCenter();

    /**
    * @details Sets the position of the center of the octree
    * @param pos The new center position
    * */
    static void setOctreeCenter(Vector3<int> pos);

    /**
     * @return The octree's maximum size (span, number of voxels on one side, calculated by 2 ^ level)
     * */
    static int getOctreeMaxSize();

    /**
    * @return The octree's maximum depth/level (sqrt(size))
    * */
    static cudaError_t getOctreeMaxDepth(int depth);
    
    /**
    * @return The camera angle (first two vector components, third is zero)
    * */
    static Vector3<> getCameraAngle2D();

    /**
    * @details Sets the camera angle
    * @param angle The angle vector - first component sets the vertical angle, second sets the horizontal, third is ignored
    * */
    static void setCameraAngle2D(Vector3<> angle);

    /**
    * @return A boolean value indicating whether the engine is initialized
    * */
    static bool getIsInitialized();

    /**
    * @details Enables/disables the texture rendering mode - if it's off, then the material rendering mode is active.
    * You can also toggle this using the "c" key in \ref inputLoop()
    * @param isEnabled The boolean value that enables/disables the texture rendering mode
    * */
    static void setTextureRenderingEnabled(bool isEnabled);

    /**
    * @return A boolean value indicating whether the texture rendering mode is enabled
    * */
    static bool getTextureRenderingEnabled();

    /**
    * @return A boolean value indicating whether the propagation of LODs during insertion is enabled
    * */
    static bool getPropagatingInsertLODsEnabled();

    /**
    * @details **Not fully working yet!** Enables/disables propagating LODs when voxels are being inserted. When enabled, it propagates information
    * about filled octree cells up the tree, which can speed up traversal as some threads stop early, but slow down the insertion process.
    * @param isEnabled The boolean value that enables/disables the propagation of LODs during insertion.
    * */
    static void setPropagatingInsertLODsEnabled(bool isEnabled);

    /**
    * @details A helper function that returns the max octree level that will fit in your free GPU VRAM (up to 10), can be useful in \ref init
    * @return The max octree level your free GPU VRAM supports
    * */
    static unsigned int getMaxOctreeLevelByGPU();

    /**
    * @return The width of the window in pixels
    * */
    static unsigned int getWindowWidth();

    /**
    * @return The height of the window in pixels
    * */
    static unsigned int getWindowHeight();

    /**
    * @return The current engine frame number
    * */
    static uint64_t getFrameNumber();

    /**
    * @return The camera moving speed
    * */
    static float getCameraMoveSpeed();

    /**
    * @details Sets the camera moving speed
    * @param speed The new moving speed of the camera
    * */
    static void setCameraMoveSpeed(float speed);

    /**
    * @return The camera turning speed
    * */
    static float getCameraTurnSpeed();

    /**
    * @details Sets the camera turning speed
    * @param speed The new turning speed of the camera
    * */
    static void setCameraTurnSpeed(float speed);

    /**
    * @return The mouse sensitivity that the engine uses
    * */
    static float getMouseSensitivity();

    /**
    * @details Sets the mouse sensitivity that the engine uses
    * @param speed The new he mouse sensitivity that will be used by the engine
    * */
    static void setMouseSensitivity(float sensitivity);

    /**
    * @return A boolean value indicating whether keyboard control is enabled in \ref inputLoop
    * */
    static bool getKeyboardControlEnabled();

    /**
    * @details Enables/disables the keyboard control is enabled in \ref inputLoop
    * @param isEnabled The boolean value that enables/disables the keyboard control
    * */
    static void setKeyboardControlEnabled(bool isEnabled);

    /**
    * @return A boolean value indicating whether mouse control is enabled in \ref inputLoop
    * */
    static bool getMouseControlEnabled();

    /**
    * @details Enables/disables the mouse control is enabled in \ref inputLoop
    * @param isEnabled The boolean value that enables/disables the mouse control
    * */
    static void setMouseControlEnabled(bool isEnabled);

    /**
    * @return A boolean value indicating whether Phong lighting is enabled
    * */
    static bool getPhongIlluminationEnabled();

    /**
    * @details **Not fully working yet!** Enables/disables the Phong illumination/lighting.
    * You can also toggle this using the "v" key in \ref inputLoop() 
    * @param isEnabled The boolean value that enables/disables the Phong lighting
    * */
    static void setPhongIlluminationEnabled(bool isEnabled);

    /**
    * @return A boolean value indicating whether displaying the information about free memory when allocating the octree or when using init or cleanup is enabled
    * */
   static bool getDisplayingMemoryInfoEnabled();

   /**
   * @details Enables/disables displaying the information about free memory when allocating the octree or when using init or cleanup is enabled
   * @param isEnabled The boolean value that enables/disables displaying the information
   * */
   static void setDisplayingMemoryInfoEnabled(bool isEnabled);

    /**
    * @return The color of the ambient light (default light when Phong lighting is enabled)
    * */
    static Vector3<> getAmbientLightColor();

    /**
    * @details Sets the ambient light color (default light when Phong lighting is enabled)
    * @param color The color vector
    * */
    static void setAmbientLightColor(Vector3<> color);

    /**
    * @return The intensity of the ambient light (default light when Phong lighting is enabled)
    * */
    static float getAmbientLightIntensity();

    /**
    * @details Sets the ambient light intensity (default light when Phong lighting is enabled)
    * @param intensity The intensity value
    * */
    static void setAmbientLightIntensity(float intensity);

    /**
    * @return The background color (visible when a ray doesn't hit any voxels)
    * */
    static Vector3<> getBackgroundColor();

    /**
    * @details Sets the background color (visible when a ray doesn't hit any voxels)
    * @param color The color vector
    * */
    static void setBackgroundColor(Vector3<> color);

    /**
    * @return The focal length value that manipulates FOV when rendering (10000 by default)
    * */
    static float getFocalLength();

    /**
    * @details Sets the focal length value that manipulates FOV when rendering (10000 by default)
    * @param focalLength The focal length value
    * */
    static void setFocalLength(float focalLength);

    /// @}
    
    /// @name Functions for interacting with voxels and pixels
    /// These allow you to insert/get voxel data and retrieve the pixels of the window (although that last feature is still experimental)
    /// @{

    /**
    * @details Inserts voxels from your flattened host id array (3D cube represented as 1D), id's equal to 0 are not inserted.
    * @param hostBlockIdArray The 1D host-allocated block id array
    * @param chunkWidth The width of the cube that will be inserted
    * @param startOffset The vector offset that the cube gets offset by, calculated from the minimal position/vertex of the octree (see \ref setOctreeMinPos for more details about that vertex)
    * @return CUDA error code (from cudaDeviceSynchronize)
    * */
    static cudaError_t insertVoxels(uint8_t* hostBlockIdArray, unsigned int chunkWidth, Vector3<int> startOffset = Vector3<int>(0, 0, 0));

    /**
    * @details Inserts voxels using your own (**x**, **y**, **z**, **frameNumber**) to **block id** mapping, so you can decide what block type should be at
    * a particular position, also taking the current frame number into consideration.
    * @param func the device lambda that takes 4 parameters: **int x**, **int y**, **int z** and **uint64_t frameNumber** and returns the **uint8_t blockType**.
    * @return CUDA error code (from cudaDeviceSynchronize)
    * */
    template<typename XYZFrameToIdFunction>
    static cudaError_t insertVoxels(XYZFrameToIdFunction func) {
        size_t totalVoxels = octree->getMaxSize();

        totalVoxels = totalVoxels * totalVoxels * totalVoxels;

        return octree->insertBlocksByXYZFrameFunction(func, frameNumber, isCalculatingInsertLODsEnabled, (totalVoxels + insertionBlockSize - 1) / insertionBlockSize, minv((size_t)insertionBlockSize, totalVoxels));
    }

    /**
    * @details Gets voxels to an array that will be host-allocated (3D cube represented as 1D). **Note:** Remember to call delete[] on your array later as
    * this function allocates host-side resources!
    * @param hostBlockIdArrayPtr Address of the pointer where the 1D host-allocated block id array will be created
    * @param chunkWidth The width of the cube to be retrieved
    * @param startOffset The vector offset that the cube is offset by, calculated from the minimal position/vertex of the octree (see \ref setOctreeMinPos for more details about that vertex)
    * @return CUDA error code (from cudaDeviceSynchronize)
    * */
    static cudaError_t getVoxels(uint8_t** hostBlockIdArrayPtr, unsigned int chunkWidth, Vector3<int> startOffset = Vector3<int>(0, 0, 0));

    /**
    * @details **Experimental feature! I have not yet verified if the dimensions match this description correctly** 
    * You will need to convert the pixels from uchar4 yourself as of now, they should be in (r, g, b, a) format.
    * Gets pixels from the window (originally a 2D array of **windowHeight** rows and **windowWidth** columns) to a 1D array
    * of **windowWidth** * **windowHeight** elements. You can get these dimensions with \ref getWindowHeight and \ref getWindowWidth.
    * **Note:** Remember to call delete[] on your array later as this function allocates host-side resources!
    * @param hostPixelArrayPtr Address of the pointer where the 1D host-allocated block id array will be created
    * @return CUDA error code (from cudaDeviceSynchronize)
    * */
    static cudaError_t getPixels(uchar4** hostPixelArrayPtr);

    /// @}

private:
    static bool isInitialized;

    static bool isTextureRenderingEnabled;

    static bool isCalculatingInsertLODsEnabled;

    static bool isKeyboardControlEnabled;

    static bool isMouseControlEnabled;

    static bool isPhongIlluminationEnabled;

    static bool isDisplayingMemoryInfoEnabled;

    static unsigned int windowWidth, windowHeight;

    static const unsigned int insertionBlockSize = 512;

    static const unsigned int renderThreadsPerBlock = 600;
    static unsigned int renderBlocksPerGrid;

    static int prevMouseX;
    static int prevMouseY;

    static Octree* octree;

    static uint64_t frameNumber;

    static dim3 maxGridSize;
    static dim3 blockSize;

    static Vector3<> cameraPos;
    static Vector3<> cameraAngle;

    static float cameraSpeed;
    static float cameraTurnSpeed;
    static float mouseSensitivity;

    static void initBlockTextures();

    static void handleCameraMovement(int mouseX, int mouseY);

    static cudaError_t initLights();
};

}