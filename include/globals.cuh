#pragma once
#include <stdint.h>
#include <string>
#include <unistd.h>
#include <algorithm>

#include "pointlight.cuh"
#include "material.cuh"

//#include "D:\threadpool\include\BS_thread_pool.hpp"

// files
extern std::string pathStr;

// screen size
extern __constant__ unsigned int SCREEN_WIDTH_DEVICE;
extern __constant__ unsigned int SCREEN_HEIGHT_DEVICE;

extern unsigned int SCREEN_WIDTH_HOST;
extern unsigned int SCREEN_HEIGHT_HOST;


// viewing parameters
constexpr float FOCAL_LENGTH = 10000; //350 //1200 //4000
constexpr float SCALE_V = 1;
constexpr float MOUSE_SENSITIVITY = 0.004;

// rendering parameters
constexpr int RENDER_DISTANCE_CHUNKS = 32; // (in chunks)

// octree memory parameters
constexpr size_t PREALLOCATE_MB_AMOUNT = 5000;
constexpr int CUDA_STACK_SIZE = 40;

// general numeric parameters
constexpr float EPSILON = 0.00001;

// chunk parameters (block amount)
constexpr int CHUNK_W = 16;

// player parameters
constexpr float PLAYER_HEIGHT = 1.75;
constexpr float CAMERA_HEIGHT = 2;

extern float PLAYER_SPEED; // 1
extern float PLAYER_SPEED_FLYING ; // 0.2
extern float PLAYER_TURN_Y_SPEED;

// ui parameters
extern bool mouseControls;
extern bool doGravity;
extern bool showFps;
extern bool doOldRendering;
extern bool generateNewChunks;
extern bool showBorder;

extern int shift;
extern int shiftX;
extern int shiftY;
extern int shiftZ;

std::string getPathStr();

extern Vector3 cameraPos;
extern Vector3 cameraAngle;

extern PointLight pointLight;

extern Material mainMaterial;

extern Vector3 unitCubeCoords[8];
//extern BS::thread_pool threadPool;