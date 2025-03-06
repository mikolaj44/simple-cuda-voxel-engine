#pragma once
#include <stdint.h>
#include <string>
#include <unistd.h>
#include <algorithm>

//#include "D:\threadpool\include\BS_thread_pool.hpp"

// files
extern std::string pathStr;

// screen size
constexpr int SCREEN_WIDTH = 1920;
constexpr int SCREEN_HEIGHT = 1080;

// viewing parameters
constexpr float FOCAL_LENGTH = 4000; //350 //1200
constexpr int SCALE_V = 1;
extern float halfVerFOV, halfHorFOV;
constexpr float MOUSE_SENSITIVITY = 0.004;

// rendering parameters
constexpr int RENDER_DISTANCE_CHUNKS = 8; // (in chunks)

// octree memory parameters
constexpr size_t PREALLOCATE_MB_AMOUNT = 1000;
constexpr size_t MEMORY_LIMIT_MB = 300;
constexpr size_t NODE_MAP_CAPACITY = 10'000'000; // 300 million is about 5 GB of VRAM used (4 byte Node)

// threading parameters
//constexpr unsigned int MAX_THREADS_AMOUNT = 1;

// general numeric parameters
constexpr float EPSILON = 0.00001;

// chunk parameters (block amount)
constexpr int CHUNK_W = 16;

// world generation parameters

constexpr float smoothing = 50;
constexpr float amplify = 50;

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

class Vector3 {

public:

    float x = 0, y = 0, z = 0;

    Vector3();

    Vector3(float x_, float y_, float z_);

    void scale(float val);
};

extern Vector3 cameraPos;
extern Vector3 cameraAngle;

extern Vector3 unitCubeCoords[8];
//extern BS::thread_pool threadPool;