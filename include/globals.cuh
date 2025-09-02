#pragma once

#include <stdint.h>
#include <string>
#include <unistd.h>
#include <algorithm>

#include "point_light.cuh"
#include "material.cuh"

# define ERROR_PRINT() cudaDeviceSynchronize(); cudaError_t error = cudaGetLastError(); printf("CUDA error: %s\n", cudaGetErrorString(error));

constexpr int CUDA_STACK_SIZE = 11;

// viewing parameters
constexpr float FOCAL_LENGTH = 10000; //350 //1200 //4000
constexpr float SCALE_V = 1;

constexpr float MOUSE_SENSITIVITY = 0.004;

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

extern Vector3<> cameraPos;
extern Vector3<> cameraAngle;

extern PointLight pointLight;

extern Material mainMaterial;