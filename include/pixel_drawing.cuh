#pragma once
#include <vector>
#include "globals.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

__device__ __host__ void setPixel(unsigned char* pixels, int x, int y, int r, int g, int b, int a = 255);

void plotLineHigh(unsigned char* pixels, int x1, int y1, int x2, int y2, int r, int g, int b, int a = 255);

void plotLineLow(unsigned char* pixels, int x1, int y1, int x2, int y2, int r, int g, int b, int a = 255);

void drawLine(unsigned char* pixels, int x1, int y1, int x2, int y2, int r, int g, int b, int a = 255);

float* _3d2dProjection(float x_, float y_, float z_);

float angleNormalize(float a);

vector<pair<int, int>> plotLineHighPoints(int x1, int y1, int x2, int y2);

vector<pair<int, int>> plotLineLowPoints(int x1, int y1, int x2, int y2);

vector<pair<int, int>> LinePoints(int x1, int y1, int x2, int y2);