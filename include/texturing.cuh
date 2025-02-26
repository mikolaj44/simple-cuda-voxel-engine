#pragma once
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

__device__ unsigned char* BlockTypeToColor(unsigned char type);