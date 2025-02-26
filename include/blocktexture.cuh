#pragma once
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

class BlockTexture {

public:
	
	int width = 16, height = 16, channels = 3;

	unsigned char* topImage = nullptr;
	unsigned char* bottomImage = nullptr;
	unsigned char* leftImage = nullptr;
	unsigned char* rightImage = nullptr;
	unsigned char* frontImage = nullptr;
	unsigned char* backImage = nullptr;

	BlockTexture(){};

	BlockTexture(int w, int h, string top, string bottom, string left, string right, string front, string back);

	BlockTexture(int w, int h, string path) : BlockTexture(w, h, path, path, path, path, path, path) {};
};