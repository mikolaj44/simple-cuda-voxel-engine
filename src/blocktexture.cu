#include <iostream>
#include "blocktexture.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

BlockTexture::BlockTexture(int w, int h, string top, string bottom, string left, string right, string front, string back) {

	width = w;
	height = h;

	int channelsInImg = 4;

	size_t imgSize = size_t(width * height * channels);

	// temporarily load images on host side

	unsigned char* topImageHost = stbi_load(top.c_str(), &width, &height, &channelsInImg, channels);
	unsigned char* bottomImageHost = stbi_load(bottom.c_str(), &width, &height, &channelsInImg, channels);
	unsigned char* leftImageHost = stbi_load(left.c_str(), &width, &height, &channelsInImg, channels);
	unsigned char* rightImageHost = stbi_load(right.c_str(), &width, &height, &channelsInImg, channels);
	unsigned char* frontImageHost = stbi_load(front.c_str(), &width, &height, &channelsInImg, channels);
	unsigned char* backImageHost = stbi_load(back.c_str(), &width, &height, &channelsInImg, channels);

	// malloc for the images to be stored on device side

	cudaMallocManaged(&topImage, imgSize);
	cudaMallocManaged(&bottomImage, imgSize);
	cudaMallocManaged(&leftImage, imgSize);
	cudaMallocManaged(&rightImage, imgSize);
	cudaMallocManaged(&frontImage, imgSize);
	cudaMallocManaged(&backImage, imgSize);

	// copy data to gpu

	memcpy(topImage, topImageHost, imgSize);
	memcpy(bottomImage, bottomImageHost, imgSize);
	memcpy(leftImage, leftImageHost, imgSize);
	memcpy(rightImage, rightImageHost, imgSize);
	memcpy(frontImage, frontImageHost, imgSize);
	memcpy(backImage, backImageHost, imgSize);

	// free temp host data

	stbi_image_free(topImageHost);
	stbi_image_free(bottomImageHost);
	stbi_image_free(leftImageHost);
	stbi_image_free(rightImageHost);
	stbi_image_free(frontImageHost);
	stbi_image_free(backImageHost);
}