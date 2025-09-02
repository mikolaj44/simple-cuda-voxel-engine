#define STB_IMAGE_IMPLEMENTATION

#include "block_variant/block_texture.cuh"
#include "stb_image.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

std::string imagePositionName[6] = {"top", "bottom", "left", "right", "front", "back"};

// https://stackoverflow.com/questions/10111784/get-image-resolution-from-image-file
bool getPngImageResolution(std::string imagePath, int& imageWidth, int& imageHeight) {
	FILE *f = fopen(imagePath.c_str(), "rb");

	if(f == 0) {
		return false;
	}

    fseek(f, 0, SEEK_END); 

    long len=ftell(f); 

    fseek(f, 0, SEEK_SET); 

    if(len < 24) {
    	fclose(f); 
        return false;
    }

    unsigned char buf[24];
	
	fread(buf, 1, 24, f);

	fclose(f);

	if(buf[0] == 0x89 && buf[1] == 'P' && buf[2] == 'N' && buf[3] == 'G' && buf[4] == 0x0D && buf[5] == 0x0A && buf[6] == 0x1A && buf[7] == 0x0A && buf[12] == 'I' && buf[13] == 'H' && buf[14] == 'D' && buf[15] == 'R') {
		imageWidth  = (buf[16] << 24) + (buf[17] << 16) + (buf[18] << 8) + (buf[19] << 0);
		imageHeight = (buf[20] << 24) + (buf[21] << 16) + (buf[22] << 8) + (buf[23] << 0);
		return true;
	}

	return false;
}

__host__ cudaError_t BlockTexture::create(int channelsInImg_, std::string* paths) {
	cudaError_t error = cudaSuccess;

	channelsInImg = channelsInImg_;

	unsigned char* hostImages[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

	for(int i = 0; i < 6; i++) {
		bool imageLoaded = getPngImageResolution(paths[i], widths[i], heights[i]);

		if(!imageLoaded) {
			throw "Could not load the " + imagePositionName[i] + "side image. Make sure that \"" + imagePositionName[i] + ".png\" is present. Also verify that all images have " + std::to_string(channelsInImg_) + " channels.";
		}

		hostImages[i] = stbi_load(paths[0].c_str(), &widths[i], &heights[i], &channelsInImg, channels);

		size_t imgSize = size_t(widths[i] * heights[i] * channels);

		error = cudaMallocManaged(&images[i], imgSize);

		if(error != cudaSuccess) {
			return error;
		}

		error = cudaMemcpy(images[i], hostImages[i], imgSize, cudaMemcpyHostToDevice);

		if(error != cudaSuccess) {
			return error;
		}

		stbi_image_free(hostImages[i]);
	}

	return error;
}

__host__ __device__ int BlockTexture::getChannels() const {
	return channels;
}

__host__ __device__ int BlockTexture::getChannelsInImg() const {
	return channelsInImg;
}

__host__ __device__ int BlockTexture::getWidth(ImagePosition position) const {
	return widths[position];
}

__host__ __device__ int BlockTexture::getHeight(ImagePosition position) const {
	return heights[position];
}

__host__ __device__ unsigned char* BlockTexture::getImage(ImagePosition position) const {
	return images[position];
}