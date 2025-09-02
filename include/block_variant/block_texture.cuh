#pragma once

#include <string>

enum ImagePosition {
	TOP,
	BOTTOM,
	LEFT,
	RIGHT,
	FRONT,
	BACK
};

class BlockTexture {
public:
	__host__ BlockTexture(int channelsInImg, std::string* paths);

	__host__ __device__ int getChannels() const;

	__host__ __device__ int getChannelsInImg() const;

	__host__ __device__ int getWidth(ImagePosition position) const;

	__host__ __device__ int getHeight(ImagePosition position) const;

	__host__ __device__ unsigned char* getImage(ImagePosition position) const;
private:
	int channels = 3;
	int channelsInImg;

	int widths[6] = {0, 0, 0, 0, 0, 0};
	int heights[6] = {0, 0, 0, 0, 0, 0};

	unsigned char* images[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
};