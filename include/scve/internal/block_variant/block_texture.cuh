#pragma once

#include <string>

namespace scve {

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
	BlockTexture() = default;

    BlockTexture(const BlockTexture&) = delete;
	
    BlockTexture(BlockTexture&&) = delete;

	~BlockTexture() = delete;

	__host__ cudaError_t init(int channelsInImg, std::string* paths);

	__host__ cudaError_t cleanup();

	BlockTexture& operator=(const BlockTexture&) = delete;

    BlockTexture& operator=(BlockTexture&&) = delete;

	__host__ __device__ int getChannels() const;

	__host__ __device__ int getChannelsInImg() const;

	__host__ __device__ int getWidth(ImagePosition position) const;

	__host__ __device__ int getHeight(ImagePosition position) const;

	__host__ __device__ unsigned char* getImage(ImagePosition position) const;
private:
	int channels = 3;
	int channelsInImg;

	int widths[6]  = {0, 0, 0, 0, 0, 0};
	int heights[6] = {0, 0, 0, 0, 0, 0};

	unsigned char* images[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
};

}