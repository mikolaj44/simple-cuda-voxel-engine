#include "blocks_data.cuh"

__device__ BlockVariant** blocks = nullptr;
__constant__ int blocksAmount = 10;

__constant__ float epsilon = 0.0001;

__device__ void setPixelById(int sX, int sY, int blockX, int blockY, int blockZ, float x, float y, float z, unsigned char blockId, unsigned char* pixels) {

    if (blockId >= blocksAmount) {
        return;
    }

    int imgWidth = blocks[blockId]->texture->width;
    int imgHeight = blocks[blockId]->texture->height;
    int imgChannels = blocks[blockId]->texture->channels;
    int imgX = 0, imgY = 0;

    // check which side of the block we are on

    if (equals(y, (float)blockY, epsilon)) { // top
        imgX = (int)(absv(x - (int)x) * imgWidth);
        imgY = (int)(absv(z - (int)z) * imgHeight);
        SetPixel(sX, sY, blocks[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels], blocks[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels + 1], blocks[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels + 2], 255, pixels);
        return;
    }
    else if (equals(y, (float)blockY + 1.0, epsilon)) { // bottom
        imgX = (int)(absv(x - (int)x) * imgWidth);
        imgY = (int)(absv(z - (int)z) * imgHeight);
        SetPixel(sX, sY, blocks[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels], blocks[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels + 1], blocks[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels + 2], 255, pixels);
        return;
    }
    else if (equals(x, (float)blockX, epsilon)) { // left
        imgX = (int)(absv(z - (int)z) * imgWidth);
        imgY = (int)(absv(y - (int)y) * imgHeight);
        SetPixel(sX, sY, blocks[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels], blocks[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels + 1], blocks[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels + 2], 255, pixels);
        return;
    }
    else if (equals(x, (float)blockX + 1.0, epsilon)) { // right
        imgX = (int)(absv(z - (int)z) * imgWidth);
        imgY = (int)(absv(y - (int)y) * imgHeight);
        SetPixel(sX, sY, blocks[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels], blocks[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels + 1], blocks[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels + 2], 255, pixels);
        return;
    }
    else if (equals(z, (float)blockZ, epsilon)) { // front
        imgX = (int)(absv(x - (int)x) * imgWidth);
        imgY = (int)(absv(y - (int)y) * imgHeight);
        SetPixel(sX, sY, blocks[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels], blocks[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels + 1], blocks[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels + 2], 255, pixels);
        return;
    }
    else if (equals(z, (float)blockZ + 1.0, epsilon)) { // back
        imgX = (int)(absv(x - (int)x) * imgWidth);
        imgY = (int)(absv(y - (int)y) * imgHeight);
        SetPixel(sX, sY, blocks[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels], blocks[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels + 1], blocks[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels + 2], 255, pixels);
        return;
    }

    //else if (equals(x, blockX, epsilon)) { // 
    //    side = 'k';
    //}

    //printf("%f %f %f %c\n", x, y, z, side);

    //printf("%d %d \n", imageX, imageY);

    //blocks[blockId]->texture->topImage[imageY * imgHeight + imageX], blocks[blockId]->texture->topImage[imageY * imgHeight + imageX], blocks[blockId]->texture->topImage[imageY * imgHeight + imageX]
}

__global__ void createBlocksData(BlockTexture** textures) {

    cudaMalloc(&blocks, sizeof(BlockVariant*) * blocksAmount);

    for (int i = 0; i < blocksAmount; i++) {
        blocks[i] = new BlockVariant(SOLID, textures[i]);
    }
}