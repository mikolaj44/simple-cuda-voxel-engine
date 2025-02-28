#include "blocks_data.cuh"

__device__ BlockVariant** blockVariants = nullptr;
__constant__ int blocksAmount = 10;

__constant__ float epsilon = 0.0001;

__device__ void setPixelById(int sX, int sY, int blockX, int blockY, int blockZ, float x, float y, float z, unsigned char blockId, unsigned char* pixels) {

    if (blockId >= blocksAmount) {
        return;
    }

    int imgWidth = blockVariants[blockId]->texture->width;
    int imgHeight = blockVariants[blockId]->texture->height;
    int imgChannels = blockVariants[blockId]->texture->channels;
    int imgX = 0, imgY = 0;

    // check which side of the block we are on

    if (equals(y, (float)blockY, epsilon)) { // top
        imgX = (int)(absv(x - (int)x) * imgWidth);
        imgY = (int)(absv(z - (int)z) * imgHeight);
        setPixel(pixels, sX, sY, blockVariants[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels], blockVariants[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels + 1], blockVariants[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels + 2], 255);
        return;
    }
    else if (equals(y, (float)blockY + 1.0, epsilon)) { // bottom
        imgX = (int)(absv(x - (int)x) * imgWidth);
        imgY = (int)(absv(z - (int)z) * imgHeight);
        setPixel(pixels, sX, sY, blockVariants[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels], blockVariants[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels + 1], blockVariants[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels + 2], 255);
        return;
    }
    else if (equals(x, (float)blockX, epsilon)) { // left
        imgX = (int)(absv(z - (int)z) * imgWidth);
        imgY = (int)(absv(y - (int)y) * imgHeight);
        setPixel(pixels, sX, sY, blockVariants[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels], blockVariants[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels + 1], blockVariants[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels + 2], 255);
        return;
    }
    else if (equals(x, (float)blockX + 1.0, epsilon)) { // right
        imgX = (int)(absv(z - (int)z) * imgWidth);
        imgY = (int)(absv(y - (int)y) * imgHeight);
        setPixel(pixels, sX, sY, blockVariants[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels], blockVariants[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels + 1], blockVariants[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels + 2], 255);
        return;
    }
    else if (equals(z, (float)blockZ, epsilon)) { // front
        imgX = (int)(absv(x - (int)x) * imgWidth);
        imgY = (int)(absv(y - (int)y) * imgHeight);
        setPixel(pixels, sX, sY, blockVariants[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels], blockVariants[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels + 1], blockVariants[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels + 2], 255);
        return;
    }
    else if (equals(z, (float)blockZ + 1.0, epsilon)) { // back
        imgX = (int)(absv(x - (int)x) * imgWidth);
        imgY = (int)(absv(y - (int)y) * imgHeight);
        setPixel(pixels, sX, sY, blockVariants[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels], blockVariants[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels + 1], blockVariants[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels + 2], 255);
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

    cudaMalloc(&blockVariants, sizeof(BlockVariant*) * blocksAmount);

    for (int i = 0; i < blocksAmount; i++) {
        blockVariants[i] = new BlockVariant(SOLID, textures[i]);
    }
}