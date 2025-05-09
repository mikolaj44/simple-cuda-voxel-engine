#include "blocks_data.cuh"
#include "cuda_math.cuh"
#include "renderer.cuh"

__device__ BlockVariant** blockVariants = nullptr;
__constant__ int blocksAmount = 4;

__constant__ float epsilon = 0.001;

// https://stackoverflow.com/questions/61277046/convert-just-a-hue-into-rgb
__device__ void hueToRGB(float hue, int& r, int& g, int&b){
    float kr = remainderf(5 + hue * 6, 6);
    float kg = remainderf(3 + hue * 6, 6);
    float kb = remainderf(1 + hue * 6, 6);

    r = (1 - maxv(minv(minv(kr, 4-kr), 1.0f), 0.0f)) * 255;
    g = (1 - maxv(minv(minv(kg, 4-kg), 1.0f), 0.0f)) * 255;
    b = (1 - maxv(minv(minv(kb, 4-kb), 1.0f), 0.0f)) * 255;
}

__device__ void getPhongIllumination(Vector3 pos, Vector3 cameraPos, Vector3 normal, Material material, PointLight light, int& r, int& g, int&b){
    material.color = Vector3(r, g, b);

    //if(pos.x > 99)
    //printf("%f %f %f\n", pos.x, pos.y, pos.z);

    r = 0;
    g = 0;
    b = 0;

	Vector3 Ln = norm(Vector3(light.pos.x - pos.x, light.pos.y - pos.y, light.pos.z - pos.z));

	if (dot(normal, Ln) < 0) {
		return;
	}

    // //printf("%d %d %d\n", r, g, b);

	Vector3 h  = norm(Vector3(cameraPos.x - pos.x, cameraPos.y - pos.y, cameraPos.z - pos.z));

    //h = mul(h, -1);

	Vector3 dh = norm(sub(mul(normal, 2 * dot(Ln, normal)), Ln));

    //dh = mul(dh, -1);

    // if(dot(h, dh) < 0){
    //     return;
    // }

    //dh = mul(dh, -1);

	Vector3 diffuseVector = Vector3(material.diffuse, material.diffuse, material.diffuse);
	Vector3 specularVector = Vector3(material.specular, material.specular, material.specular);

	Vector3 lighting = vmul(light.color, mul(diffuseVector, dot(normal, Ln)));

    // if(isnan( pow(dot(h, dh), (int)material.specularExponent) )){
    //     printf("%f %f\n", dot(h, dh), material.specularExponent);
    // }

    //Vector3 lighting = vmul(light.color, add(mul(diffuseVector, dot(normal, Ln)), mul(specularVector, pow(dot(dh, h), (int)material.specularExponent))));

	if (lighting.x > 255)
		lighting.x = 255;
	if (lighting.y > 255)
		lighting.y = 255;
	if (lighting.z > 255)
		lighting.z = 255;

	lighting = div(lighting, 255.0);
	lighting = vmul(lighting, material.color);

    r = (int)lighting.x;
    g = (int)lighting.y;
    b = (int)lighting.z;

    // r = (int)material.color.x;
    // g = (int)material.color.y;
    // g = (int)material.color.z;
}

__device__ void setPixelById(int sX, int sY, int blockX, int blockY, int blockZ, float x, float y, float z, unsigned char blockId, uchar4* pixels, Vector3 cameraPos, PointLight light, bool textureRenderingEnabled) {  
    if (textureRenderingEnabled && blockId >= blocksAmount) {
        return;
    }

    int imgWidth, imgHeight, imgChannels;

    if(textureRenderingEnabled){
        imgWidth = blockVariants[blockId]->texture->width;
        imgHeight = blockVariants[blockId]->texture->height;
        imgChannels = blockVariants[blockId]->texture->channels;
    }

    int imgX = 0, imgY = 0;

    int r, g, b;
    Vector3 normal;

    // check which side of the block we are on

    if (equals(y, (float)blockY, epsilon)) { // top
        if(textureRenderingEnabled){
            imgX = (int)(absv(x - floor(x)) * imgWidth);
            imgY = imgHeight - (int)(absv(z - floor(z)) * imgHeight);

            r = blockVariants[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels];
            g = blockVariants[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels + 1];
            b = blockVariants[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels + 2];
        }

        normal = Vector3(0, -1, 0);
    }
    else if (equals(y, (float)blockY + 1.0, epsilon)) { // bottom
        if(textureRenderingEnabled){
            imgX = (int)(absv(x - floor(x)) * imgWidth);
            imgY = (int)(absv(z - floor(z)) * imgHeight);

            r = blockVariants[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels];
            g = blockVariants[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels + 1];
            b = blockVariants[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels + 2];
        }

        normal = Vector3(0, 1, 0);
    }
    else if (equals(x, (float)blockX, epsilon)) { // left
        if(textureRenderingEnabled){
            imgX = imgWidth - (int)(absv(z - floor(x)) * imgWidth);
            imgY = (int)(absv(y - floor(y)) * imgHeight);

            r = blockVariants[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels];
            g = blockVariants[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels + 1];
            b = blockVariants[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels + 2];
        }

        normal = Vector3(-1, 0, 0);
    }
    else if (equals(x, (float)blockX + 1.0, epsilon)) { // right
        if(textureRenderingEnabled){
            imgX = imgWidth - (int)(absv(z - ceil(x)) * imgWidth);
            imgY = (int)(absv(y - floor(y)) * imgHeight);

            r = blockVariants[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels];
            g = blockVariants[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels + 1];
            b = blockVariants[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels + 2];
        }

        normal = Vector3(1, 0, 0);
    }
    else if (equals(z, (float)blockZ, epsilon)) { // front
        if(textureRenderingEnabled){
            imgX = (int)(absv(x - floor(x)) * imgWidth);
            imgY = (int)(absv(y - floor(y)) * imgHeight);

            r = blockVariants[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels];
            g = blockVariants[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels + 1];
            b = blockVariants[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels + 2];
        }

        normal = Vector3(0, 0, -1);
    }
    else if (equals(z, (float)blockZ + 1.0, epsilon)) { // back
        if(textureRenderingEnabled){
            imgX = imgWidth - (int)(absv(x - floor(x)) * imgWidth);
            imgY = (int)(absv(y - floor(y)) * imgHeight);

            r = blockVariants[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels];
            g = blockVariants[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels + 1];
            b = blockVariants[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels + 2];
        }

        normal = Vector3(0, 0, 1);
    }
    else {
        return;
    }
    
    if(!textureRenderingEnabled) {
        //printf("%f\n", float(blockId));
        hueToRGB(float(blockId) * 2.8125 / 360.0, r, g, b);
    }

    getPhongIllumination(Vector3(x, y, z), cameraPos, normal, blockVariants[blockId]->material, light, r, g, b);

    setPixel(pixels, sX, sY, r, g, b, 255);

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
        blockVariants[i] = new BlockVariant(Material(Vector3(255,255,255), 1, 0, 20), textures[i]);
    }
}