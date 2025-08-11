#include "blocks_data.cuh"
#include "cuda_math.cuh"
#include "cuda_renderer.cuh"

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

__device__ void getPhongIllumination(Vector3<> pos, Vector3<> cameraPos, Vector3<> normal, Material material, PointLight light, int& r, int& g, int&b){
    // material.color = Vector3<>(r, g, b);

    // //if(pos.x > 99)
    // //printf("%f %f %f\n", pos.x, pos.y, pos.z);

    // r = 0;
    // g = 0;
    // b = 0;

	// Vector3<> Ln = norm(Vector3<>(light.pos.x - pos.x, light.pos.y - pos.y, light.pos.z - pos.z));

	// if (dot(normal, Ln) < 0) {
	// 	return;
	// }

    // // //printf("%d %d %d\n", r, g, b);

	// Vector3 h  = norm(Vector3(cameraPos.x - pos.x, cameraPos.y - pos.y, cameraPos.z - pos.z));

    // //h = mul(h, -1);

	// Vector3<> dh = norm(sub(mul(normal, 2 * dot(Ln, normal)), Ln));

    // //dh = mul(dh, -1);

    // // if(dot(h, dh) < 0){
    // //     return;
    // // }

    // //dh = mul(dh, -1);

	// Vector3<> diffuseVector = Vector3<>(material.diffuse, material.diffuse, material.diffuse);
	// Vector3<> specularVector = Vector3<>(material.specular, material.specular, material.specular);

	// Vector3<> lighting = vmul(light.color, mul(diffuseVector, dot(normal, Ln)));

    // // if(isnan( pow(dot(h, dh), (int)material.specularExponent) )){
    // //     printf("%f %f\n", dot(h, dh), material.specularExponent);
    // // }

    // //Vector3 lighting = vmul(light.color, add(mul(diffuseVector, dot(normal, Ln)), mul(specularVector, pow(dot(dh, h), (int)material.specularExponent))));

	// if (lighting.x > 255)
	// 	lighting.x = 255;
	// if (lighting.y > 255)
	// 	lighting.y = 255;
	// if (lighting.z > 255)
	// 	lighting.z = 255;

	// lighting = div(lighting, 255.0);
	// lighting = vmul(lighting, material.color);

    // r = (int)lighting.x;
    // g = (int)lighting.y;
    // b = (int)lighting.z;

    // r = (int)material.color.x;
    // g = (int)material.color.y;
    // g = (int)material.color.z;
}

__global__ void createBlocksData(BlockTexture** textures) {
    cudaMalloc(&blockVariants, sizeof(BlockVariant*) * blocksAmount);

    for (int i = 0; i < blocksAmount; i++) {
        blockVariants[i] = new BlockVariant(Material(Vector3<>(255,255,255), 1, 0, 20), textures[i]);
    }
}