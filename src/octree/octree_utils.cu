#include "octree/octree_utils.cuh"

#include "globals.cuh"
#include "cuda_math.cuh"
#include "block_variant/block_variant_manager.cuh"
#include "point_light.cuh"

namespace octree_utils {
    __device__ unsigned int nodeLevel(uint32_t mortonCode, unsigned int octreeLevel){
        int depth = 0;

        for (uint32_t code = mortonCode; code != 1; code >>= 3, depth++);

        return octreeLevel - depth;
    }

    __device__ unsigned int nodeSize(uint32_t mortonCode, unsigned int octreeLevel){
        return 1 << nodeLevel(mortonCode, octreeLevel);
    }

    __device__ Vector3<> getBlockHitPos(Vector3<int> blockPos, Vector3<> rayOrigin, Vector3<> rayDirection) {
        if (rayDirection.x == 0 || rayDirection.y == 0 || rayDirection.z == 0) {
            return Vector3<>();
        }
        
        float tmin =  minv(static_cast<float>(static_cast<float>(blockPos.x) - rayOrigin.x) / rayDirection.x, static_cast<float>(static_cast<float>(blockPos.x) + 1.0 - rayOrigin.x) / rayDirection.x);
        float tymin = minv(static_cast<float>(static_cast<float>(blockPos.y) - rayOrigin.y) / rayDirection.y, static_cast<float>(static_cast<float>(blockPos.y) + 1.0 - rayOrigin.y) / rayDirection.y);
        float tzmin = minv(static_cast<float>(static_cast<float>(blockPos.z) - rayOrigin.z) / rayDirection.z, static_cast<float>(static_cast<float>(blockPos.z) + 1.0 - rayOrigin.z) / rayDirection.z);
    
        tmin = maxv(maxv(tmin, tymin), tzmin);

        return Vector3<>(rayOrigin.x + tmin * rayDirection.x, rayOrigin.y + tmin * rayDirection.y, rayOrigin.z + tmin * rayDirection.z);
    }

    namespace revelles {
        __device__ unsigned char firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm, float epsilon) {
            float maxV = maxv(maxv(tx0, ty0), tz0);
    
            unsigned char v = 0;
    
            if (equals(maxV,tx0, epsilon)) {
                if (tym < tx0)
                    v |= 2;
                if (tzm < tx0)
                    v |= 1;
                return v;
            }
    
            if (equals(maxV, ty0, epsilon)) {
                if (txm < ty0)
                    v |= 4;
                if (tzm < ty0)
                    v |= 1;
                return v;
            }
    
            if (txm < tz0)
                v |= 4;
            if (tym < tz0)
                v |= 2;
            return v;
        }
    
        __device__ unsigned char newNode(float tx, unsigned char i1, float ty, unsigned char i2, float tz, unsigned char i3, float epsilon) {
            float minV = minv(minv(tx, ty), tz);
    
            if (equals(minV, tx, epsilon)) {
                return i1;
            }
            if (equals(minV, ty, epsilon)) {
                return i2;
            }
            return i3;
        }
    
        __device__ uint32_t childMortonRevelles(uint32_t mortonCode, unsigned char revellesChildIndex){
            static unsigned char reversed[8] = {0, 4, 2, 6, 1, 5, 3, 7};
    
            return (mortonCode << 3) | reversed[revellesChildIndex];
        }
    }
}