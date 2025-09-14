#include "scve/internal/octree/octree_utils.cuh"

#include "scve/internal/cuda_math.cuh"
#include "scve/internal/block_variant/block_variant_manager.cuh"
#include "scve/internal/structure/vector3.h"

namespace scve::octree_utils {
    __device__ scve::Vector3<> getBlockHitPos(scve::Vector3<int> blockPos, scve::Vector3<> rayOrigin, scve::Vector3<> rayDirection, float epsilon) {
        // if (rayDirection.x == 0)
        //     rayDirection.x = blockPos.x;

        // if (rayDirection.y == 0)
        //     rayDirection.y = blockPos.y;

        // if (rayDirection.z == 0)
        //     rayDirection.z = blockPos.z;

        // if (equals(rayDirection.x, 0, epsilon) || equals(rayDirection.y, 0, epsilon) || equals(rayDirection.z, 0, epsilon)) {
        //     return Vector3<>(blockPos.x, blockPos.y, blockPos.z);
        // }
        
        float tmin  = minv(static_cast<float>(static_cast<float>(blockPos.x) - rayOrigin.x) / rayDirection.x, static_cast<float>(static_cast<float>(blockPos.x) + 1.0 - rayOrigin.x) / rayDirection.x);
        float tymin = minv(static_cast<float>(static_cast<float>(blockPos.y) - rayOrigin.y) / rayDirection.y, static_cast<float>(static_cast<float>(blockPos.y) + 1.0 - rayOrigin.y) / rayDirection.y);
        float tzmin = minv(static_cast<float>(static_cast<float>(blockPos.z) - rayOrigin.z) / rayDirection.z, static_cast<float>(static_cast<float>(blockPos.z) + 1.0 - rayOrigin.z) / rayDirection.z);
    
        tmin = maxv(maxv(tmin, tymin), tzmin);

        scve::Vector3<> result = scve::Vector3<>(rayOrigin.x + tmin * rayDirection.x, rayOrigin.y + tmin * rayDirection.y, rayOrigin.z + tmin * rayDirection.z);

        // if(absv(result.x) - absv(blockPos.x) >= 0) {
        //     result.x = blockPos.x;
        // }

        // if(absv(result.y) - absv(blockPos.y) >= 0) {
        //     result.y = blockPos.y;
        // }

        // if(absv(result.z) - absv(blockPos.z) >= 0) {
        //     result.z = blockPos.z;
        // }

        return result;
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