#include "octree/octree_utils.cuh"

#include "globals.cuh"
#include "cuda_math.cuh"
#include "blocks_data.cuh"
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

    __device__ void drawTexturePixel(int blockX, int blockY, int blockZ, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, unsigned char blockId, uchar4* pixels, bool textureRenderingEnabled) {
        if (dX == 0 || dY == 0 || dZ == 0) { // for now
            return;
        }
        
        float tmin =  minv((float)((float)blockX - oX) / dX, (float)((float)blockX + 1.0 - oX) / dX);
        float tymin = minv((float)((float)blockY - oY) / dY, (float)((float)blockY + 1.0 - oY) / dY);
        float tzmin = minv((float)((float)blockZ - oZ) / dZ, (float)((float)blockZ + 1.0 - oZ) / dZ);
    
        tmin = maxv(maxv(tmin, tymin), tzmin);
    
        float x = oX + tmin * dX;
        float y = oY + tmin * dY;
        float z = oZ + tmin * dZ;
        
        setPixelById(sX, sY, blockX, blockY, blockZ, x, y, z, blockId, pixels, Vector3(oX, oY, oZ), PointLight(Vector3(1500,-500,-5000), Vector3(255, 255, 255)), textureRenderingEnabled); //oX - dX * 1000, oY - dY * 1000, oZ - dZ * 1000    }
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