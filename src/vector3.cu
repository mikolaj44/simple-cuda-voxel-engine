#include "vector3.cuh"

float len(Vector3 v){
    #ifdef __CUDA_ARCH__
        return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	#else
        return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	#endif
    
}

float dot(Vector3 v1, Vector3 v2){
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

Vector3 mul(Vector3 v, float val){
    return Vector3(v.x * val, v.y * val, v.z * val);
}

Vector3 div(Vector3 v, float val){
    return Vector3(v.x / val, v.y / val, v.z / val);
}

Vector3 norm(Vector3 v){
    return div(v, len(v));
}

Vector3 add(Vector3 v1, Vector3 v2){
    return Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

Vector3 sub(Vector3 v1, Vector3 v2){
    return Vector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

Vector3 vmul(Vector3 v1, Vector3 v2){
    return Vector3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

Vector3 pow(Vector3 v, float val){
    #ifdef __CUDA_ARCH__
        return Vector3(powf(v.x, val), powf(v.y, val), powf(v.z, val));
    #else
        return Vector3(pow(v.x, val), pow(v.y, val), pow(v.z, val));
	#endif
}