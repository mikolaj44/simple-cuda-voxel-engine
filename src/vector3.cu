#include "vector3.cuh"

template <typename T>
__device__ __host__ float Vector3<T>::len(){
    #ifdef __CUDA_ARCH__
        return sqrtf(static_cast<float>(x) * static_cast<float>(x)+ static_cast<float>(y) * static_cast<float>(y) + static_cast<float>(z) * static_cast<float>(z));
	#else
        return sqrt(static_cast<float>(x) * static_cast<float>(x)+ static_cast<float>(y) * static_cast<float>(y) + static_cast<float>(z) * static_cast<float>(z));
	#endif
    
}

template <typename T>
__device__ __host__ Vector3<T>& Vector3<T>::norm(){
    div(*this, len(*this));
    return *this;
}

template <typename T>
__device__ __host__ float Vector3<T>::dot(const Vector3<T>& other){
    return static_cast<float>(x) * static_cast<float>(other.x) + static_cast<float>(y)* static_cast<float>(other.y) + static_cast<float>(z) * static_cast<float>(other.z);
}

template <typename T>
__device__ __host__ Vector3<T>& Vector3<T>::add(T val){
    x = static_cast<T>(x + val);
    y = static_cast<T>(y + val);
    z = static_cast<T>(z + val);
    return *this;
}

template <typename T>
__device__ __host__ Vector3<T>& Vector3<T>::sub(T val){
    x = static_cast<T>(x - val);
    y = static_cast<T>(y - val);
    z = static_cast<T>(z - val);
    return *this;
}

template <typename T>
__device__ __host__ Vector3<T>& Vector3<T>::mul(T val){
    x = static_cast<T>(x * val);
    y = static_cast<T>(y * val);
    z = static_cast<T>(z * val);
    return *this;
}

template <typename T>
__device__ __host__ Vector3<T>& Vector3<T>::div(T val){
    x = static_cast<T>(x / val);
    y = static_cast<T>(y / val);
    z = static_cast<T>(z / val);
    return *this;
}

template <typename T>
__device__ __host__ Vector3<T>& Vector3<T>::pow(T val){
    #ifdef __CUDA_ARCH__
        x = static_cast<T>(powf(x, val));
        y = static_cast<T>(powf(y, val));
        z = static_cast<T>(powf(z, val));
    #else
        x = static_cast<T>(pow(x, val));
        y = static_cast<T>(pow(y, val));
        z = static_cast<T>(pow(z, val));
    #endif
    
    return *this;
}

template <typename T>
__device__ __host__ Vector3<T>& Vector3<T>::add(const Vector3<T>& other){
    x = static_cast<T>(x + other.x);
    y = static_cast<T>(y + other.y);
    z = static_cast<T>(z + other.z);
    return *this;
}

template <typename T>
__device__ __host__ Vector3<T>& Vector3<T>::sub(const Vector3<T>& other){
    x = static_cast<T>(x - other.x);
    y = static_cast<T>(y - other.y);
    z = static_cast<T>(z - other.z);
    return *this;
}

template <typename T>
__device__ __host__ Vector3<T>& Vector3<T>::mul(const Vector3<T>& other){
    x = static_cast<T>(x * other.x);
    y = static_cast<T>(y * other.y);
    z = static_cast<T>(z * other.z);
    return *this;
}

template <typename T>
__device__ __host__ Vector3<T>& Vector3<T>::div(const Vector3<T>& other){
    x = static_cast<T>(x / other.x);
    y = static_cast<T>(y / other.y);
    z = static_cast<T>(z / other.z);
    return *this;
}