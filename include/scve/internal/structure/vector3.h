#pragma once

namespace scve {

/**
* @brief This is a class mostly used to represent things like positions or colors. It is used like **Vector3<>** (implicitly **Vector3<float>**) or **Vector3<int>** \par
* All of these methods are pretty simple math operations so I will only write some simple details about them here. \rtfinclude
* Operations like **Vector3<T>& add(T val)** modify the vector the operation is performed on, so you can also use their versions which create a copy, like
* **Vector3<T> add(Vector3<T> v1, Vector3<T> v2)** for example. There are also equality (==) and inequality (!=) operators which compare all of the components.
*/
template<typename T = float>
class alignas(4) Vector3 {
public:
    T x;
    T y;
    T z;

    __host__ __device__ Vector3() : x(T{}), y(T{}), z(T{}) {};

    __host__ __device__ Vector3(T val) : x(val), y(val), z(val) {};

    __host__ __device__ Vector3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {};

    __host__ __device__ operator Vector3<int>() const {
        return Vector3<int>(x, y, z);
    }

    __host__ __device__ operator Vector3<float>() const {
        return Vector3<float>(x, y, z);
    }

    __host__ __device__ float len(){
        #ifdef __CUDA_ARCH__
            return sqrtf(static_cast<float>(x) * static_cast<float>(x) + static_cast<float>(y) * static_cast<float>(y) + static_cast<float>(z) * static_cast<float>(z));
        #else
            return sqrt(static_cast<float>(x) * static_cast<float>(x) + static_cast<float>(y) * static_cast<float>(y) + static_cast<float>(z) * static_cast<float>(z));
        #endif
        
    }

    __host__ __device__ Vector3<T>& norm(){
        div(len());
        return *this;
    }

    __host__ __device__ float dot(const Vector3<T>& other){
        return static_cast<float>(x) * static_cast<float>(other.x) + static_cast<float>(y)* static_cast<float>(other.y) + static_cast<float>(z) * static_cast<float>(other.z);
    }

    __host__ __device__ Vector3<T>& add(T val){
        x = static_cast<T>(x + val);
        y = static_cast<T>(y + val);
        z = static_cast<T>(z + val);
        return *this;
    }

    __host__ __device__ Vector3<T>& sub(T val){
        x = static_cast<T>(x - val);
        y = static_cast<T>(y - val);
        z = static_cast<T>(z - val);
        return *this;
    }

    __host__ __device__ Vector3<T>& mul(T val){
        x = static_cast<T>(x * val);
        y = static_cast<T>(y * val);
        z = static_cast<T>(z * val);
        return *this;
    }

    __host__ __device__ Vector3<T>& div(T val){
        x = static_cast<T>(x / val);
        y = static_cast<T>(y / val);
        z = static_cast<T>(z / val);
        return *this;
    }

    __host__ __device__ Vector3<T>& pow(T val){
        #ifdef __CUDA_ARCH__
            x = static_cast<T>(powf(x, val));
            y = static_cast<T>(powf(y, val));
            z = static_cast<T>(powf(z, val));
        #else
            x = static_cast<T>(std::pow(x, val));
            y = static_cast<T>(std::pow(y, val));
            z = static_cast<T>(std::pow(z, val));
        #endif
        
        return *this;
    }

    __host__ __device__ Vector3<T>& clamp(T val){
        if (x > val) {
            x = val;
        }
        if (y > val) {
            y = val;
        }
        if (z > val) {
            z = val;
        }
        
        return *this;
    } 

    __host__ __device__ Vector3<T>& add(const Vector3<T>& other){
        x = static_cast<T>(x + other.x);
        y = static_cast<T>(y + other.y);
        z = static_cast<T>(z + other.z);
        return *this;
    }

    __host__ __device__ Vector3<T>& sub(const Vector3<T>& other){
        x = static_cast<T>(x - other.x);
        y = static_cast<T>(y - other.y);
        z = static_cast<T>(z - other.z);
        return *this;
    }

    __host__ __device__ Vector3<T>& mul(const Vector3<T>& other){
        x = static_cast<T>(x * other.x);
        y = static_cast<T>(y * other.y);
        z = static_cast<T>(z * other.z);
        return *this;
    }

    __host__ __device__ Vector3<T>& div(const Vector3<T>& other){
        x = static_cast<T>(x / other.x);
        y = static_cast<T>(y / other.y);
        z = static_cast<T>(z / other.z);
        return *this;
    }

    friend __host__ __device__ bool operator==(const Vector3<T>& v1, const Vector3<T>& v2){
        return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
    }

    friend __host__ __device__ bool operator!=(const Vector3<T>& v1, const Vector3<T>& v2){
        return !(v1 == v2);
    }

    __host__ __device__ static Vector3<T> norm(Vector3<T> v) {
        return v.norm();
    }

    __host__ __device__ static float dot(Vector3<T> v1, Vector3<T> v2) {
        return v1.dot(v2);
    }

    __host__ __device__ static Vector3<T> add(Vector3<T> v, T val) {
        return v.add(val);
    }

    __host__ __device__ static Vector3<T> sub(Vector3<T> v, T val) {
        return v.sub(val);
    }

    __host__ __device__ static Vector3<T> mul(Vector3<T> v, T val) {
        return v.mul(val);
    }

    __host__ __device__ static Vector3<T> div(Vector3<T> v, T val) {
        return v.div(val);
    }

    __host__ __device__ static Vector3<T> pow(Vector3<T> v, T val) {
        return v.pow(val);
    }

    __host__ __device__ static Vector3<T> clamp(Vector3<T> v, T val) {
        return v.clamp(val);
    }

    __host__ __device__ static Vector3<T> add(Vector3<T> v1, Vector3<T> v2) {
        return v1.add(v2);
    }

    __host__ __device__ static Vector3<T> sub(Vector3<T> v1, Vector3<T> v2) {
        return v1.sub(v2);
    }

    __host__ __device__ static Vector3<T> mul(Vector3<T> v1, Vector3<T> v2) {
        return v1.mul(v2);
    }

    __host__ __device__ static Vector3<T> div(Vector3<T> v1, Vector3<T> v2) {
        return v1.div(v2);
    }
};

}