#pragma once

namespace scve {

template<typename T = float>
class alignas(4) Vector3 {
public:
    T x;
    T y;
    T z;

    __device__ __host__ Vector3() : x(T{}), y(T{}), z(T{}) {};

    __device__ __host__ Vector3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {};

    __device__ __host__ float len();

    __device__ __host__ Vector3<T>& norm();

    __device__ __host__ float dot(const Vector3<T>& other);
    
    __device__ __host__ Vector3<T>& add(T val);

    __device__ __host__ Vector3<T>& sub(T val);

    __device__ __host__ Vector3<T>& mul(T val);

    __device__ __host__ Vector3<T>& div(T val);

    __device__ __host__ Vector3<T>& pow(T val);

    __device__ __host__ Vector3<T>& clamp(T val);

    __device__ __host__ Vector3<T>& add(const Vector3<T>& other);

    __device__ __host__ Vector3<T>& sub(const Vector3<T>& other);

    __device__ __host__ Vector3<T>& mul(const Vector3<T>& other);

    __device__ __host__ Vector3<T>& div(const Vector3<T>& other);

    friend __device__ __host__ bool operator==(const Vector3<T>& v1, const Vector3<T>& v2){
        return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
    }

    friend __device__ __host__ bool operator!=(const Vector3<T>& v1, const Vector3<T>& v2){
        return !(v1 == v2);
    }

    __device__ __host__ static Vector3<T> norm(Vector3<T> v) {
        return v.norm();
    }

    __device__ __host__ static float dot(Vector3<T> v1, Vector3<T> v2) {
        return v1.dot(v2);
    }

    __device__ __host__ static Vector3<T> add(Vector3<T> v, T val) {
        return v.add(val);
    }

    __device__ __host__ static Vector3<T> sub(Vector3<T> v, T val) {
        return v.sub(val);
    }

    __device__ __host__ static Vector3<T> mul(Vector3<T> v, T val) {
        return v.mul(val);
    }

    __device__ __host__ static Vector3<T> div(Vector3<T> v, T val) {
        return v.div(val);
    }

    __device__ __host__ static Vector3<T> pow(Vector3<T> v, T val) {
        return v.pow(val);
    }

    __device__ __host__ static Vector3<T> clamp(Vector3<T> v, T val) {
        return v.clamp(val);
    }

    __device__ __host__ static Vector3<T> add(Vector3<T> v1, Vector3<T> v2) {
        return v1.add(v2);
    }

    __device__ __host__ static Vector3<T> sub(Vector3<T> v1, Vector3<T> v2) {
        return v1.sub(v2);
    }

    __device__ __host__ static Vector3<T> mul(Vector3<T> v1, Vector3<T> v2) {
        return v1.mul(v2);
    }

    __device__ __host__ static Vector3<T> div(Vector3<T> v1, Vector3<T> v2) {
        return v1.div(v2);
    }
};

}