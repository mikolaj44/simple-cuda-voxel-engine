#pragma once

template<typename T = int>
class BlockInfo {
public:
    Vector3<T> pos;
    uint8_t id;

    __device__ BlockInfo() : pos(Vector3<T>{}), id(uint8_t{}) {};

    __device__ BlockInfo(Vector3<T> pos_, uint8_t id_) : pos(pos_), id(id_) {};
    
    friend __device__ bool operator==(const BlockInfo<T>& b1, const BlockInfo<T>& b2){
        return b1.pos == b2.pos && b1.id == b2.id;
    }
    friend __device__ bool operator!=(const BlockInfo<T>& b1, const BlockInfo<T>& b2){
        return !(b2 == b2);
    }
};