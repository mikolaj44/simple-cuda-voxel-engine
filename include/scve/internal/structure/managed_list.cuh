#pragma once

namespace scve {

template<typename T>
class ManagedList {
public:
    __host__ ManagedList() {};

    __host__ cudaError_t init(unsigned int initialCapacity) {
        capacity = initialCapacity;

        return cudaMallocManaged(&array, sizeof(T) * initialCapacity);
    }

    __host__ cudaError_t cleanup() {
        return cudaFree(array);
    }

    __host__ cudaError_t add(T element) {
        cudaError_t error = cudaSuccess;

        if(currentSize == capacity) {
            error = grow();

            if(error != cudaSuccess) {
                return error;
            }
        }

        array[currentSize++] = element;

        return error;
    }

    __host__ __device__ T& operator[] (int index) {
        return array[index];
    }

    __host__ __device__ unsigned int size() {
        return currentSize;
    }
private:
    T* array;

    const unsigned int ADDITIONAL_ALLOCATION_MULTIPLIER = 1;

    unsigned int currentSize = 0;

    unsigned int capacity;

    __host__ cudaError_t grow() {
        T* arrayCopy;

        size_t newCapacity = capacity + (capacity * ADDITIONAL_ALLOCATION_MULTIPLIER);

        cudaError_t error = cudaMallocManaged(&arrayCopy, newCapacity * sizeof(T));

        if(error != cudaSuccess) {
            return error;
        }

        error = cudaMemcpy(arrayCopy, array, capacity * sizeof(T), cudaMemcpyDefault);

        if(error != cudaSuccess) {
            cudaFree(arrayCopy);
            return error;
        }

        error = cudaFree(array);

        if(error != cudaSuccess) {
            cudaFree(arrayCopy);
            return error;
        }

        array = arrayCopy;

        capacity = newCapacity;

        return error;
    }
};

}