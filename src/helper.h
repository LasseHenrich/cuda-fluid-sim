#pragma once
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                                                                    \
    do {                                                                                                    \
        cudaError_t err = (call);                                                                           \
        if (err != cudaSuccess) {                                                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) \
                      << std::endl;                                                                         \
            exit(EXIT_FAILURE);                                                                             \
        }                                                                                                   \
    } while (0)

inline __device__ int idx3d(int x, int y, int z, int width, int height) {
    return (z * height + y) * width + x; // z is depth, y is height, x is width
}

inline dim3 getThreadsPerBlock() {
    return dim3(8, 8, 8);
}

inline dim3 getBlocksPerGrid(int width, int height, int depth) {
    return dim3((width + 7) / 8, (height + 7) / 8, (depth + 7) / 8);
}