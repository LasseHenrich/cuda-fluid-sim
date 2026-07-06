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