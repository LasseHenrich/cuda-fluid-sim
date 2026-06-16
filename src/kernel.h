#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                                                    \
    do {                                                                                                    \
        cudaError_t err = (call);                                                                           \
        if (err != cudaSuccess) {                                                                           \
            std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                                             \
        }                                                                                                   \
    } while (0)

void registerTexture(unsigned int glTexture, cudaGraphicsResource** glTextureCudaHandle);
void unregisterTexture(cudaGraphicsResource* cudaPBO);
void runCudaKernel(cudaGraphicsResource* cudaPBO, int widht, int height, float time);