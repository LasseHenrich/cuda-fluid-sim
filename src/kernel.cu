#include <device_launch_parameters.h>

#include <iostream>

#include "kernel.h"

__global__ void generatePlasma(cudaSurfaceObject_t surface, int width,
                               int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float u = (float)x / width;
    float v = (float)y / height;

    unsigned char r =
        (unsigned char)(sinf(u * 5.0f + time) * 127.5f + 127.5f);  // [0, 255]
    unsigned char g = (unsigned char)(cosf(u * 5.0f + time) * 127.5f + 127.5f);
    unsigned char b =
        (unsigned char)(sinf((u + v) * 3.0f + time) * 127.5f + 127.5f);

    uchar4 color = make_uchar4(r, g, b, 255);

    surf2Dwrite(color, surface, x * sizeof(uchar3),
                y);  // write directly to OpenGL texture
}

void registerTexture(unsigned int glTextureId, cudaGraphicsResource** cudaPBO) {
    cudaError_t err =
        cudaGraphicsGLRegisterImage(cudaPBO, glTextureId, GL_TEXTURE_2D,
                                    cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to register GL texture: " << cudaGetErrorString(err)
                  << "\n";
    }
}

void unregisterTexture(cudaGraphicsResource* cudaPBO) {
    cudaGraphicsUnregisterResource(cudaPBO);
}

void runCudaKernel(cudaGraphicsResource* cudaPBO, int width, int height,
                   float time) {
    cudaGraphicsMapResources(1, &cudaPBO);

    cudaArray_t textureArray;
    cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaPBO, 0, 0);

    cudaResourceDesc resDesc;  // resource descriptor
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    cudaSurfaceObject_t surface;
    cudaCreateSurfaceObject(&surface, &resDesc);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    generatePlasma<<<gridSize, blockSize>>>(surface, width, height, time);

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &cudaPBO);
}