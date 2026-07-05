#include <device_launch_parameters.h>

#include <iostream>

#include "helper.h"
#include "kernels/plasmaKernel.h"

/// @brief Test plasma kernel, purely for debugging
__global__ void generatePlasma(cudaSurfaceObject_t surface, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float u = (float)x / width;
    float v = (float)y / height;

    unsigned char r = (unsigned char)(sinf(u * 5.0f + time) * 127.5f + 127.5f);  // [0, 255]
    unsigned char g = (unsigned char)(cosf(u * 5.0f + time) * 127.5f + 127.5f);
    unsigned char b = (unsigned char)(sinf((u + v) * 3.0f + time) * 127.5f + 127.5f);

    uchar4 color = make_uchar4(r, g, b, 255);

    surf2Dwrite(color, surface, x * sizeof(uchar4),
                y);  // write directly to OpenGL texture
}

/// @brief lock texture, run kernels to compute colors, unlock texture
void runCudaKernel(cudaGraphicsResource* glTextureCudaHandle, int width, int height, float time) {
    CHECK_CUDA(cudaGraphicsMapResources(1, &glTextureCudaHandle)); // locks texture and handing ownership to cuda

    cudaArray_t textureArray;
    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&textureArray, glTextureCudaHandle, 0, 0));

    cudaResourceDesc resDesc;  // resource descriptor
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    cudaSurfaceObject_t surface;
    CHECK_CUDA(cudaCreateSurfaceObject(&surface, &resDesc));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    generatePlasma<<<gridSize, blockSize>>>(surface, width, height, time);
    CHECK_CUDA(cudaGetLastError()); // kernel launches are silent and asynchronous otherwise

    CHECK_CUDA(cudaDestroySurfaceObject(surface));
    CHECK_CUDA(cudaGraphicsUnmapResources(1, &glTextureCudaHandle));
}