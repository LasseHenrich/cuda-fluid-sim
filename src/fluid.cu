#include "fluid.h"

#include "helper.h"

FluidFields allocateFields(int width, int height) {
    // Note that in the lecture we always used cudaMemcpy to init the device vectors,
    // but since we don't need to pass any data from the host, we can simply use cudaMemset instead

    FluidFields fields;

    fields.width = width;
    fields.height = height;
    size_t memSizeScalar_float1 = width * height * sizeof(float);
    size_t memSizeScalar_float2 = width * height * sizeof(float2);

    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMalloc(&fields.velocity[i], memSizeScalar_float2));
        CHECK_CUDA(cudaMalloc(&fields.dye[i], memSizeScalar_float1));
        CHECK_CUDA(cudaMalloc(&fields.pressure[i], memSizeScalar_float1));

        CHECK_CUDA(cudaMemset(fields.velocity[i], 0, memSizeScalar_float2));
        CHECK_CUDA(cudaMemset(fields.dye[i], 0, memSizeScalar_float1));
        CHECK_CUDA(cudaMemset(fields.pressure[i], 0, memSizeScalar_float1));
    }

    return fields;
}

void freeFields(FluidFields& fields) {
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaFree(fields.velocity[i]));
        CHECK_CUDA(cudaFree(fields.dye[i]));
        CHECK_CUDA(cudaFree(fields.pressure[i]));
    }
}

/// @brief temporary test pattern: a filled dye circle in the grid center
__global__ void seedDyeKernel(float* dye, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float dx = x - width / 2.0f;
    float dy = y - height / 2.0f;
    dye[y * width + x] = (dx * dx + dy * dy < 60.0f * 60.0f) ? 1.0f : 0.0f;
}

__global__ void dyeToColorKernel(const float* dye, cudaSurfaceObject_t surface, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float d = fminf(dye[y * width + x], 1.0f);  // clamp since dye can exceed 1
    unsigned char c = (unsigned char)(d * 255.0f);
    surf2Dwrite(make_uchar4(c, c, c, 255), surface, x * sizeof(uchar4), y);
}

void seedDye(FluidFields& f) {
    dim3 blockSize(16, 16);
    dim3 gridSize((f.width + 15) / 16, (f.height + 15) / 16);
    seedDyeKernel<<<gridSize, blockSize>>>(f.dye[0], f.width, f.height);
    CHECK_CUDA(cudaGetLastError());
}

void renderDye(FluidFields& fields, cudaSurfaceObject_t surface) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((fields.width + 15) / 16, (fields.height + 15) / 16);
    dyeToColorKernel<<<threadsPerBlock, blocksPerGrid>>>(fields.dye[0], surface, fields.width, fields.height);
    CHECK_CUDA(cudaGetLastError());
}
