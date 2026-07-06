#include "fluid.h"
#include "helper.h"

const int STROKE_RADIUS = 32;

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

__global__ void initVortexKernel(float2* velocity, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float dx = x - width / 2.0f;
    float dy = y - height / 2.0f;
    const float omega = 1.0f;
    velocity[y * width + x] = make_float2(-omega * dy, omega * dx);
}

void initVortex(FluidFields& fields) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((fields.width + 15) / 16, (fields.height + 15) / 16);
    initVortexKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.velocity[0], fields.width, fields.height);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void injectDyeAtPointKernel(float* dye, int width, int height, int centerX, int centerY) {
    // Todo: We should launch just 2rx2r threads, not for the whole grid
    int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    int threadY = blockDim.y * blockIdx.y + threadIdx.y;
    if (threadX >= width || threadY >= height) return;

    float dx = threadX - centerX;
    float dy = threadY - centerY;
    float r = STROKE_RADIUS;
    if (dx * dx + dy * dy < r * r) {
        dye[threadY * width + threadX] += 0.02f * expf(-(dx * dx + dy * dy) / (r * r));  // Gaussian splat
    }
}

void injectDyeAtPoint(FluidFields& fields, int x, int y) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((fields.width + 15) / 16, (fields.height + 15) / 16);
    injectDyeAtPointKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.dye[0], fields.width, fields.height, x,
                                                               y);  // directly write to [0]... we need no reading
    CHECK_CUDA(cudaGetLastError());
}

__global__ void injectForceAtPointKernel(float2* velocity, int width, int height, int centerX, int centerY,
                                         float2 force) {
    // Todo: We should launch just 2rx2r threads, not for the whole grid
    // Todo: Cleanup, moving shared code with injectDyeAtPointKernel to a separate helper
    int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    int threadY = blockDim.y * blockIdx.y + threadIdx.y;
    if (threadX >= width || threadY >= height) return;

    float dx = threadX - centerX;
    float dy = threadY - centerY;
    float r = STROKE_RADIUS;
    if (dx * dx + dy * dy < r * r) {
        float multiplier = expf(-(dx * dx + dy * dy) / (r * r));  // Again gaussian falloff
        velocity[threadY * width + threadX].x += force.x * multiplier;
        velocity[threadY * width + threadX].y += force.y * multiplier;
    }
}

void injectForceAtPoint(FluidFields& fields, int x, int y, float2 force) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((fields.width + 15) / 16, (fields.height + 15) / 16);
    injectForceAtPointKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.velocity[0], fields.width, fields.height, x, y,
                                                                 force);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void dyeToColorKernel(const float* dye, cudaSurfaceObject_t surface, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float d = fminf(dye[y * width + x], 1.0f);  // clamp since dye can exceed 1
    unsigned char c = (unsigned char)(d * 255.0f);
    surf2Dwrite(make_uchar4(c, c, c, 255), surface, x * sizeof(uchar4), y);
}

void renderDye(FluidFields& fields, cudaSurfaceObject_t surface) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((fields.width + 15) / 16, (fields.height + 15) / 16);
    dyeToColorKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.dye[0], surface, fields.width, fields.height);
    CHECK_CUDA(cudaGetLastError());
}
