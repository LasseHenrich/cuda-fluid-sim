#include "fluid.h"
#include "helper.h"

const int STROKE_RADIUS = 8;

FluidFields allocateFields(int width, int height, int depth) {
    // Note that in the lecture we always used cudaMemcpy to init the device vectors,
    // but since we don't need to pass any data from the host, we can simply use cudaMemset instead

    FluidFields fields;

    fields.width = width;
    fields.height = height;
    fields.depth = depth;

    size_t memSizeScalar_float1 = width * height * depth * sizeof(float);
    size_t memSizeScalar_float4 = width * height * depth * sizeof(float4);

    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMalloc(&fields.velocity[i], memSizeScalar_float4));
        CHECK_CUDA(cudaMalloc(&fields.dye[i], memSizeScalar_float1));
        CHECK_CUDA(cudaMalloc(&fields.pressure[i], memSizeScalar_float1));

        CHECK_CUDA(cudaMemset(fields.velocity[i], 0, memSizeScalar_float4));
        CHECK_CUDA(cudaMemset(fields.dye[i], 0, memSizeScalar_float1));
        CHECK_CUDA(cudaMemset(fields.pressure[i], 0, memSizeScalar_float1));
    }

    CHECK_CUDA(cudaMalloc(&fields.divergence, memSizeScalar_float1));
    CHECK_CUDA(cudaMemset(fields.divergence, 0, memSizeScalar_float1));

    return fields;
}

void freeFields(FluidFields& fields) {
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaFree(fields.velocity[i]));
        CHECK_CUDA(cudaFree(fields.dye[i]));
        CHECK_CUDA(cudaFree(fields.pressure[i]));
    }

    CHECK_CUDA(cudaFree(fields.divergence));
}

__global__ void initVortexKernel(float4* velocity, int width, int height, int depth) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    float dx = x - width / 2.0f;
    float dy = y - height / 2.0f;
    const float omega = 1.0f;
    velocity[idx3d(x, y, z, width, height)] = make_float4(-omega * dy, omega * dx, 0, 0);
}

void initVortex(FluidFields& fields) {
    initVortexKernel<<<getBlocksPerGrid(fields.width, fields.height, fields.depth), getThreadsPerBlock()>>>(
        fields.velocity[0], fields.width, fields.height, fields.depth);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void injectDyeAtPointKernel(float* dye, int width, int height, int depth, int centerX, int centerY,
                                       int centerZ) {
    // Todo: We should launch just 2rx2r threads, not for the whole grid
    int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    int threadY = blockDim.y * blockIdx.y + threadIdx.y;
    int threadZ = blockDim.z * blockIdx.z + threadIdx.z;
    if (threadX >= width || threadY >= height || threadZ >= depth) return;

    float r = STROKE_RADIUS;
    float dx = threadX - centerX;
    float dy = threadY - centerY;
    float dz = threadZ - centerZ;
    if (dx * dx + dy * dy + dz * dz < r * r) {
        dye[idx3d(threadX, threadY, threadZ, width, height)] += 0.02f * expf(-(dx * dx + dy * dy + dz * dz) / (r * r));  // Gaussian splat
    }
}

void injectDyeAtPoint(FluidFields& fields, int x, int y, int z) {
    injectDyeAtPointKernel<<<getBlocksPerGrid(fields.width, fields.height, fields.depth), getThreadsPerBlock()>>>(
        fields.dye[0], fields.width, fields.height, fields.depth, x, y,
        z);  // directly write to [0]... we need no reading
    CHECK_CUDA(cudaGetLastError());
}

__global__ void injectForceAtPointKernel(float4* velocity, int width, int height, int depth, int centerX, int centerY,
                                         int centerZ, float3 force) {
    // Todo: We should launch just 2rx2r threads, not for the whole grid
    // Todo: Cleanup, moving shared code with injectDyeAtPointKernel to a separate helper
    int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    int threadY = blockDim.y * blockIdx.y + threadIdx.y;
    int threadZ = blockDim.z * blockIdx.z + threadIdx.z;
    if (threadX >= width || threadY >= height || threadZ >= depth) return;

    float r = STROKE_RADIUS;
    float dx = threadX - centerX;
    float dy = threadY - centerY;
    float dz = threadZ - centerZ;
    if (dx * dx + dy * dy + dz * dz < r * r) {
        float multiplier = expf(-(dx * dx + dy * dy + dz * dz) / (r * r));  // Again gaussian falloff
        velocity[idx3d(threadX, threadY, threadZ, width, height)].x += force.x * multiplier;
        velocity[idx3d(threadX, threadY, threadZ, width, height)].y += force.y * multiplier;
        velocity[idx3d(threadX, threadY, threadZ, width, height)].z += force.z * multiplier;
    }
}

void injectForceAtPoint(FluidFields& fields, int x, int y, int z, float3 force) {
    injectForceAtPointKernel<<<getBlocksPerGrid(fields.width, fields.height, fields.depth), getThreadsPerBlock()>>>(
        fields.velocity[0], fields.width, fields.height, fields.depth, x, y, z, force);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void dyeToColorKernel(const float* dye, cudaSurfaceObject_t surface, int width, int height, int depth) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float d = fminf(dye[idx3d(x, y, depth / 2, width, height)], 1.0f);  // clamp since dye can exceed 1
    unsigned char c = (unsigned char)(d * 255.0f);
    surf2Dwrite(make_uchar4(c, c, c, 255), surface, x * sizeof(uchar4), y);
}

void renderDye(FluidFields& fields, cudaSurfaceObject_t surface) {
    dim3 threadsPerBlock = dim3(16, 16);
    dim3 blocksPerGrid = dim3((fields.width + 15) / 16, (fields.height + 15) / 16);
    dyeToColorKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.dye[0], surface, fields.width, fields.height, fields.depth);
    CHECK_CUDA(cudaGetLastError());
}
