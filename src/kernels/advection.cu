#include <utility>

#include "advection.h"
#include "helper.h"

/// @brief // backtrace to where a quantity (dye or velocity) came from
__device__ float3 backTrace(const float4* velocity, int x, int y, int z, int width, int height, int depth,
                            float deltaTime) {
    float4 myVel = velocity[idx3d(x, y, z, width, height)];
    float sourceX = x - deltaTime * myVel.x;
    float sourceY = y - deltaTime * myVel.y;
    float sourceZ = z - deltaTime * myVel.z;

    // clamping to grid bounds [0, GRID_WIDTH-1)
    sourceX = fminf(fmaxf(sourceX, 0.0f), width - 1.0001f);
    sourceY = fminf(fmaxf(sourceY, 0.0f), height - 1.0001f);
    sourceZ = fminf(fmaxf(sourceZ, 0.0f), depth - 1.0001f);

    return make_float3(sourceX, sourceY, sourceZ);
}

__device__ __host__ inline float4 operator*(float mult, float4 a) {
    return make_float4(mult * a.x, mult * a.y, mult * a.z, 0); // we don't use the fourth dimension right now, so spare the extra computation...
}

__device__ __host__ inline float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0);
}

__device__ __host__ inline float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0);
}

/// @brief trilinear interpolation between eight surrounding cells, (0,0,0) (left,bottom,front) to (1,1,1)
/// (right,top,back), to determine value of a quantity
template <typename T>
__device__ T trilinearlyInterpolate(const T* field, float x, float y, float z, int width, int height) {
    int x0 = (int)x, y0 = (int)y, z0 = (int)z;
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

    // 1. sample 8 corners
    T val000 = field[idx3d(x0, y0, z0, width, height)];
    T val100 = field[idx3d(x1, y0, z0, width, height)];
    T val010 = field[idx3d(x0, y1, z0, width, height)];
    T val110 = field[idx3d(x1, y1, z0, width, height)];
    T val001 = field[idx3d(x0, y0, z1, width, height)];
    T val101 = field[idx3d(x1, y0, z1, width, height)];
    T val011 = field[idx3d(x0, y1, z1, width, height)];
    T val111 = field[idx3d(x1, y1, z1, width, height)];

    float fracX = x - x0, fracY = y - y0, fracZ = z - z0;

    // 2. 4 lerps along x axis
    T valFrontBottom = (1 - fracX) * val000 + fracX * val100;
    T valFrontTop = (1 - fracX) * val010 + fracX * val110;
    T valBackBottom = (1 - fracX) * val001 + fracX * val101;
    T valBackTop = (1 - fracX) * val011 + fracX * val111;

    // 3. 2 lerps along y axis
    T valFront = (1 - fracY) * valFrontBottom + fracY * valFrontTop;
    T valBack = (1 - fracY) * valBackBottom + fracY * valBackTop;

    // 4. 1 lerp along z axis
    T val = (1 - fracZ) * valFront + fracZ * valBack;

    return val;
}

__global__ void advectDyeKernel(const float4* velocity, const float* dyeIn, float* dyeOut, int width, int height,
                                int depth, float deltaTime) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    float3 source = backTrace(velocity, x, y, z, width, height, depth, deltaTime);
    float sourceX = source.x;
    float sourceY = source.y;
    float sourceZ = source.z;
    
    float dye = trilinearlyInterpolate(dyeIn, sourceX, sourceY, sourceZ, width, height);
    
    dyeOut[idx3d(x, y, z, width, height)] = dye;
}

void advectDye(FluidFields& fields, float deltaTime) {
    advectDyeKernel<<<getBlocksPerGrid(fields.width, fields.height, fields.depth), getThreadsPerBlock()>>>(
        fields.velocity[0], fields.dye[0], fields.dye[1], fields.width, fields.height, fields.depth, deltaTime);
    CHECK_CUDA(cudaGetLastError());
    std::swap(fields.dye[0], fields.dye[1]);
}

__global__ void advectVelocityKernel(const float4* velIn, float4* velOut, int width, int height, int depth,
                                     float deltaTime) {
    // Todo: Cleanup, moving shared code with advectDyeKernel to a separate helper

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    float3 source = backTrace(velIn, x, y, z, width, height, depth, deltaTime);
    float sourceX = source.x;
    float sourceY = source.y;
    float sourceZ = source.z;

    float4 vel = trilinearlyInterpolate(velIn, sourceX, sourceY, sourceZ, width, height);

    velOut[idx3d(x, y, z, width, height)] = vel;
}

void advectVelocity(FluidFields& fields, float deltaTime) {
    advectVelocityKernel<<<getBlocksPerGrid(fields.width, fields.height, fields.depth), getThreadsPerBlock()>>>(
        fields.velocity[0], fields.velocity[1], fields.width, fields.height, fields.depth, deltaTime);
    CHECK_CUDA(cudaGetLastError());
    std::swap(fields.velocity[0], fields.velocity[1]);
}