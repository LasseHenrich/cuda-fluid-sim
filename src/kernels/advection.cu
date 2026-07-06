#include "advection.h"
#include "helper.h"
#include <utility>

__global__ void advectDyeKernel(const float2* velocity, const float* dyeIn, float* dyeOut, int width, int height,
                                float deltaTime) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // backtrace to where fluid came from
    float2 myVel = velocity[y * width + x];
    float sourceX = x - deltaTime * myVel.x;
    float sourceY = y - deltaTime * myVel.y;

    // clamping to grid bounds [0, GRID_WIDTH-1)
    sourceX = fminf(fmaxf(sourceX, 0.0f), width - 1.0001f);
    sourceY = fminf(fmaxf(sourceY, 0.0f), height - 1.0001f);

    // bilinear interpolation between four surrounding cells, (0,0) (left,bottom) to (1,1) (right,top)

    int x0 = (int)sourceX, y0 = (int)sourceY;
    int x1 = x0 + 1, y1 = y0 + 1;

    float dye00 = dyeIn[y0 * width + x0];
    float dye10 = dyeIn[y0 * width + x1];
    float dye01 = dyeIn[y1 * width + x0];
    float dye11 = dyeIn[y1 * width + x1];

    float dyeBottom = (1 - (sourceX - x0)) * dye00 + (sourceX - x0) * dye10;
    float dyeTop = (1 - (sourceX - x0)) * dye01 + (sourceX - x0) * dye11;

    float dye = (1 - (sourceY - y0)) * dyeBottom + (sourceY - y0) * dyeTop;

    dyeOut[y * width + x] = dye;
}

void advectDye(FluidFields& fields, float deltaTime) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((fields.width + 15) / 16, (fields.height + 15) / 16);
    advectDyeKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.velocity[0], fields.dye[0], fields.dye[1], fields.width,
                                                        fields.height, deltaTime);
    CHECK_CUDA(cudaGetLastError());
    std::swap(fields.dye[0], fields.dye[1]);
}
