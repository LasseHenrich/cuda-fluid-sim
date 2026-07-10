#include "simple.h"

/// @brief One iteration of the Jacobi iteration to solve for pressure
__global__ void jacobiKernel_simple(const float* pressureIn, float* pressureOut, const float* divergence, int width,
                                    int height, int depth) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    int xLeft = max(x - 1, 0), xRight = min(x + 1, width - 1);
    int yBottom = max(y - 1, 0), yTop = min(y + 1, height - 1);
    int zFront = max(z - 1, 0), zBack = min(z + 1, depth - 1);

    float p =
        (1 / 6.0f) * (pressureIn[idx3d(xLeft, y, z, width, height)] + pressureIn[idx3d(xRight, y, z, width, height)] +
                      pressureIn[idx3d(x, yBottom, z, width, height)] + pressureIn[idx3d(x, yTop, z, width, height)] +
                      pressureIn[idx3d(x, y, zFront, width, height)] + pressureIn[idx3d(x, y, zBack, width, height)] -
                      divergence[idx3d(x, y, z, width, height)]);

    pressureOut[idx3d(x, y, z, width, height)] = p;
}

void computePressure_simple(FluidFields& fields, int iterationCount) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((fields.width + OUT_SIZE - 1) / OUT_SIZE, (fields.height + OUT_SIZE - 1) / OUT_SIZE,
                         (fields.depth + OUT_SIZE - 1) / OUT_SIZE);
    for (int i = 0; i < iterationCount; i++) {
        jacobiKernel_simple<<<blocksPerGrid, threadsPerBlock>>>(
            fields.pressure[0], fields.pressure[1], fields.divergence, fields.width, fields.height, fields.depth);
        std::swap(fields.pressure[0], fields.pressure[1]);
    }
    CHECK_CUDA(cudaGetLastError());
}