#include "helper.h"
#include "projection.h"

/// @brief Computes the change of the velocity field using finite differences
__global__ void divergenceKernel(const float4* velocity, float* divergence, int width, int height, int depth) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    int xLeft = max(x - 1, 0);
    int xRight = min(x + 1, width - 1);
    int yBottom = max(y - 1, 0);
    int yTop = min(y + 1, height - 1);
    int zFront = max(z - 1, 0);
    int zBack = min(z + 1, depth - 1);

    float div =
        0.5f * (velocity[idx3d(xRight, y, z, width, height)].x - velocity[idx3d(xLeft, y, z, width, height)].x) +
        0.5f * (velocity[idx3d(x, yTop, z, width, height)].y - velocity[idx3d(x, yBottom, z, width, height)].y) +
        0.5f * (velocity[idx3d(x, y, zBack, width, height)].z - velocity[idx3d(x, y, zFront, width, height)].z);

    divergence[idx3d(x, y, z, width, height)] = div;
}

/// @brief One iteration of the Jacobi iteration to solve for pressure
__global__ void jacobiKernel(const float* pressureIn, float* pressureOut, const float* divergence, int width,
                             int height, int depth) {
    // Todo: Cleanup by merging shared logic with divergenceKernel

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    int xLeft = max(x - 1, 0);
    int xRight = min(x + 1, width - 1);
    int yBottom = max(y - 1, 0);
    int yTop = min(y + 1, height - 1);
    int zFront = max(z - 1, 0);
    int zBack = min(z + 1, depth - 1);

    float p =
        (1 / 6.0f) * (pressureIn[idx3d(xLeft, y, z, width, height)] + pressureIn[idx3d(xRight, y, z, width, height)] +
                      pressureIn[idx3d(x, yBottom, z, width, height)] + pressureIn[idx3d(x, yTop, z, width, height)] +
                      pressureIn[idx3d(x, y, zFront, width, height)] + pressureIn[idx3d(x, y, zBack, width, height)] -
                      divergence[idx3d(x, y, z, width, height)]);

    pressureOut[idx3d(x, y, z, width, height)] = p;
}

/// @brief Subtracts the pressure gradient from the velocity, where the gradient is calculated using finite differences
__global__ void subtractGradientKernel(float4* velocity, const float* pressure, int width, int height, int depth) {
    // Todo: Cleanup by merging shared logic with divergenceKernel

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    int xLeft = max(x - 1, 0);
    int xRight = min(x + 1, width - 1);
    int yBottom = max(y - 1, 0);
    int yTop = min(y + 1, height - 1);
    int zFront = max(z - 1, 0);
    int zBack = min(z + 1, depth - 1);

    float gradX = 0.5f * (pressure[idx3d(xRight, y, z, width, height)] - pressure[idx3d(xLeft, y, z, width, height)]);
    float gradY = 0.5f * (pressure[idx3d(x, yTop, z, width, height)] - pressure[idx3d(x, yBottom, z, width, height)]);
    float gradZ = 0.5f * (pressure[idx3d(x, y, zBack, width, height)] - pressure[idx3d(x, y, zFront, width, height)]);

    velocity[idx3d(x, y, z, width, height)].x -= gradX;
    velocity[idx3d(x, y, z, width, height)].y -= gradY;
    velocity[idx3d(x, y, z, width, height)].z -= gradZ;
}

/// @brief Enforces the no-slip boundary condition, which dictates that velocity equals zero on the boundaries
__global__ void noSlipKernel(float4* velocity, int width, int height, int depth) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x != 0 && x != width - 1 && y != 0 && y != height - 1 && z != 0 && z != depth - 1) return;

    velocity[idx3d(x, y, z, width, height)] = make_float4(0, 0, 0, 0);
}

void project(FluidFields& fields, int jacobiIterations) {
    dim3 threadsPerBlock = getThreadsPerBlock();
    dim3 blocksPerGrid = getBlocksPerGrid(fields.width, fields.height, fields.depth);

    divergenceKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.velocity[0], fields.divergence, fields.width,
                                                         fields.height, fields.depth);
    CHECK_CUDA(cudaGetLastError());

    for (int i = 0; i < jacobiIterations; i++) {
        jacobiKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.pressure[0], fields.pressure[1], fields.divergence,
                                                         fields.width, fields.height, fields.depth);
        std::swap(fields.pressure[0], fields.pressure[1]);
    }
    CHECK_CUDA(cudaGetLastError());

    // in-place is fine here
    subtractGradientKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.velocity[0], fields.pressure[0], fields.width,
                                                               fields.height, fields.depth);

    // again, in-place is fine
    noSlipKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.velocity[0], fields.width, fields.height, fields.depth);
}