#include "rbgs.h"

__global__ void rgbsKernel(float* pressure, const float* divergence, int width, int height, int depth, int parity) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;
    if ((x + y + z) % 2 != parity) return;

    int xLeft = max(x - 1, 0), xRight = min(x + 1, width - 1);
    int yBottom = max(y - 1, 0), yTop = min(y + 1, height - 1);
    int zFront = max(z - 1, 0), zBack = min(z + 1, depth - 1);

    float p = (1 / 6.0f) * (pressure[idx3d(xLeft, y, z, width, height)] + pressure[idx3d(xRight, y, z, width, height)] +
                            pressure[idx3d(x, yBottom, z, width, height)] + pressure[idx3d(x, yTop, z, width, height)] +
                            pressure[idx3d(x, y, zFront, width, height)] + pressure[idx3d(x, y, zBack, width, height)] -
                            divergence[idx3d(x, y, z, width, height)]);

    pressure[idx3d(x, y, z, width, height)] = p;
}

void jacobiIteration_rgbs(FluidFields& fields, int iterationCount) {
    dim3 threadsPerBlock = getThreadsPerBlock();
    dim3 blocksPerGrid = getBlocksPerGrid(fields.width, fields.height, fields.depth);

    iterationCount /= 2;  // since convergence rate is roughly double of standard Jacobi iterations

    for (int i = 0; i < iterationCount; i++) {
        rgbsKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.pressure[0], fields.divergence, fields.width,
                                                       fields.height, fields.depth, 0);
        rgbsKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.pressure[0], fields.divergence, fields.width,
                                                       fields.height, fields.depth, 1);
    }
    CHECK_CUDA(cudaGetLastError());
}