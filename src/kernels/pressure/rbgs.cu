#include "rbgs.h"

__device__ int idxRB(int x, int y, int z, int width, int height) { return idx3d(x / 2, y, z, width / 2, height); }

/// @brief more advanced rbgs implementation with fully coalesced memory accesses
__global__ void rbgsKernel_coalesced(float* pressureRed, float* pressureBlack, const float* divergence, int width,
                                     int height, int depth, int parity) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= width / 2 || y >= height || z >= depth) return;

    int x = 2 * tx + (y + z + parity) % 2;  // either 2*tx+0 or 2*tx+1, depending on row and slice and parity

    float* neighborList = (parity == 0) ? pressureBlack : pressureRed;

    int xLeft = max(x - 1, 0), xRight = min(x + 1, width - 1);
    int yBottom = max(y - 1, 0), yTop = min(y + 1, height - 1);
    int zFront = max(z - 1, 0), zBack = min(z + 1, depth - 1);

    float p = (1 / 6.0f) *
              (neighborList[idxRB(xLeft, y, z, width, height)] + neighborList[idxRB(xRight, y, z, width, height)] +
               neighborList[idxRB(x, yBottom, z, width, height)] + neighborList[idxRB(x, yTop, z, width, height)] +
               neighborList[idxRB(x, y, zFront, width, height)] + neighborList[idxRB(x, y, zBack, width, height)] -
               divergence[idx3d(x, y, z, width, height)]);

    float* self = (parity == 0) ? pressureRed : pressureBlack;
    self[idxRB(x, y, z, width, height)] = p;
}

__global__ void mergeRedBlackPressureKernel(float* pressure, const float* pressureRed, const float* pressureBlack,
                                            int width, int height, int depth) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    const float* src = ((x + y + z) % 2 == 0) ? pressureRed : pressureBlack;
    pressure[idx3d(x, y, z, width, height)] = src[idxRB(x, y, z, width, height)];
}

/// @brief very simple rbgs solution, disregarding any coalescing concepts
__global__ void rbgsKernel_simple(float* pressure, const float* divergence, int width, int height, int depth,
                                  int parity) {
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

void computePressure_rbgs(FluidFields& fields, int iterationCount) {
    if (fields.width % 2 != 0) {
        std::cerr << "Error: RBGS requires even width" << std::endl;
        return;
    }
    
    dim3 threadsPerBlock = getThreadsPerBlock();
    dim3 blocksPerGrid = getBlocksPerGrid(fields.width, fields.height, fields.depth);
    dim3 blocksPerGridRB = getBlocksPerGrid(fields.width / 2, fields.height, fields.depth);

    iterationCount /= 2;  // since convergence rate is roughly double of standard Jacobi iterations

    for (int i = 0; i < iterationCount; i++) {
        // rbgsKernel_simple<<<blocksPerGrid, threadsPerBlock>>>(fields.pressure[0], fields.divergence, fields.width,
        //                                                       fields.height, fields.depth, 0);
        // rbgsKernel_simple<<<blocksPerGrid, threadsPerBlock>>>(fields.pressure[0], fields.divergence, fields.width,
        //                                                       fields.height, fields.depth, 1);
        rbgsKernel_coalesced<<<blocksPerGridRB, threadsPerBlock>>>(
            fields.pressureRed, fields.pressureBlack, fields.divergence, fields.width, fields.height, fields.depth, 0);
        rbgsKernel_coalesced<<<blocksPerGridRB, threadsPerBlock>>>(
            fields.pressureRed, fields.pressureBlack, fields.divergence, fields.width, fields.height, fields.depth, 1);
    }
    CHECK_CUDA(cudaGetLastError());

    mergeRedBlackPressureKernel<<<blocksPerGrid, threadsPerBlock>>>(
        fields.pressure[0], fields.pressureRed, fields.pressureBlack, fields.width, fields.height, fields.depth);
}