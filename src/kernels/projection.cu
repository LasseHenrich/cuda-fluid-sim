#include "helper.h"
#include "projection.h"

const int TILE_SIZE = 10;             // with halo
const int OUT_SIZE = TILE_SIZE - 2;  // interior cells

/// @brief Computes the change of the velocity field using finite differences
__global__ void divergenceKernel(const float4* velocity, float* divergence, int width, int height, int depth) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    // boundary condition enforcement: Set to the same cell, so zero gradient (Neumann)
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

    __shared__ float tile[TILE_SIZE * TILE_SIZE * TILE_SIZE];

    // thread coordinates within block/tile (incl. halo), range [0, TILE_SIZE)
    int tx_block = threadIdx.x, ty_block = threadIdx.y, tz_block = threadIdx.z;

    // thread coordinates within global grid, range [blockIdx * OUT_SIZE - 1, blockIdx * OUT_SIZE - 1 + TILE_SIZE)
    int tx_global = blockIdx.x * OUT_SIZE + tx_block - 1;
    int ty_global = blockIdx.y * OUT_SIZE + ty_block - 1;
    int tz_global = blockIdx.z * OUT_SIZE + tz_block - 1;

    // again enforcing boundary conditions
    int tx_global_clamped = min(max(tx_global, 0), width - 1);
    int ty_global_clamped = min(max(ty_global, 0), height - 1);
    int tz_global_clamped = min(max(tz_global, 0), depth - 1);

    tile[idx3d(tx_block, ty_block, tz_block, TILE_SIZE, TILE_SIZE)] =
        pressureIn[idx3d(tx_global_clamped, ty_global_clamped, tz_global_clamped, width, height)];

    __syncthreads();

    bool inHalo = tx_block == 0 || tx_block == TILE_SIZE - 1 || ty_block == 0 || ty_block == TILE_SIZE - 1 ||
                  tz_block == 0 || tz_block == TILE_SIZE - 1;

    if (inHalo || tx_global >= width || ty_global >= height || tz_global >= depth) return;

    float pLeft = tile[idx3d(tx_block - 1, ty_block, tz_block, TILE_SIZE, TILE_SIZE)];
    float pRight = tile[idx3d(tx_block + 1, ty_block, tz_block, TILE_SIZE, TILE_SIZE)];
    float pBottom = tile[idx3d(tx_block, ty_block - 1, tz_block, TILE_SIZE, TILE_SIZE)];
    float pTop = tile[idx3d(tx_block, ty_block + 1, tz_block, TILE_SIZE, TILE_SIZE)];
    float pFront = tile[idx3d(tx_block, ty_block, tz_block - 1, TILE_SIZE, TILE_SIZE)];
    float pBack = tile[idx3d(tx_block, ty_block, tz_block + 1, TILE_SIZE, TILE_SIZE)];

    float div = divergence[idx3d(tx_global, ty_global, tz_global, width, height)];

    float p = (1.0f / 6.0f) * (pLeft + pRight + pBottom + pTop + pFront + pBack - div);

    pressureOut[idx3d(tx_global, ty_global, tz_global, width, height)] = p;
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
        dim3 j_threadsPerBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
        dim3 j_blocksPerGrid((fields.width + OUT_SIZE - 1) / OUT_SIZE, (fields.height + OUT_SIZE - 1) / OUT_SIZE,
                             (fields.depth + OUT_SIZE - 1) / OUT_SIZE);
        jacobiKernel<<<j_blocksPerGrid, j_threadsPerBlock>>>(fields.pressure[0], fields.pressure[1], fields.divergence,
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