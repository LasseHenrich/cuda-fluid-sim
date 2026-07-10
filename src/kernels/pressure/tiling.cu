#include "tiling.h"

/// @brief One iteration of the Jacobi iteration to solve for pressure
__global__ void jacobiKernel_tiling(const float* pressureIn, float* pressureOut, const float* divergence, int width,
                             int height, int depth) {
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

void computePressure_tiling(FluidFields& fields, int iterationCount) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((fields.width + OUT_SIZE - 1) / OUT_SIZE, (fields.height + OUT_SIZE - 1) / OUT_SIZE,
                         (fields.depth + OUT_SIZE - 1) / OUT_SIZE);
    for (int i = 0; i < iterationCount; i++) {
        jacobiKernel_tiling<<<blocksPerGrid, threadsPerBlock>>>(fields.pressure[0], fields.pressure[1], fields.divergence,
                                                             fields.width, fields.height, fields.depth);
        std::swap(fields.pressure[0], fields.pressure[1]);
    }
    CHECK_CUDA(cudaGetLastError());
}