#include "slab.h"

const int TILE_WIDTH = 32;
const int TILE_HEIGHT = 8;

const int OUT_WIDTH = TILE_WIDTH - 2;
const int OUT_HEIGHT = TILE_HEIGHT - 2;

__global__ void jacobiKernel_slab(const float* pressureIn, float* pressureOut, const float* divergence, int width,
                                  int height, int depth) {
    __shared__ float tile[TILE_HEIGHT * TILE_WIDTH];

    int tx_block = threadIdx.x, ty_block = threadIdx.y;

    // thread coordinates within global grid, range [blockIdx * OUT_SIZE - 1, blockIdx * OUT_SIZE - 1 + TILE_SIZE)
    int tx_global = blockIdx.x * OUT_WIDTH + tx_block - 1;
    int ty_global = blockIdx.y * OUT_HEIGHT + ty_block - 1;

    // again enforcing boundary conditions
    int tx_global_clamped = min(max(tx_global, 0), width - 1);
    int ty_global_clamped = min(max(ty_global, 0), height - 1);

    float pFront = pressureIn[idx3d(tx_global_clamped, ty_global_clamped, 0, width, height)];
    float pSelf = pFront;  // pSelf is the cell within the loop of which we currently calculate the value

    bool inHalo = tx_block == 0 || tx_block == TILE_WIDTH - 1 || ty_block == 0 || ty_block == TILE_HEIGHT - 1;
    bool outOfRange = tx_global >= width || ty_global >= height;

    for (int z = 0; z < depth; z++) {
        int zBack = min(z + 1, depth - 1);
        float pBack = pressureIn[idx3d(tx_global_clamped, ty_global_clamped, zBack, width, height)];

        tile[ty_block * TILE_WIDTH + tx_block] = pSelf;
        __syncthreads();

        if (!inHalo && !outOfRange) {
            float pLeft = tile[ty_block * TILE_WIDTH + tx_block - 1];
            float pRight = tile[ty_block * TILE_WIDTH + tx_block + 1];
            float pBottom = tile[(ty_block - 1) * TILE_WIDTH + tx_block];
            float pTop = tile[(ty_block + 1) * TILE_WIDTH + tx_block];

            float div = divergence[idx3d(tx_global, ty_global, z, width, height)];

            float p = (1.0f / 6.0f) * (pLeft + pRight + pBottom + pTop + pFront + pBack - div);

            pressureOut[idx3d(tx_global, ty_global, z, width, height)] = p;
        }

        __syncthreads();

        pFront = pSelf;
        pSelf = pBack;
    }
}

void computePressure_slab(FluidFields& fields, int iterationCount) {
    dim3 threadsPerBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 blocksPerGrid((fields.width + OUT_WIDTH - 1) / OUT_WIDTH, (fields.height + OUT_HEIGHT - 1) / OUT_HEIGHT);
    for (int i = 0; i < iterationCount; i++) {
        jacobiKernel_slab<<<blocksPerGrid, threadsPerBlock>>>(fields.pressure[0], fields.pressure[1], fields.divergence,
                                                             fields.width, fields.height, fields.depth);
        std::swap(fields.pressure[0], fields.pressure[1]);
    }
    CHECK_CUDA(cudaGetLastError());
}