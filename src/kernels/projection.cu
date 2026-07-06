#include "helper.h"
#include "projection.h"

const int JACOBI_ITERATIONS = 40;

/// @brief Computes the change of the velocity field using finite differences
__global__ void divergenceKernel(const float2* velocity, float* divergence, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int xLeft = max(x - 1, 0);
    int xRight = min(x + 1, width - 1);
    int yBottom = max(y - 1, 0);
    int yTop = min(y + 1, height - 1);

    float div = 0.5f * (velocity[y * width + xRight].x - velocity[y * width + xLeft].x) +
                0.5f * (velocity[yTop * width + x].y - velocity[yBottom * width + x].y);

    divergence[y * width + x] = div;
}

/// @brief One iteration of the Jacobi iteration to solve for pressure
__global__ void jacobiKernel(const float* pressureIn, float* pressureOut, const float* divergence, int width,
                             int height) {
    // Todo: Cleanup by merging shared logic with divergenceKernel

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int xLeft = max(x - 1, 0);
    int xRight = min(x + 1, width - 1);
    int yBottom = max(y - 1, 0);
    int yTop = min(y + 1, height - 1);

    float p = 0.25f * (pressureIn[y * width + xLeft] + pressureIn[y * width + xRight] +
                       pressureIn[yBottom * width + x] + pressureIn[yTop * width + x] - divergence[y * width + x]);

    pressureOut[y * width + x] = p;
}

/// @brief Subtracts the pressure gradient from the velocity, where the gradient is calculated using finite differences
__global__ void subtractGradientKernel(float2* velocity, const float* pressure, int width, int height) {
    // Todo: Cleanup by merging shared logic with divergenceKernel

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int xLeft = max(x - 1, 0);
    int xRight = min(x + 1, width - 1);
    int yBottom = max(y - 1, 0);
    int yTop = min(y + 1, height - 1);

    float gradX = 0.5f * (pressure[y * width + xRight] - pressure[y * width + xLeft]);
    float gradY = 0.5f * (pressure[yTop * width + x] - pressure[yBottom * width + x]);

    velocity[y * width + x].x -= gradX;
    velocity[y * width + x].y -= gradY;
}

/// @brief Enforces the no-slip boundary condition, which dictates that velocity equals zero on the boundaries
__global__ void noSlipKernel(float2* velocity, int width, int height) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x != 0 && x != width - 1 && y != 0 && y != height - 1) return;

    velocity[y * width + x] = make_float2(0, 0);
}

void project(FluidFields& fields) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((fields.width + 15) / 16, (fields.height + 15) / 16);

    divergenceKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.velocity[0], fields.divergence, fields.width,
                                                         fields.height);
    CHECK_CUDA(cudaGetLastError());

    for (int i = 0; i < JACOBI_ITERATIONS; i++) {
        jacobiKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.pressure[0], fields.pressure[1], fields.divergence,
                                                         fields.width, fields.height);
        std::swap(fields.pressure[0], fields.pressure[1]);
    }
    CHECK_CUDA(cudaGetLastError());

    // in-place is fine here
    subtractGradientKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.velocity[0], fields.pressure[0], fields.width,
                                                               fields.height);

    // again, in-place is fine
    noSlipKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.velocity[0], fields.width, fields.height);
}