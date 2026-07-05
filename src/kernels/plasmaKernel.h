#pragma once
#include <cuda_runtime.h>

void runCudaKernel(cudaGraphicsResource* glTextureCudaHandle, int width, int height, float time);