#pragma once
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

void registerTexture(unsigned int glTextureId, cudaGraphicsResource** cudaPBO);
void unregisterTexture(cudaGraphicsResource* cudaPBO);
void runCudaKernel(cudaGraphicsResource* cudaPBO, int widht, int height,
                   float time);