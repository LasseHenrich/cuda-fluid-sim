#include "helper.h"
#include "interop.h"

/// @brief creates CUDA handle for the texture so CUDA kernels can write into it
void registerTexture(unsigned int glTexture, cudaGraphicsResource** glTextureCudaHandle) {
    CHECK_CUDA(cudaGraphicsGLRegisterImage(glTextureCudaHandle, glTexture, GL_TEXTURE_2D,
                                           cudaGraphicsRegisterFlagsWriteDiscard));
}

void unregisterTexture(cudaGraphicsResource* glTextureCudaHandle) {
    cudaGraphicsUnregisterResource(glTextureCudaHandle);
}