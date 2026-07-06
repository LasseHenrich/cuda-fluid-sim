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

cudaSurfaceObject_t mapTextureSurface(cudaGraphicsResource* handle) {
    CHECK_CUDA(cudaGraphicsMapResources(1, &handle));  // locks texture and handing ownership to cuda

    cudaArray_t textureArray;
    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&textureArray, handle, 0, 0));

    cudaResourceDesc resDesc;  // resource descriptor
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    cudaSurfaceObject_t surface;
    CHECK_CUDA(cudaCreateSurfaceObject(&surface, &resDesc));
    return surface;
}

void unmapTextureSurface(cudaGraphicsResource* handle, cudaSurfaceObject_t surface) {
    CHECK_CUDA(cudaDestroySurfaceObject(surface));
    CHECK_CUDA(cudaGraphicsUnmapResources(1, &handle));
}
