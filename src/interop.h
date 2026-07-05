#pragma once
#include <GL/glew.h> // must come before cuda_gl_interop.h (defines some keys on Windows)
#include <cuda_gl_interop.h>

void registerTexture(unsigned int glTexture, cudaGraphicsResource** glTextureCudaHandle);
void unregisterTexture(cudaGraphicsResource* glTextureCudaHandle);