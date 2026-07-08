#pragma once
#include <cuda_runtime.h>

#include "fluid.h"

/// @brief Calls a kernel to convert dye to color and write it to the surface texture. Renders the specified slice along
/// the z-axis, not 3D.
void renderSlice(FluidFields& fields, cudaSurfaceObject_t surface, int renderWidth, int renderHeight, int sliceZ);

/// @brief Calls a kernel to perform orthographic ray marching through the fluid volume and write the resulting color to the surface texture.
void renderVolume(FluidFields& fields, cudaSurfaceObject_t surface, int renderWidth, int renderHeight, float3 camPos,
                  float3 forward, float3 right, float3 up);