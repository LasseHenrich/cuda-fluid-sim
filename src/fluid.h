#pragma once
#include <cuda_runtime.h>

/// @brief All fields used for simulation, most stored on device.
/// Size GRID_WIDTH x GRID_HEIGHT.
/// Indexing row-major (idx = y * width + x).
/// Pairs of two for ping-ponging.
struct FluidFields {
    float2* velocity[2];  // DEVICE
    float* dye[2];  // DEVICE. concentration of dye carried by the fluid at a point, used to calculate rendered color
    float* pressure[2];  // DEVICE
    int width, height;   // HOST
};

/// @brief Allocate a FluidFields object on device (incl. setting width and height)
FluidFields allocateFields(int width, int height);

/// @brief Free the FluidFields object
void freeFields(FluidFields& fields);

void seedDye(FluidFields& f);  // temporary: fill a test pattern until mouse splat exists

/// @brief Call a kernel to convert dye to color and write it to the surface texture
void renderDye(FluidFields& fields, cudaSurfaceObject_t surface);