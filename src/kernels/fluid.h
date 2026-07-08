#pragma once
#include <cuda_runtime.h>

/// @brief All fields used for simulation, most stored on device.
/// Size GRID_WIDTH x GRID_HEIGHT.
/// Indexing row-major (idx = y * width + x).
/// Pairs of two for ping-ponging.
struct FluidFields {
    // Note: velocity could be float3, but float4 is exactly one full transaction per thread (16 bytes), so one full transaction
    // ~> more natural for memory coalescing
    float4* velocity[2];  // DEVICE. Velocity of the fluid at a point
    float* dye[2];  // DEVICE. Concentration of dye carried by the fluid at a point, used to calculate rendered color
    float* pressure[2];  // DEVICE. Pressure of the fluid. Ping-ponging since the last frame's pressure is a very good
                         // initial guess for the current frame
    float* divergence;   // DEVICE. Net change of velocity. Computed anew each frame, no ping-pong needed.
    int width, height, depth;  // HOST
};

/// @brief Allocate a FluidFields object on device (incl. setting width and height)
FluidFields allocateFields(int width, int height, int depth);

/// @brief Free the FluidFields object
void freeFields(FluidFields& fields);

/// @brief Initialize the velocity field with a simple vortex around the z-axis
void initVortex(FluidFields& fields);

/// @brief Inject some dye at the passed position in grid space ~> changes dye field
void injectDyeAtPoint(FluidFields& fields, int x, int y, int z);

/// @brief Inject force at the passed position in grid space with the passed forced ~> changes velocity field
void injectForceAtPoint(FluidFields& fields, int x, int y, int z, float3 force);