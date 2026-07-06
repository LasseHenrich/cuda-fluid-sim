#pragma once
#include "fluid.h"

/// @brief Pull/advect dye along fluid's velocity field
void advectDye(FluidFields& fields, float deltaTime);

/// @brief Self-advection of the velocity field
void advectVelocity(FluidFields& fields, float deltaTime);