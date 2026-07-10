#pragma once
#include "helper.h"
#include "kernels/fluid.h"

void jacobiIteration_tiling(FluidFields& fields, int iterationCount);