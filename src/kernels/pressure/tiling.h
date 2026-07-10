#pragma once
#include "helper.h"
#include "kernels/fluid.h"

void computePressure_tiling(FluidFields& fields, int iterationCount);