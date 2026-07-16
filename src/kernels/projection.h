#pragma once
#include "fluid.h"
#include "helper.h"

void project(FluidFields& fields, int jacobiIterationCount, PressureEvalMode pressureEvalMode);