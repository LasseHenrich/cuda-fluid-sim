/// This is purely for testing

#pragma once
#include <cuda_runtime.h>

void runPlasmaKernel(cudaSurfaceObject_t surface, int width, int height, int depth, float time);