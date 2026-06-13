> Ongoing _GPU Computing_ lecture project

# Real-Time 3D Fluid Simulator in CUDA

## Overview
I aim to implement a real-time 3D fluid simulator based on the _Stable Fluids_ framework. The simulator will solve the Navier-Stokes equations on the GPU -- this parallelization should make performance much better than computing the PDEs on a CPU. The calculated fluid state will also need to be rendered on the screen.

As a wrapper / core deliverable, my goal is to write an interactive application that allows for real-time force injection and smoke manipulation.

Note that my key reference for my 2D version of this is [this GPU guide](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu). Based on this, the implementation will showcase concepts such as stencil tiling, memory coalescing and generally balancing high compute and bandwith utilization.

## Possible scope expansions
If enough time, I'd like to tackle the following challenges:
1. Implement further __performance enhancements__ like a basic multigrid solver.
1. Since I will likely use something OpenGL (or something similar) for rendering, data shouldn't be copied back to the host CPU just to be again pushed to OpenGL for rendering ~> It should be possible to __map CUDA device memory arrays directly to OpenGL Vertex Buffers__.
1. __Benchmarking__ across various grid dimensions.
