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


# Setup, Building and Running

## Windows
Make sure vcpkg is installed:
```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install glfw3 glew
```

Create a build dir, configure, build, and run the code:
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
You might need to add `-DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"` to the configure command

## ~~WSL~~ (dropped support, didn't get it working on WSL)
Make sure the following packages are installed:
```bash
sudo apt install -y \
    build-essential \
    cmake \
    libgl1-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    pkg-config
```
Make sure `glxgears` is installed. If it's not, run
```bash
sudo apt install -y mesa-utils
```
Create a build dir, configure, build, and run the code:
```bash
mkdir build
cd build
cmake ..
make
./cuda_fluid_sim
```