# Real-Time 3D Fluid Simulator in CUDA

## Overview
This is a real-time 3D fluid simulator based on the _Stable Fluids_ framework. The simulator solves the Navier-Stokes equations on the GPU, improving performance greatly over a CPU-based computation of the PDEs. The calculated fluid state is rendered on-screen with either a slice view or orthogonal ray-marching and a rotating camera. The application allows you to inject a fluid as well as force in real-time while exposing a GUI that let's you control various options about the injection, rendering and the projection algorithm. The implementation showcases GPU optimization concepts such as *Iterative Stencil Loops* (ISLs) via Tiling, memory coalescing and coarsening, the *Slab*/2.5D method and *Red-Black Gauss-Seidel* (RBGS).

[Docs.md](Docs.md) provides some theoretical and code documentation, [Measurements.md](measurements/Measurements.md) details benchmarks we conducted.

# Setup, Building and Running

## Windows
Make sure vcpkg is installed:
```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install glfw3 glew imgui[glfw-binding,opengl3-binding]
```

Create a build dir, configure, build, and run the code in PowerShell:
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
.\Release\cuda_fluid_sim.exe
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