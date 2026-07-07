cmake_minimum_required(VERSION 3.18)
project(CudaFluidSim LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native) # with this, nvcc compiles for your actual GPU instead of a default fallback

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glew REQUIRED)

add_executable(cuda_fluid_sim)

target_sources(cuda_fluid_sim PRIVATE
    src/kernels/advection.cu
    src/kernels/advection.h
    src/kernels/fluid.cu
    src/kernels/fluid.h
    src/kernels/plasmaKernel.h
    src/kernels/plasmaKernel.cu
    src/kernels/projection.h
    src/kernels/projection.cu
    src/helper.h
    src/interop.h
    src/interop.cu
    src/main.cpp
    src/timing.h
)

target_include_directories(cuda_fluid_sim PRIVATE src)

target_link_libraries(cuda_fluid_sim PRIVATE
    glfw
    GLEW::GLEW
    CUDA::cudart # cuda runtime
    OpenGL::GL
)
