cmake_minimum_required(VERSION 3.10)
project(CudaFluidSim LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glew REQUIRED)

add_executable(cuda_fluid_sim src/main.cpp src/kernel.cu src/kernel.h)

target_include_directories(cuda_fluid_sim PRIVATE
    ${OpenGL_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(cuda_fluid_sim PRIVATE
    ${OpenGL_LIBRARIES}
    glfw
    GLEW::GLEW
    CUDA::cudart # cuda runtime
)