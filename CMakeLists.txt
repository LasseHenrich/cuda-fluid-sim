cmake_minimum_required(VERSION 3.10)
project(CudaFluidSim LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
find_package(OpenGL REQUIRED)

pkg_check_modules(GLFW REQUIRED glfw3)
pkg_check_modules(GLEW REQUIRED glew)

add_executable(cuda_fluid_sim src/main.cpp)

target_include_directories(cuda_fluid_sim PRIVATE
    ${OpenGL_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
)

target_link_libraries(cuda_fluid_sim PRIVATE
    ${OpenGL_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${GLEW_LIBRARIES}
)