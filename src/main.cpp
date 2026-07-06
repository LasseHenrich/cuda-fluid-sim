#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "interop.h"
#include "kernels/plasmaKernel.h"
#include "fluid.h"

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

const int GRID_WIDTH = 512;
const int GRID_HEIGHT = 512;

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CUDA Fluid Simulator", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);  // all rendering commands issued by this thread to be drawn
                                     // inside this window's canvas

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to init GLEW" << std::endl;
        return -1;
    }

    // allocate graphics memory
    GLuint glTexture;  // id for gpu texture
    glGenTextures(1, &glTexture);
    glBindTexture(GL_TEXTURE_2D,
                  glTexture);  // telling OpenGL that any altered settings
                               // should apply to this texture id
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, GRID_WIDTH, GRID_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 NULL);  // allocate empty, uninitialized VRAM memory on the gpu
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  // typical minification filter
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  // typical magnification filter
    glBindTexture(GL_TEXTURE_2D, 0);  // unbind texture to not accidentally modify later

    // (frame buffer object) off-screen render target wrapper that we'll use as a helper to read pixel data
    GLuint blitFBO;
    glGenFramebuffers(1, &blitFBO);

    cudaGraphicsResource* glTextureCudaHandle;  // "memory translation bridge" for CUDA to access OpenGL VRAM block
    registerTexture(glTexture, &glTextureCudaHandle);

    FluidFields fields = allocateFields(GRID_WIDTH, GRID_HEIGHT);
    seedDye(fields);

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        float time = (float)glfwGetTime();

        cudaSurfaceObject_t surface = mapTextureSurface(glTextureCudaHandle);
        // runPlasmaKernel(surface, GRID_WIDTH, GRID_HEIGHT, time);
        renderDye(fields, surface);
        unmapTextureSurface(glTextureCudaHandle, surface);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, blitFBO);  // prepare fbo for blitting / read operation
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, glTexture,
                               0);  // attach CUDA-modified texture into container's reading slot

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);  // active window screen as draw destination
        glBlitFramebuffer(0, 0, GRID_WIDTH, GRID_HEIGHT, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_COLOR_BUFFER_BIT,
                          GL_NEAREST);  // Blitting: Copies pixels from read destination to draw destination

        glfwSwapBuffers(window);  // prepare next frame
        glfwPollEvents();         // check any OS operations
    }

    // cleanup
    freeFields(fields);
    unregisterTexture(glTextureCudaHandle);
    glDeleteTextures(1, &glTexture);
    glDeleteFramebuffers(1, &blitFBO);
    glfwTerminate();

    return 0;
}