#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "interop.h"
#include "kernels/advection.h"
#include "kernels/fluid.h"
#include "kernels/plasmaKernel.h"

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

const int GRID_WIDTH = 512;
const int GRID_HEIGHT = 512;

void processInput(GLFWwindow* window, FluidFields& fields) {
    // The last reported state for every key is saved in per-window state arrays and can be polled with glfwGetKey
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        // screen space, so in interval [0, WINDOW_WIDTH-1] and [0, WINDOW_HEIGHT-1] resp.
        double screenX, screenY;
        glfwGetCursorPos(window, &screenX, &screenY);

        // convert from screen to grid space for calculation
        int gridX = int((screenX / WINDOW_WIDTH) * GRID_WIDTH);
        int gridY =
            int((1.0 - screenY / WINDOW_HEIGHT) * GRID_HEIGHT);  // flip, since screen space starts in top-left corner

        injectDyeAtPoint(fields, gridX, gridY);

        // ToDo: Store prevX/prevY across frames to update the velocity field later
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
    initVortex(fields);

    float lastTime = (float)glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        float time = (float)glfwGetTime();
        float deltaTime =
            fminf(time - lastTime,
                  0.05f);  // clamp at at least 20 FPS to prevent accuracy jumps, e.g. after moving the window
        lastTime = time;

        processInput(window, fields);
        advectDye(fields, deltaTime);

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