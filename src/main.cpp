#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "interop.h"
#include "kernels/advection.h"
#include "kernels/fluid.h"
#include "kernels/projection.h"
#include "kernels/render.h"
#include "timing.h"

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

const int GRID_WIDTH = 128;
const int GRID_HEIGHT = 128;
const int GRID_DEPTH = 128;

const float FORCE_SCALE = 1.0f;

enum RenderMode { SLICE, RAY_MARCHING };

const RenderMode RENDER_MODE = RAY_MARCHING;

// render mode: slice
const int SLICE_Z = GRID_DEPTH / 2;

// render mode: ray marching
const float ROTATION_SPEED = 0.5f;
const float ORBIT_RADIUS = 200.0f;
const int RENDER_WIDTH = 512;
const int RENDER_HEIGHT = 512;

void processInput(GLFWwindow* window, FluidFields& fields) {
    // The last reported state for every key is saved in per-window state arrays and can be polled with glfwGetKey
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    static int prevGridX = 0, prevGridY = 0;
    static bool wasPressed = false;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        // screen space, so in interval [0, WINDOW_WIDTH-1] and [0, WINDOW_HEIGHT-1] resp.
        double screenX, screenY;
        glfwGetCursorPos(window, &screenX, &screenY);

        // convert from screen to grid space for calculation
        int gridX = int((screenX / WINDOW_WIDTH) * GRID_WIDTH);
        int gridY =
            int((1.0 - screenY / WINDOW_HEIGHT) * GRID_HEIGHT);  // flip, since screen space starts in top-left corner

        injectDyeAtPoint(fields, gridX, gridY, SLICE_Z);

        if (wasPressed) {
            // Todo: Make force strength dependent on deltaTime
            float3 force = make_float3((gridX - prevGridX) * FORCE_SCALE, (gridY - prevGridY) * FORCE_SCALE, 0);
            injectForceAtPoint(fields, gridX, gridY, SLICE_Z, force);
        }

        prevGridX = gridX;
        prevGridY = gridY;
        wasPressed = true;
    } else {
        wasPressed = false;
    }
}

void simulateStep(FluidFields& fields, float deltaTime, CudaTimer& velAdvectionTimer, CudaTimer& projectionTimer,
                  CudaTimer& dyeAdvectionTimer) {
    velAdvectionTimer.startTimer();
    advectVelocity(fields, deltaTime);
    velAdvectionTimer.endTimer();

    projectionTimer.startTimer();
    project(fields);
    projectionTimer.endTimer();

    dyeAdvectionTimer.startTimer();
    advectDye(fields, deltaTime);
    dyeAdvectionTimer.endTimer();
}

void render(FluidFields& fields, cudaGraphicsResource* glTextureCudaHandle, GLuint glTexture, GLuint blitFBO,
            float time, CudaTimer& renderingTimer) {
    cudaSurfaceObject_t surface = mapTextureSurface(glTextureCudaHandle);
    renderingTimer.startTimer();

    if (RENDER_MODE == RenderMode::SLICE) {
        renderSlice(fields, surface, RENDER_WIDTH, RENDER_HEIGHT, SLICE_Z);
    } else if (RENDER_MODE == RenderMode::RAY_MARCHING) {
        float angle = ROTATION_SPEED * time;
        float centerX = 0.5f * GRID_WIDTH, centerY = 0.5f * GRID_HEIGHT, centerZ = 0.5f * GRID_DEPTH;
        float3 forward = make_float3(-sinf(angle), 0.0f, -cosf(angle));
        float3 camPos = make_float3(centerX - ORBIT_RADIUS * forward.x, centerY, centerZ - ORBIT_RADIUS * forward.z);
        float3 right = make_float3(cosf(angle), 0.0f, -sinf(angle));
        float3 up = make_float3(0.0f, 1.0f, 0.0f);

        renderVolume(fields, surface, RENDER_WIDTH, RENDER_HEIGHT, camPos, forward, right, up);
    } else {
        std::cerr << "Invalid render mode" << std::endl;
    }

    renderingTimer.endTimer();
    unmapTextureSurface(glTextureCudaHandle, surface);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, blitFBO);  // prepare fbo for blitting / read operation
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, glTexture,
                           0);  // attach CUDA-modified texture into container's reading slot

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);  // active window screen as draw destination
    glBlitFramebuffer(0, 0, RENDER_WIDTH, RENDER_HEIGHT, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_COLOR_BUFFER_BIT,
                      GL_NEAREST);  // Blitting: Copies pixels from read destination to draw destination
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
    glfwSwapInterval(1);             // vsync

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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, RENDER_WIDTH, RENDER_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 NULL);  // allocate empty, uninitialized VRAM memory on the gpu
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  // typical minification filter
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  // typical magnification filter
    glBindTexture(GL_TEXTURE_2D, 0);  // unbind texture to not accidentally modify later

    // (frame buffer object) off-screen render target wrapper that we'll use as a helper to read pixel data
    GLuint blitFBO;
    glGenFramebuffers(1, &blitFBO);

    cudaGraphicsResource* glTextureCudaHandle;  // "memory translation bridge" for CUDA to access OpenGL VRAM block
    registerTexture(glTexture, &glTextureCudaHandle);

    FluidFields fields = allocateFields(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH);
    initVortex(fields);

    float time_prev = (float)glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        float time = (float)glfwGetTime();
        float deltaTime =
            fminf(time - time_prev,
                  0.05f);  // clamp at at least 20 FPS to prevent accuracy jumps, e.g. after moving the window
        time_prev = time;

        static CudaTimer velAdvectionTimer, projectionTimer, dyeAdvectionTimer, renderingTimer;

        // 1. INPUT PROCESSING
        processInput(window, fields);

        // 2. SIMULATION
        simulateStep(fields, deltaTime, velAdvectionTimer, projectionTimer, dyeAdvectionTimer);

        // 3. RENDERING
        render(fields, glTextureCudaHandle, glTexture, blitFBO, time, renderingTimer);

        // print only occasionally, since the synching introduces overhead
        static int frameCount = 0;
        if (frameCount++ % 50 == 0) {
            printf("velocity advection %.3f ms | projection %.3f ms | dye advection %.3f ms | rendering %.3f ms\n",
                   velAdvectionTimer.elapsedTimeInMs(), projectionTimer.elapsedTimeInMs(),
                   dyeAdvectionTimer.elapsedTimeInMs(), renderingTimer.elapsedTimeInMs());
        }

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