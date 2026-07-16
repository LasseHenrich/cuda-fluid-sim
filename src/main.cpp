#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <iostream>

#include "interop.h"
#include "kernels/advection.h"
#include "kernels/fluid.h"
#include "kernels/projection.h"
#include "kernels/render.h"
#include "timing.h"

const int WINDOW_WIDTH = 1000;
const int WINDOW_HEIGHT = 600;
const int SIM_VIEW_OFFSET_X = 200;  // offset for simulation rendering
const int SIM_WINDOW_WIDTH = WINDOW_WIDTH - SIM_VIEW_OFFSET_X;

const int GRID_WIDTH = 128;
const int GRID_HEIGHT = 128;
const int GRID_DEPTH = 128;

bool enableForceInjection = false;
int forceRadius = 8;
float forceScale = 50.0f;

bool enableDyeInjection = true;
int dyeStrokeRadius = 8;
float dyeStrokeStrength = 2.0f;

enum RenderMode { SLICE, RAY_MARCHING };
RenderMode renderMode = RenderMode::SLICE;

// render mode: slice
int sliceZ = GRID_DEPTH / 2;

// render mode: ray marching
const float ROTATION_SPEED = 0.5f;
const float ORBIT_RADIUS = 200.0f;
const int RENDER_WIDTH = 512;
const int RENDER_HEIGHT = 512;
float viewSizeMultiplier = 1.7f;

// jacobi iteration for pressure evaluation
PressureEvalMode pressureEvalMode = PressureEvalMode::SIMPLE;
int jacobiIterationCount = 40;

void processGUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    int windowPadding = 0;
    ImGui::SetNextWindowPos(ImVec2(windowPadding, windowPadding));
    ImGui::SetNextWindowSize(ImVec2(SIM_VIEW_OFFSET_X - 2 * windowPadding, WINDOW_HEIGHT - 2 * windowPadding));

    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);

    ImGui::SeparatorText("Scene Controls");

    const char* renderModeNames[] = {"Slice", "Ray Marching"};
    int currentRenderMode = (int)renderMode;
    if (ImGui::Combo("Render Mode", &currentRenderMode, renderModeNames, IM_ARRAYSIZE(renderModeNames))) {
        renderMode = (RenderMode)currentRenderMode;
    }

    if (renderMode == RenderMode::RAY_MARCHING) {
        ImGui::SliderFloat("view size", &viewSizeMultiplier, 1.0f, 2.0f);
    }

    // show regardless of render mode, since it is also used by the injections
    ImGui::SliderInt("Slice Z", &sliceZ, 0, GRID_DEPTH - 1);

    ImGui::Checkbox("Enable Force Injection", &enableForceInjection);
    if (enableForceInjection) {
        ImGui::SliderInt("Force Radius", &forceRadius, 5, 30);

        int _scale = forceScale * 0.1f;
        ImGui::SliderInt("Force Scale", &_scale, 1, 10);
        forceScale = _scale * 10;
    }

    ImGui::Checkbox("Enable Dye Injection", &enableDyeInjection);
    if (enableDyeInjection) {
        ImGui::SliderInt("Stroke Radius", &dyeStrokeRadius, 2, 16);

        int _strength = dyeStrokeStrength * 2;
        ImGui::SliderInt("Stroke Strength", &_strength, 1, 10);
        dyeStrokeStrength = _strength * 0.5f;
    }

    ImGui::SeparatorText("Algorithm Controls");

    ImGui::SliderInt("Jacobi Iterations", &jacobiIterationCount, 20, 100);

    const char* jacobiEvalModeNames[] = {"Simple", "Tiling", "Slab", "Red-Black Gauss Seidel"};
    int currentJacobiEvalMode = (int)pressureEvalMode;
    if (ImGui::Combo("Jacobi Evaluation Mode", &currentJacobiEvalMode, jacobiEvalModeNames,
                     IM_ARRAYSIZE(jacobiEvalModeNames))) {
        pressureEvalMode = (PressureEvalMode)currentJacobiEvalMode;
        // Todo: When switching to rbgs, we should initialize pressureRed and pressureBlack from pressure as a warm
        // start for the search
    }

    ImGui::End();
}

void processInput(GLFWwindow* window, FluidFields& fields, float deltaTime) {
    // The last reported state for every key is saved in per-window state arrays and can be polled with glfwGetKey
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    static int prevGridX = 0, prevGridY = 0;
    static bool wasPressed = false;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS &&
        !ImGui::GetIO().WantCaptureMouse) {  // only if not interacting with GUI
        // screen space, so in interval [0, WINDOW_WIDTH-1] and [0, WINDOW_HEIGHT-1] resp.
        double screenX, screenY;
        glfwGetCursorPos(window, &screenX, &screenY);

        // convert from screen to grid space for calculation
        int gridX = int(((screenX - SIM_VIEW_OFFSET_X) / SIM_WINDOW_WIDTH) * GRID_WIDTH);
        int gridY =
            int((1.0 - screenY / WINDOW_HEIGHT) * GRID_HEIGHT);  // flip, since screen space starts in top-left corner

        gridX = std::min(std::max(gridX, 0), GRID_WIDTH);
        gridY = std::min(std::max(gridY, 0), GRID_HEIGHT);

        if (enableDyeInjection) {
            injectDyeAtPoint(fields, gridX, gridY, sliceZ, dyeStrokeRadius, dyeStrokeStrength * deltaTime);
        }

        if (wasPressed && enableForceInjection) {
            float3 force = make_float3((gridX - prevGridX) * forceScale * deltaTime,
                                       (gridY - prevGridY) * forceScale * deltaTime, 0);
            injectForceAtPoint(fields, gridX, gridY, sliceZ, forceRadius, force);
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
    project(fields, jacobiIterationCount, pressureEvalMode);
    projectionTimer.endTimer();

    dyeAdvectionTimer.startTimer();
    advectDye(fields, deltaTime);
    dyeAdvectionTimer.endTimer();
}

void renderSim(FluidFields& fields, cudaGraphicsResource* glTextureCudaHandle, GLuint glTexture, GLuint blitFBO,
               float time, CudaTimer& renderingTimer) {
    cudaSurfaceObject_t surface = mapTextureSurface(glTextureCudaHandle);

    renderingTimer.startTimer();

    if (renderMode == RenderMode::SLICE) {
        renderSlice(fields, surface, RENDER_WIDTH, RENDER_HEIGHT, sliceZ);
    } else if (renderMode == RenderMode::RAY_MARCHING) {
        float angle = ROTATION_SPEED * time;
        float centerX = 0.5f * GRID_WIDTH, centerY = 0.5f * GRID_HEIGHT, centerZ = 0.5f * GRID_DEPTH;
        float3 forward = make_float3(-sinf(angle), -0.4f, -cosf(angle));
        float3 camPos = make_float3(centerX - ORBIT_RADIUS * forward.x, centerY + GRID_HEIGHT * 0.65f,
                                    centerZ - ORBIT_RADIUS * forward.z);
        float3 right = make_float3(cosf(angle), 0.0f, -sinf(angle));
        float3 up = make_float3(0.0f, 1.0f, 0.0f);

        renderVolume(fields, surface, RENDER_WIDTH, RENDER_HEIGHT, camPos, forward, right, up, viewSizeMultiplier);
    } else {
        std::cerr << "Invalid render mode" << std::endl;
    }

    renderingTimer.endTimer();
    unmapTextureSurface(glTextureCudaHandle, surface);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, blitFBO);  // prepare fbo for blitting / read operation
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, glTexture,
                           0);  // attach CUDA-modified texture into container's reading slot

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);  // active window screen as draw destination
    glBlitFramebuffer(0, 0, RENDER_WIDTH, RENDER_HEIGHT, SIM_VIEW_OFFSET_X, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                      GL_COLOR_BUFFER_BIT,
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

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);  // init window and input handling
    ImGui_ImplOpenGL3_Init("#version 330");      // init OpenGL rendering backend

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

        // 1. GUI CREATION AND PROCESSING
        processGUI();

        // 2. INPUT PROCESSING
        processInput(window, fields, deltaTime);

        // 3. SIMULATION
        simulateStep(fields, deltaTime, velAdvectionTimer, projectionTimer, dyeAdvectionTimer);

        // 4. SIMULATION RENDERING
        renderSim(fields, glTextureCudaHandle, glTexture, blitFBO, time, renderingTimer);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

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

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();

    return 0;
}