#include "helper.h"
#include "render.h"

__global__ void renderSliceKernel(const float* dye, cudaSurfaceObject_t surface, int boxWidth, int boxHeight,
                                  int renderWidth, int renderHeight, int sliceZ) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= renderWidth || y >= renderHeight) return;

    int gridX = x * boxWidth / renderWidth;
    int gridY = y * boxHeight / renderHeight;

    float d = fminf(dye[idx3d(gridX, gridY, sliceZ, boxWidth, boxHeight)], 1.0f);  // clamp since dye can exceed 1
    unsigned char c = (unsigned char)(d * 255.0f);
    surf2Dwrite(make_uchar4(c, c, c, 255), surface, x * sizeof(uchar4), y);
}

void renderSlice(FluidFields& fields, cudaSurfaceObject_t surface, int renderWidth, int renderHeight, int sliceZ) {
    dim3 threadsPerBlock = dim3(16, 16);
    dim3 blocksPerGrid = dim3((renderWidth + 15) / 16, (renderHeight + 15) / 16);
    renderSliceKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.dye[0], surface, fields.width, fields.height,
                                                          renderWidth, renderHeight, sliceZ);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void renderVolumeKernel(const float* dye, cudaSurfaceObject_t surface, int boxWidth, int boxHeight,
                                   int boxDepth, int renderWidth, int renderHeight, float3 camPos, float3 forward,
                                   float3 right, float3 up, float viewSizeMultiplier) {
    int pixelX = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelY = blockDim.y * blockIdx.y + threadIdx.y;
    if (pixelX >= renderWidth || pixelY >= renderHeight) return;

    // orthographic projection, i.e., rays are spread across viewSize already at camera position and same direction
    // (forward) for every ray
    // ref. Kay's "Slab Method"

    const float viewSize = boxWidth * viewSizeMultiplier;

    // map pixel coordinates to "world space" (as defined by the box's bounds)
    float u = ((pixelX / (float)renderWidth) - 0.5f) * viewSize;  // [-viewSize/2, viewSize/2]
    float v = ((pixelY / (float)renderHeight) - 0.5f) * viewSize;

    // origin of the ray
    float originX = camPos.x + u * right.x + v * up.x;
    float originY = camPos.y + u * right.y + v * up.y;
    float originZ = camPos.z + u * right.z + v * up.z;

    // t distances along ray where it intersects min/max bounds of the box (i.e. 0 and boxWidth/Height/Depth)
    float tx0 = (0.0f - originX) / forward.x, tx1 = (boxWidth - originX) / forward.x;
    float ty0 = (0.0f - originY) / forward.y, ty1 = (boxHeight - originY) / forward.y;
    float tz0 = (0.0f - originZ) / forward.z, tz1 = (boxDepth - originZ) / forward.z;

    // close and far distances
    float tCloseX = fminf(tx0, tx1), tFarX = fmaxf(tx0, tx1);
    float tCloseY = fminf(ty0, ty1), tFarY = fmaxf(ty0, ty1);
    float tCloseZ = fminf(tz0, tz1), tFarZ = fmaxf(tz0, tz1);

    float tClose =
        fmaxf(fmaxf(tCloseX, tCloseY), tCloseZ);     // largest of minimum entry distances (ray entered all three slabs)
    float tFar = fminf(fminf(tFarX, tFarY), tFarZ);  // smallest of maximum exit distances (ray left first slab)

    float brightness = 0.0f;  // "result" value of the pixel
    float transmittance = 1.0f;

    if (tFar > fmaxf(tClose, 0.0f)) {  // max with 0 to handle the case where the far plane is behind the camera,
                                       // i.e. camera is in the box
        const float stepSize = 0.5f;
        const float absorption = 0.2f, gain = 0.2f;  // ToDo: adjust if needed

        for (float t = fmaxf(tClose, 0.0f); t < tFar && transmittance > 0.01f;
             t += stepSize) {  // when transmittance is very low, we can stop early ~> remaining samples won't add much

            // sample point in box
            int sampleX = min(max((int)(originX + t * forward.x), 0), boxWidth - 1);
            int sampleY = min(max((int)(originY + t * forward.y), 0), boxHeight - 1);
            int sampleZ = min(max((int)(originZ + t * forward.z), 0), boxDepth - 1);

            float d = dye[idx3d(sampleX, sampleY, sampleZ, boxWidth, boxHeight)];

            float transparency = expf(-absorption * d * stepSize);  // Beer-Lambert law
            brightness += transmittance * d * gain * stepSize;
            transmittance *= transparency;
        }
    }

    unsigned char c = (unsigned char)(fminf(brightness, 1.0f) * 255.0f);
    surf2Dwrite(make_uchar4(c, c, c, 255), surface, pixelX * sizeof(uchar4), pixelY);
}

void renderVolume(FluidFields& fields, cudaSurfaceObject_t surface, int renderWidth, int renderHeight, float3 camPos,
                  float3 forward, float3 right, float3 up, float viewSizeMultiplier) {
    dim3 threadsPerBlock = dim3(16, 16);
    dim3 blocksPerGrid = dim3((renderWidth + 15) / 16, (renderHeight + 15) / 16);
    renderVolumeKernel<<<blocksPerGrid, threadsPerBlock>>>(fields.dye[0], surface, fields.width, fields.height,
                                                           fields.depth, renderWidth, renderHeight, camPos, forward,
                                                           right, up, viewSizeMultiplier);
    CHECK_CUDA(cudaGetLastError());
}
