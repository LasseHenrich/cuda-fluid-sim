#pragma once
#include <cuda_runtime.h>

#include "helper.h"

/// @brief Measures the time between the Cuda events invoked by startTimer() and endTimer()
struct CudaTimer {
    cudaEvent_t start, stop;

    CudaTimer() {
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
    }
    // ToDo: We may want to use cudaEventDestroy in the destructor

    void startTimer() { CHECK_CUDA(cudaEventRecord(start)); }

    void endTimer() { CHECK_CUDA(cudaEventRecord(stop)); }

    /// @brief Check how long start and end events are apart from each other. Note that this calls cudaEventSynchronize,
    /// so only call this function when blocking CPU execution is acceptable
    float elapsedTimeInMs() {
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        return milliseconds;
    }
};