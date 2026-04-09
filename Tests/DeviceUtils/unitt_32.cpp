/**
 * @file unitt_32.cpp
 * @author Lucas Gasparino
 * @brief Unit test DeviceUtils members using FP32 data type
 * @version 0.1
 * @date 2024-06-01
 * @details Creates 2 FP32 arrays on the host, copies them to device, then calls a kernel using DeviceUtils member functions.
 */
#include "DeviceUtils.hpp"
#include "unittKernels.cuh"
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstdint>

int main() {
    // Basic info
    const uint32_t narr = static_cast<uint32_t>(1024);

    // Host arrays
    PUSH_RANGE("Host Array Creation", 0);
    float *h_input = (float *)calloc(narr, sizeof(float));
    float *h_output = (float *)calloc(narr, sizeof(float));
    for (uint32_t i = 0; i < narr; ++i) {
        h_input[i] = static_cast<float>(i+1);
    }
    POP_RANGE();

    // Device arrays
    PUSH_RANGE("Device Array Creation", 1);
    float *d_input, *d_output;
    // NOTE: separate between AMD and NVIDIA at this stage, as DeviceMemory is tested later
    #if defined(USE_CUDA)
        CUDA_CHECK(cudaMalloc(&d_input, narr * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, narr * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, narr * sizeof(float), cudaMemcpyHostToDevice));
    #elif defined(USE_HIP)
        HIP_CHECK(hipMalloc(&d_input, narr * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_output, narr * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_input, h_input, narr * sizeof(float), hipMemcpyHostToDevice));
    #endif
    DeviceUtils::DeviceSynchronize();
    POP_RANGE();

    // Create kernel launch info
    uint32_t numThreads = 256;
    uint32_t numBlocks = (narr + numThreads - 1) / numThreads;
    dim3 block(numThreads,1,1);
    dim3 grid(numBlocks,1,1);
    DeviceUtils::Stream_t kStream;
    DeviceUtils::StreamCreate(&kStream);

    // Launch the kernel
    DeviceUtils::launchKernel(testKernel<uint32_t,float>, grid, block, kStream, d_input, d_output, narr);
    DeviceUtils::DeviceSynchronize();

    // Copy back and error check (o = 2*i)
    PUSH_RANGE("Copy Back and Check", 2);
    #if defined(USE_CUDA)
        CUDA_CHECK(cudaMemcpy(h_output, d_output, narr * sizeof(float), cudaMemcpyDeviceToHost));
    #elif defined(USE_HIP)
        HIP_CHECK(hipMemcpy(h_output, d_output, narr * sizeof(float), hipMemcpyDeviceToHost));
    #endif
    DeviceUtils::DeviceSynchronize();
    for (uint32_t i = 0; i < narr; ++i) {
        float expected = 2.0f * static_cast<float>(i+1);
        if (h_output[i] != expected) {
            std::cerr << "Error at index " << i << ": expected " << expected << ", got " << h_output[i] << std::endl;
            return -1;
        }
    }
    printf("Last entry = %f\n", h_output[narr-1]);
    std::cout << "Test passed successfully!" << std::endl;
    POP_RANGE();

    // Cleanup
    DeviceUtils::StreamDestroy(kStream);
    #if defined(USE_CUDA)
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    #elif defined(USE_HIP)
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    #endif
    free(h_input);
    free(h_output);
    return 0;
}