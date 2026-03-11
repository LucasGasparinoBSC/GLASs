#include "DeviceUtils.hpp"
#include "unittKernels.cuh"
#include <iostream>
#include <cstdlib>
#include <cstdio>

int main() {

    // Create 2 bf16 arrays on device
    uint32_t arrSize = 1024;
    DeviceUtils::bf16 *d_arr1;
    PUSH_RANGE("Device Memory Allocation", 1);
    #if defined(USE_CUDA)
        CUDA_CHECK(cudaMalloc(&d_arr1, arrSize * sizeof(DeviceUtils::bf16)));
        CUDA_CHECK(cudaMemset(d_arr1, 0, arrSize * sizeof(DeviceUtils::bf16)));
    #elif defined(USE_HIP)
        HIP_CHECK(hipMalloc(&d_arr1, arrSize * sizeof(DeviceUtils::bf16)));
        HIP_CHECK(hipMemset(d_arr1, 0, arrSize * sizeof(DeviceUtils::bf16)));
    #endif
    POP_RANGE();

    // Create a kernel stream
    DeviceUtils::Stream_t kernelStream;
    DeviceUtils::StreamCreate(&kernelStream);

    // Launch a kernel to initialize the array
    uint32_t tpb = 256;
    uint32_t bpg = (arrSize + tpb - 1) / tpb;
    dim3 grid(bpg,1,1);
    dim3 block(tpb,1,1);
    PUSH_RANGE("testKernel",1);
    DeviceUtils::launchKernel(testKernel, grid, block, kernelStream, d_arr1, arrSize);
    DeviceUtils::StreamSynchronize(kernelStream);
    POP_RANGE();

    // Copy back to host and check
    DeviceUtils::bf16 *h_arr1 = (DeviceUtils::bf16*)calloc(arrSize , sizeof(DeviceUtils::bf16));
    PUSH_RANGE("Device to Host Copy", 1);
    #if defined(USE_CUDA)
        CUDA_CHECK(cudaMemcpy(h_arr1, d_arr1, arrSize * sizeof(DeviceUtils::bf16), cudaMemcpyDeviceToHost));
    #elif defined(USE_HIP)
        HIP_CHECK(hipMemcpy(h_arr1, d_arr1, arrSize * sizeof(DeviceUtils::bf16), hipMemcpyDeviceToHost));
    #endif
    POP_RANGE();

    PUSH_RANGE("Validation", 1);
    for (uint32_t i = 0; i < arrSize; ++i) {
        //DeviceUtils::bf16 expected = __float2bfloat16(static_cast<float>(i));
        DeviceUtils::bf16 expected = DeviceUtils::floatToBf16(static_cast<float>(i));
        if (DeviceUtils::bf16ToFloat(h_arr1[i]) != DeviceUtils::bf16ToFloat(expected)) {
            std::cerr << "Mismatch at index " << i << ": expected " << DeviceUtils::bf16ToFloat(expected)
                      << ", got " << DeviceUtils::bf16ToFloat(h_arr1[i]) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    POP_RANGE();
    std::cout << "All values match expected results!" << std::endl;
    printf("Last value: %f\n", DeviceUtils::bf16ToFloat(h_arr1[arrSize - 1]));

    // Cleanup
    free(h_arr1);
    DeviceUtils::StreamDestroy(kernelStream);
    #if defined(USE_CUDA)
        CUDA_CHECK(cudaFree(d_arr1));
    #elif defined(USE_HIP)
        HIP_CHECK(hipFree(d_arr1));
    #endif
    return 0;
}