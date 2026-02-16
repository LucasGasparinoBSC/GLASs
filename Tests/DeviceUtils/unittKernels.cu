#include "unittKernels.cuh"

__global__ void testKernel(DeviceUtils::bf16* data, uint32_t N) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        DeviceUtils::bf16 value = __float2bfloat16(static_cast<float>(idx));
        data[idx] = value;
    }
}