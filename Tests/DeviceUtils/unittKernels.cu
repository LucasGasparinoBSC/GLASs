#include "unittKernels.cuh"

template<typename ITYPE, typename RTYPE>
__global__ void testKernel(const RTYPE* input, RTYPE* output, ITYPE N) {
    ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] + input[idx]; // Simple operation for testing, o = i + i
    }
}

template __global__ void testKernel<uint32_t, float>(const float* input, float* output, uint32_t N);
template __global__ void testKernel<uint64_t, float>(const float* input, float* output, uint64_t N);
template __global__ void testKernel<uint32_t, double>(const double* input, double* output, uint32_t N);
template __global__ void testKernel<uint64_t, double>(const double* input, double* output, uint64_t N);
#ifdef USE_GPU
    template __global__ void testKernel<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16* input, DeviceUtils::bf16* output, uint32_t N);
    template __global__ void testKernel<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16* input, DeviceUtils::bf16* output, uint64_t N);
#endif
