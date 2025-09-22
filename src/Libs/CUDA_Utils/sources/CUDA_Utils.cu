#include "CUDA_Utils.cuh"

template <typename ITYPE, typename RTYPE>
__global__ void axpy(const RTYPE a, RTYPE* x, RTYPE* y, ITYPE N) {
    ITYPE i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < N) {
        y[i] += a * x[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename ITYPE, typename RTYPE>
__global__ void dot_product(const RTYPE* x, const RTYPE* y, RTYPE* result, ITYPE N) {
}

INSTANTIATE_KERNELS(uint32_t, float)
INSTANTIATE_KERNELS(uint64_t, float)
INSTANTIATE_KERNELS(uint32_t, double)
INSTANTIATE_KERNELS(uint64_t, double)
INSTANTIATE_KERNELS(uint32_t, __nv_bfloat16)
INSTANTIATE_KERNELS(uint64_t, __nv_bfloat16)