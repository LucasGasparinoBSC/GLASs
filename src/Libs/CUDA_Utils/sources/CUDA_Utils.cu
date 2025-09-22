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
    // Global index
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread index
    ITYPE tid = threadIdx.x;

    // SHM cache
    __shared__ RTYPE cache[256]; // Assuming blockDim.x == 256

    // Partial accumulations
    double tmp = static_cast<double>(0);
    while (gid < N) {
        tmp += static_cast<double>(x[gid] * y[gid]);
        gid += blockDim.x * gridDim.x;
    }
    cache[tid] = static_cast<RTYPE>(tmp);
    __syncthreads();

    // Reduce within blocks
    ITYPE i = blockDim.x / 2;
    while (i != 0) {
        if (tid < i) {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Atomic add to a temp scalar
    double temp = 0.0;
    if (tid == 0) {
        atomicAdd(&temp, cache[0]);
        result[0] = static_cast<RTYPE>(temp);
    }
}

INSTANTIATE_KERNELS(uint32_t, float)
INSTANTIATE_KERNELS(uint64_t, float)
INSTANTIATE_KERNELS(uint32_t, double)
INSTANTIATE_KERNELS(uint64_t, double)
INSTANTIATE_KERNELS(uint32_t, __nv_bfloat16)
INSTANTIATE_KERNELS(uint64_t, __nv_bfloat16)