/**
 * @file Kernels_Lv1.cu
 * @author Lucas Gasparino (lucas.gasparino3110@gmail.com)
 * @brief Specific template instantiations for device kernels in Kernels_Lv1
 * @version 0.1
 * @date 2025-10-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "Kernels_Lv1.cuh"

template <typename ITYPE, typename RTYPE>
__global__ void axpy(const RTYPE a, const RTYPE* x, RTYPE* y, const ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        y[gid] += a * x[gid];
        gid += blockDim.x * gridDim.x;
    }
}
template __global__ void axpy<uint32_t, float>(const float, const float*, float*, const uint32_t);
template __global__ void axpy<uint64_t, float>(const float, const float*, float*, const uint64_t);
template __global__ void axpy<uint32_t, double>(const double, const double*, double*, const uint32_t);
template __global__ void axpy<uint64_t, double>(const double, const double*, double*, const uint64_t);
template __global__ void axpy<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16, const DeviceUtils::bf16*, DeviceUtils::bf16*, const uint32_t);
template __global__ void axpy<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16, const DeviceUtils::bf16*, DeviceUtils::bf16*, const uint64_t);

template <typename ITYPE, typename RTYPE>
__global__ void dot_product(const RTYPE* a, const RTYPE* b, double* r, ITYPE N) {
    // Indexes
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    ITYPE tid = threadIdx.x;

    // Partials
    __shared__ float cache[TILE_SIZE];
    double value = 0.0;
    while (gid < N) {
        value += static_cast<double>(a[gid] * b[gid]);
        gid += blockDim.x * gridDim.x;
    }
    cache[tid] = value;
    __syncthreads();

    // Reduction in shared memory
    ITYPE i = blockDim.x / 2;
    while (i != 0) {
        if (tid < i) {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Atomic add to global result
    if (tid == 0) {
        atomicAdd(r, cache[0]);
    }
}
template __global__ void dot_product<uint32_t, float>(const float*, const float*, double*, uint32_t);
template __global__ void dot_product<uint64_t, float>(const float*, const float*, double*, uint64_t);
template __global__ void dot_product<uint32_t, double>(const double*, const double*, double*, uint32_t);
template __global__ void dot_product<uint64_t, double>(const double*, const double*, double*, uint64_t);
template __global__ void dot_product<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16*, const DeviceUtils::bf16*, double*, uint32_t);
template __global__ void dot_product<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16*, const DeviceUtils::bf16*, double*, uint64_t);


template <typename ITYPE, typename RTYPE>
__global__ void scale(const RTYPE a, RTYPE* x, ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        x[gid] *= a;
        gid += blockDim.x * gridDim.x;
    }
}
template __global__ void scale<uint32_t, float>(const float, float*, uint32_t);
template __global__ void scale<uint64_t, float>(const float, float*, uint64_t);
template __global__ void scale<uint32_t, double>(const double, double*, uint32_t);
template __global__ void scale<uint64_t, double>(const double, double*, uint64_t);
template __global__ void scale<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16, DeviceUtils::bf16*, uint32_t);
template __global__ void scale<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16, DeviceUtils::bf16*, uint64_t);

template <typename ITYPE, typename RTYPE>
__global__ void copy_array(const RTYPE* x, RTYPE* y, ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        y[gid] = x[gid];
        gid += blockDim.x * gridDim.x;
    }
}
template __global__ void copy_array<uint32_t, float>(const float*, float*, uint32_t);
template __global__ void copy_array<uint64_t, float>(const float*, float*, uint64_t);
template __global__ void copy_array<uint32_t, double>(const double*, double*, uint32_t);
template __global__ void copy_array<uint64_t, double>(const double*, double*, uint64_t);
template __global__ void copy_array<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16*, DeviceUtils::bf16*, uint32_t);
template __global__ void copy_array<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16*, DeviceUtils::bf16*, uint64_t);

template <typename ITYPE, typename RTYPE>
__global__ void array_sqrt(RTYPE* x, ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    double value;
    while (gid < N) {
        value = static_cast<double>(x[gid]);
        x[gid] = static_cast<RTYPE>(std::sqrt(value));
        gid += blockDim.x * gridDim.x;
    }
}
template __global__ void array_sqrt<uint32_t, float>(float*, uint32_t);
template __global__ void array_sqrt<uint64_t, float>(float*, uint64_t);
template __global__ void array_sqrt<uint32_t, double>(double*, uint32_t);
template __global__ void array_sqrt<uint64_t, double>(double*, uint64_t);
template __global__ void array_sqrt<uint32_t, DeviceUtils::bf16>(DeviceUtils::bf16*, uint32_t);
template __global__ void array_sqrt<uint64_t, DeviceUtils::bf16>(DeviceUtils::bf16*, uint64_t);

template <typename ITYPE, typename RTYPE>
__global__ void array_invert(RTYPE* x, ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    double value;
    while (gid < N) {
        value = static_cast<double>(x[gid]);
        x[gid] = static_cast<RTYPE>(1.0 / value);
        gid += blockDim.x * gridDim.x;
    }
}
template __global__ void array_invert<uint32_t, float>(float*, uint32_t);
template __global__ void array_invert<uint64_t, float>(float*, uint64_t);
template __global__ void array_invert<uint32_t, double>(double*, uint32_t);
template __global__ void array_invert<uint64_t, double>(double*, uint64_t);
template __global__ void array_invert<uint32_t, DeviceUtils::bf16>(DeviceUtils::bf16*, uint32_t);
template __global__ void array_invert<uint64_t, DeviceUtils::bf16>(DeviceUtils::bf16*, uint64_t);

template <typename ITYPE, typename RTYPE>
__global__ void pointwise_multiply(const RTYPE* x, RTYPE* y, ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        y[gid] *= x[gid];
        gid += blockDim.x * gridDim.x;
    }
}
template __global__ void pointwise_multiply<uint32_t, float>(const float*, float*, uint32_t);
template __global__ void pointwise_multiply<uint64_t, float>(const float*, float*, uint64_t);
template __global__ void pointwise_multiply<uint32_t, double>(const double*, double*, uint32_t);
template __global__ void pointwise_multiply<uint64_t, double>(const double*, double*, uint64_t);
template __global__ void pointwise_multiply<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16*, DeviceUtils::bf16*, uint32_t);
template __global__ void pointwise_multiply<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16*, DeviceUtils::bf16*, uint64_t);

template <typename ITYPE, typename RTYPE>
__global__ void set_array(RTYPE* x, RTYPE value, ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        x[gid] = value;
        gid += blockDim.x * gridDim.x;
    }
}
template __global__ void set_array<uint32_t, float>(float*, float, uint32_t);
template __global__ void set_array<uint64_t, float>(float*, float, uint64_t);
template __global__ void set_array<uint32_t, double>(double*, double, uint32_t);
template __global__ void set_array<uint64_t, double>(double*, double, uint64_t);
template __global__ void set_array<uint32_t, DeviceUtils::bf16>(DeviceUtils::bf16*, DeviceUtils::bf16, uint32_t);
template __global__ void set_array<uint64_t, DeviceUtils::bf16>(DeviceUtils::bf16*, DeviceUtils::bf16, uint64_t);

template <typename ITYPE, typename RTYPE_IN, typename RTYPE_OUT>
__global__ void convert_array(const ITYPE N, const RTYPE_IN* input, RTYPE_OUT* output) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        output[gid] = static_cast<RTYPE_OUT>(input[gid]);
        gid += blockDim.x * gridDim.x;
    }
}
template __global__ void convert_array<uint32_t, float, double>(const uint32_t, const float*, double*);
template __global__ void convert_array<uint64_t, float, double>(const uint64_t, const float*, double*);
template __global__ void convert_array<uint32_t, double, float>(const uint32_t, const double*, float*);
template __global__ void convert_array<uint64_t, double, float>(const uint64_t, const double*, float*);
template __global__ void convert_array<uint32_t, float, DeviceUtils::bf16>(const uint32_t, const float*, DeviceUtils::bf16*);
template __global__ void convert_array<uint64_t, float, DeviceUtils::bf16>(const uint64_t, const float*, DeviceUtils::bf16*);
template __global__ void convert_array<uint32_t, DeviceUtils::bf16, float>(const uint32_t, const DeviceUtils::bf16*, float*);
template __global__ void convert_array<uint64_t, DeviceUtils::bf16, float>(const uint64_t, const DeviceUtils::bf16*, float*);
template __global__ void convert_array<uint32_t, double, DeviceUtils::bf16>(const uint32_t, const double*, DeviceUtils::bf16*);
template __global__ void convert_array<uint64_t, double, DeviceUtils::bf16>(const uint64_t, const double*, DeviceUtils::bf16*);
template __global__ void convert_array<uint32_t, DeviceUtils::bf16, double>(const uint32_t, const DeviceUtils::bf16*, double*);
template __global__ void convert_array<uint64_t, DeviceUtils::bf16, double>(const uint64_t, const DeviceUtils::bf16*, double*);