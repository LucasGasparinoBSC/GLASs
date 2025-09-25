#include "CUDA_Utils.cuh"

// Explicit instantiations
template __global__ void axpy<uint32_t, float>(float, float*, float*, uint32_t);
template __global__ void axpy<uint64_t, float>(float, float*, float*, uint64_t);
template __global__ void axpy<uint32_t, double>(double, double*, double*, uint32_t);
template __global__ void axpy<uint64_t, double>(double, double*, double*, uint64_t);
template __global__ void axpy<uint32_t, __nv_bfloat16>(__nv_bfloat16, __nv_bfloat16*, __nv_bfloat16*, uint32_t);
template __global__ void axpy<uint64_t, __nv_bfloat16>(__nv_bfloat16, __nv_bfloat16*, __nv_bfloat16*, uint64_t);

template __global__ void dot_product<uint32_t, float>(const float*, const float*, float*, uint32_t);
template __global__ void dot_product<uint64_t, float>(const float*, const float*, float*, uint64_t);
template __global__ void dot_product<uint32_t, double>(const double*, const double*, float*, uint32_t);
template __global__ void dot_product<uint64_t, double>(const double*, const double*, float*, uint64_t);