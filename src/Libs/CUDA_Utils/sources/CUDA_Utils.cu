#include "CUDA_Utils.cuh"

// Explicit instantiations
template __global__ void axpy<uint32_t, float>(const float, const float*, float*, const uint32_t);
template __global__ void axpy<uint64_t, float>(const float, const float*, float*, const uint64_t);
template __global__ void axpy<uint32_t, double>(const double, const double*, double*, const uint32_t);
template __global__ void axpy<uint64_t, double>(const double, const double*, double*, const uint64_t);
template __global__ void axpy<uint32_t, __nv_bfloat16>(const __nv_bfloat16, const __nv_bfloat16*, __nv_bfloat16*, const uint32_t);
template __global__ void axpy<uint64_t, __nv_bfloat16>(const __nv_bfloat16, const __nv_bfloat16*, __nv_bfloat16*, const uint64_t);

template __global__ void dot_product<uint32_t, float>(const float*, const float*, float*, uint32_t);
template __global__ void dot_product<uint64_t, float>(const float*, const float*, float*, uint64_t);
template __global__ void dot_product<uint32_t, double>(const double*, const double*, float*, uint32_t);
template __global__ void dot_product<uint64_t, double>(const double*, const double*, float*, uint64_t);
template __global__ void dot_product<uint32_t, __nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, float*, uint32_t);
template __global__ void dot_product<uint64_t, __nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, float*, uint64_t);

template __global__ void scale<uint32_t, float>(const float, float*, uint32_t);
template __global__ void scale<uint64_t, float>(const float, float*, uint64_t);
template __global__ void scale<uint32_t, double>(const double, double*, uint32_t);
template __global__ void scale<uint64_t, double>(const double, double*, uint64_t);
template __global__ void scale<uint32_t, __nv_bfloat16>(const __nv_bfloat16, __nv_bfloat16*, uint32_t);
template __global__ void scale<uint64_t, __nv_bfloat16>(const __nv_bfloat16, __nv_bfloat16*, uint64_t);

template __global__ void copy_array<uint32_t, float>(const float*, float*, uint32_t);
template __global__ void copy_array<uint64_t, float>(const float*, float*, uint64_t);
template __global__ void copy_array<uint32_t, double>(const double*, double*, uint32_t);
template __global__ void copy_array<uint64_t, double>(const double*, double*, uint64_t);
template __global__ void copy_array<uint32_t, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, uint32_t);
template __global__ void copy_array<uint64_t, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, uint64_t);

template __global__ void array_sqrt<uint32_t, float>(float*, uint32_t);
template __global__ void array_sqrt<uint64_t, float>(float*, uint64_t);
template __global__ void array_sqrt<uint32_t, double>(double*, uint32_t);
template __global__ void array_sqrt<uint64_t, double>(double*, uint64_t);
template __global__ void array_sqrt<uint32_t, __nv_bfloat16>(__nv_bfloat16*, uint32_t);
template __global__ void array_sqrt<uint64_t, __nv_bfloat16>(__nv_bfloat16*, uint64_t);

template __global__ void array_invert<uint32_t, float>(float*, uint32_t);
template __global__ void array_invert<uint64_t, float>(float*, uint64_t);
template __global__ void array_invert<uint32_t, double>(double*, uint32_t);
template __global__ void array_invert<uint64_t, double>(double*, uint64_t);
template __global__ void array_invert<uint32_t, __nv_bfloat16>(__nv_bfloat16*, uint32_t);
template __global__ void array_invert<uint64_t, __nv_bfloat16>(__nv_bfloat16*, uint64_t);

template __global__ void pointwise_multiply<uint32_t, float>(const float*, float*, uint32_t);
template __global__ void pointwise_multiply<uint64_t, float>(const float*, float*, uint64_t);
template __global__ void pointwise_multiply<uint32_t, double>(const double*, double*, uint32_t);
template __global__ void pointwise_multiply<uint64_t, double>(const double*, double*, uint64_t);
template __global__ void pointwise_multiply<uint32_t, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, uint32_t);
template __global__ void pointwise_multiply<uint64_t, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, uint64_t);