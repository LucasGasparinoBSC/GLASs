#include "DeviceSide.cuh"

template __global__ void diagMatVec_device<uint32_t, float>(const float* Amat, const float* x_in, float* x_out, uint32_t N);
template __global__ void diagMatVec_device<uint64_t, float>(const float* Amat, const float* x_in, float* x_out, uint64_t N);
template __global__ void diagMatVec_device<uint32_t, double>(const double* Amat, const double* x_in, double* x_out, uint32_t N);
template __global__ void diagMatVec_device<uint64_t, double>(const double* Amat, const double* x_in, double* x_out, uint64_t N);
template __global__ void diagMatVec_device<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16* Amat, const DeviceUtils::bf16* x_in, DeviceUtils::bf16* x_out, uint32_t N);
template __global__ void diagMatVec_device<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16* Amat, const DeviceUtils::bf16* x_in, DeviceUtils::bf16* x_out, uint64_t N);