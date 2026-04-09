#include "DeviceSide.cuh"

template <typename ITYPE, typename RTYPE>
__global__ void diagMatVec_device(const RTYPE* Adiag, const RTYPE* x_in, RTYPE* x_out, ITYPE N) {
    ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        x_out[idx] = Adiag[idx] * x_in[idx];
        idx += blockDim.x * gridDim.x;
    }
}

template __global__ void diagMatVec_device<uint32_t, float>(const float* Amat, const float* x_in, float* x_out, uint32_t N);
template __global__ void diagMatVec_device<uint64_t, float>(const float* Amat, const float* x_in, float* x_out, uint64_t N);
template __global__ void diagMatVec_device<uint32_t, double>(const double* Amat, const double* x_in, double* x_out, uint32_t N);
template __global__ void diagMatVec_device<uint64_t, double>(const double* Amat, const double* x_in, double* x_out, uint64_t N);
template __global__ void diagMatVec_device<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16* Amat, const DeviceUtils::bf16* x_in, DeviceUtils::bf16* x_out, uint32_t N);
template __global__ void diagMatVec_device<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16* Amat, const DeviceUtils::bf16* x_in, DeviceUtils::bf16* x_out, uint64_t N);