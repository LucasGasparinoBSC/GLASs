#include "AuxKernels.cuh"

namespace AuxKernels
{
    template __global__ void matvec_nohalo<uint32_t, float>(const float* cl, const float* dl, const float* el, const float* x, float* y, const uint32_t n);
    template __global__ void matvec_nohalo<uint64_t, float>(const float* cl, const float* dl, const float* el, const float* x, float* y, const uint64_t n);
    template __global__ void matvec_nohalo<uint32_t, double>(const double* cl, const double* dl, const double* el, const double* x, double* y, const uint32_t n);
    template __global__ void matvec_nohalo<uint64_t, double>(const double* cl, const double* dl, const double* el, const double* x, double* y, const uint64_t n);
    template __global__ void matvec_nohalo<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16* cl, const DeviceUtils::bf16* dl, const DeviceUtils::bf16* el, const DeviceUtils::bf16* x, DeviceUtils::bf16* y, const uint32_t n);
    template __global__ void matvec_nohalo<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16* cl, const DeviceUtils::bf16* dl, const DeviceUtils::bf16* el, const DeviceUtils::bf16* x, DeviceUtils::bf16* y, const uint64_t n);

    template __global__ void matvec_halo<uint32_t, float>(const float* cl, const float* dl, const float* el, const float* x, const float* ghosts, float* y, const uint32_t n);
    template __global__ void matvec_halo<uint64_t, float>(const float* cl, const float* dl, const float* el, const float* x, const float* ghosts, float* y, const uint64_t n);
    template __global__ void matvec_halo<uint32_t, double>(const double* cl, const double* dl, const double* el, const double* x, const double* ghosts, double* y, const uint32_t n);
    template __global__ void matvec_halo<uint64_t, double>(const double* cl, const double* dl, const double* el, const double* x, const double* ghosts, double* y, const uint64_t n);
    template __global__ void matvec_halo<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16* cl, const DeviceUtils::bf16* dl, const DeviceUtils::bf16* el, const DeviceUtils::bf16* x, const DeviceUtils::bf16* ghosts, DeviceUtils::bf16* y, const uint32_t n);
    template __global__ void matvec_halo<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16* cl, const DeviceUtils::bf16* dl, const DeviceUtils::bf16* el, const DeviceUtils::bf16* x, const DeviceUtils::bf16* ghosts, DeviceUtils::bf16* y, const uint64_t n);
}