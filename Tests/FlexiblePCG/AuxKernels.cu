#include "AuxKernels.cuh"

namespace AuxKernels
{
    #if defined (USE_HIP)
        // Only handle internal nodes, from 1 to n-2, inclusive.
        template <typename ITYPE, typename RTYPE>
        __global__ void matvec_nohalo(const RTYPE* cl, const RTYPE* dl, const RTYPE* el, const RTYPE* x, RTYPE* y, const ITYPE n)
        {
            ITYPE i = blockIdx.x * blockDim.x + threadIdx.x + 1;
            while (i < n - 1)
            {
                y[i] = cl[i -1] * x[i - 1] + dl[i] * x[i] + el[i + 1] * x[i + 1];
                i += blockDim.x * gridDim.x;
            }
        }

        // Handle halos with received data
        template <typename ITYPE, typename RTYPE>
        __global__ void matvec_halo(const RTYPE* cl, const RTYPE* dl, const RTYPE* el, const RTYPE* x, const RTYPE* ghosts, RTYPE* y, const ITYPE n)
        {
            if (threadIdx.x == 0)
            {
                // Left internal node
                y[0] = dl[0]*x[0] + el[1]*x[1] + ghosts[0]*ghosts[1];

                // Right internal node
                y[n-1] = cl[n-2]*x[n-2] + dl[n-1]*x[n-1] + ghosts[2]*ghosts[3];
            }
        }

        // Diagonal preconditioner kernel
        template <typename ITYPE, typename RTYPE>
        __global__ void diag_precond(const RTYPE* dl, const RTYPE* r, RTYPE* z, const ITYPE n)
        {
            ITYPE i = blockIdx.x * blockDim.x + threadIdx.x;
            while (i < n)
            {
                z[i] = r[i] / dl[i];
                i += blockDim.x * gridDim.x;
            }
        }

        // Kernel to fill the buffer for halo exchange
        template <typename ITYPE, typename RTYPE>
        __global__ void fillBuffer(const RTYPE* cl, const RTYPE* el, const RTYPE* x, RTYPE* buffer, const ITYPE n)
        {
            if (threadIdx.x == 0)
            {
                // Fill left halo data
                buffer[0] = cl[0] * x[0]; // left_cl contribution
                buffer[1] = el[1] * x[1]; // left_el contribution

                // Fill right halo data
                buffer[2] = cl[n-2] * x[n-2]; // right_cl contribution
                buffer[3] = el[n-1] * x[n-1]; // right_el contribution
            }
        }
    #endif

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

    template __global__ void diag_precond<uint32_t, float>(const float* dl, const float* r, float* z, const uint32_t n);
    template __global__ void diag_precond<uint64_t, float>(const float* dl, const float* r, float* z, const uint64_t n);
    template __global__ void diag_precond<uint32_t, double>(const double* dl, const double* r, double* z, const uint32_t n);
    template __global__ void diag_precond<uint64_t, double>(const double* dl, const double* r, double* z, const uint64_t n);
    template __global__ void diag_precond<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16* dl, const DeviceUtils::bf16* r, DeviceUtils::bf16* z, const uint32_t n);
    template __global__ void diag_precond<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16* dl, const DeviceUtils::bf16* r, DeviceUtils::bf16* z, const uint64_t n);

    template __global__ void fillBuffer<uint32_t, float>(const float* cl, const float* el, const float* x, float* buffer, const uint32_t n);
    template __global__ void fillBuffer<uint64_t, float>(const float* cl, const float* el, const float* x, float* buffer, const uint64_t n);
    template __global__ void fillBuffer<uint32_t, double>(const double* cl, const double* el, const double* x, double* buffer, const uint32_t n);
    template __global__ void fillBuffer<uint64_t, double>(const double* cl, const double* el, const double* x, double* buffer, const uint64_t n);
    template __global__ void fillBuffer<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16* cl, const DeviceUtils::bf16* el, const DeviceUtils::bf16* x, DeviceUtils::bf16* buffer, const uint32_t n);
    template __global__ void fillBuffer<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16* cl, const DeviceUtils::bf16* el, const DeviceUtils::bf16* x, DeviceUtils::bf16* buffer, const uint64_t n);
}