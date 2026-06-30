#ifndef DEVICE_SIDE_CUH
#define DEVICE_SIDE_CUH

#include "DeviceUtils.hpp"

#if defined (USE_CUDA)
    template <typename ITYPE, typename RTYPE>
    __global__ void diagMatVec_device(const RTYPE* Adiag, const RTYPE* x_in, RTYPE* x_out, ITYPE N) {
        ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < N) {
            x_out[idx] = Adiag[idx] * x_in[idx];
            idx += blockDim.x * gridDim.x;
        }
    }
#elif defined (USE_HIP)
    template <typename ITYPE, typename RTYPE>
    __global__ void diagMatVec_device(const RTYPE* Adiag, const RTYPE* x_in, RTYPE* x_out, ITYPE N);
#endif

#endif //! DEVICE_SIDE_CUH