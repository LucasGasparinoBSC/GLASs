#ifndef DEVICE_SIDE_CUH
#define DEVICE_SIDE_CUH

#include "DeviceUtils.hpp"

template <typename ITYPE, typename RTYPE>
__global__ void diagMatVec_device(const RTYPE* Adiag, const RTYPE* x_in, RTYPE* x_out, ITYPE N);

#endif //! DEVICE_SIDE_CUH