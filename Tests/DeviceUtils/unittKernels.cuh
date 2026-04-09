#ifndef UNITTKERNELS_CUH
#define UNITTKERNELS_CUH

#pragma once

#include "DeviceUtils.hpp"

template<typename ITYPE, typename RTYPE>
__global__ void testKernel(const RTYPE* input, RTYPE* output, ITYPE N);

#endif // UNITTKERNELS_CUH