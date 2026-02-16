#ifndef UNITTKERNELS_CUH
#define UNITTKERNELS_CUH

#pragma once

#include "DeviceUtils.hpp"

__global__ void testKernel(DeviceUtils::bf16* data, uint32_t N);

#endif // UNITTKERNELS_CUH