#ifndef UNITT_KERNEL_CUH
#define UNITT_KERNEL_CUH

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "ConjugateGradient.hpp"

__global__ void diagMatVec_32(const float* A, const float* x_in, float* x_out, uint32_t N);
__global__ void diagMatVec_64(const double* A, const double* x_in, double* x_out, uint32_t N);
void runSolver_32(uint32_t nrows, float* A, ConjugateGradient<uint32_t, float>& solver);

#endif // UNITT_KERNEL_CUH