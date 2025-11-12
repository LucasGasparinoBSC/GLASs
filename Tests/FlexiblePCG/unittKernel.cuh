#ifndef UNITT_KERNEL_CUH
#define UNITT_KERNEL_CUH

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "ConjugateGradient.hpp"

__global__ void tridiagdiagMatVec_32(const float* c, const float *d, const float *e, const float* x_in, float* x_out, const uint32_t N);
__global__ void tridiagdiagMatVec_64(const double* c, const double *d, const double *e, const double* x_in, double* x_out, const uint32_t N);
/*
void runSolver_32(uint32_t nrows, float* A, ConjugateGradient<uint32_t, float>& solver);
void runSolver_64(uint32_t nrows, double* A, ConjugateGradient<uint32_t, double>& solver);
*/

#endif // UNITT_KERNEL_CUH