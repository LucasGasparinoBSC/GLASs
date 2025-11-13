#ifndef UNITT_KERNEL_CUH
#define UNITT_KERNEL_CUH

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "ConjugateGradient.hpp"
#include "halo_ops.hpp"

__global__ void tridiagMatVec_32(const float* c, const float *d, const float *e, const float* x_in, float* x_out, const uint32_t N);
__global__ void diagPrecond_32(const float* d_d, const float* r_in, float* r_out, const uint32_t N);

__global__ void tridiagMatVec_64(const double* c, const double *d, const double *e, const double* x_in, double* x_out, const uint32_t N);
__global__ void diagPrecond_64(const double* d_d, const double* r_in, double* r_out, const uint32_t N);

void runSolver_32(uint32_t nrows, const float* c_d, const float* d_d, const float* e_d, ConjugateGradient<uint32_t, float>& solver);
void runSolver_64(uint32_t nrows, const double* c_d, const double* d_d, const double* e_d, ConjugateGradient<uint32_t, double>& solver);

#endif // UNITT_KERNEL_CUH