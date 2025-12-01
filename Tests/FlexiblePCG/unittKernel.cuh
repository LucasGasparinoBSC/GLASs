#ifndef UNITT_KERNEL_CUH
#define UNITT_KERNEL_CUH

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "ConjugateGradient.hpp"
#include "halo_ops.hpp"

template<typename ITYPE, typename RTYPE>
__global__ void modify_xout(const ITYPE prank, const ITYPE nranks, const RTYPE* ldata, const RTYPE* rdata, const RTYPE* d_c, const RTYPE* d_e, RTYPE* x_out, const ITYPE N);

__global__ void tridiagMatVec_16(const __nv_bfloat16* c, const __nv_bfloat16 *d, const __nv_bfloat16 *e, const __nv_bfloat16* x_in, __nv_bfloat16* x_out, const uint32_t N);
__global__ void diagPrecond_16(const __nv_bfloat16* d_d, const __nv_bfloat16* r_in, __nv_bfloat16* r_out, const uint32_t N);

__global__ void tridiagMatVec_32(const float* c, const float *d, const float *e, const float* x_in, float* x_out, const uint32_t N);
__global__ void diagPrecond_32(const float* d_d, const float* r_in, float* r_out, const uint32_t N);

__global__ void tridiagMatVec_64(const double* c, const double *d, const double *e, const double* x_in, double* x_out, const uint32_t N);
__global__ void diagPrecond_64(const double* d_d, const double* r_in, double* r_out, const uint32_t N);

void runSolver_16(Comm_Utils commObj, uint32_t nrows, const __nv_bfloat16* c_d, const __nv_bfloat16* d_d, const __nv_bfloat16* e_d, __nv_bfloat16* ldata, __nv_bfloat16* rdata, ConjugateGradient<uint32_t, __nv_bfloat16>& solver);
void runSolver_32(Comm_Utils commObj, uint32_t nrows, const float* c_d, const float* d_d, const float* e_d, float* ldata, float* rdata,ConjugateGradient<uint32_t, float>& solver);
void runSolver_64(Comm_Utils commObj, uint32_t nrows, const double* c_d, const double* d_d, const double* e_d, double* ldata, double* rdata,ConjugateGradient<uint32_t, double>& solver);

#endif // UNITT_KERNEL_CUH