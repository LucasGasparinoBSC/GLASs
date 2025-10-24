/**
 * @file CUDA_Utils.cuh
 * @author Lucas Gasparino (lucas.gasparino3110@gmail.com)
 * @brief CUDA utility functions and kernels
 * @details This file contains a set of typical CUDA utility functions and kernels
 *          for operations such as AXPY, dot product, scaling, copying arrays, etc.,
            as well a macros for checking CUDA errors and using NVTX for profiling.
            The kernel launch parameters defined here are generic and can be adjusted
            based on the specific needs of the application/HW capabilities.
            Each kernel is templated to support different data types (see CUDA_Utils.cu for details).
            The utility also provides a generic kernel launcher function that
            allows usage of this library from non-CUDA C++ code.
 * @version 0.1
 * @date 2025-10-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef __CUDA_UTILS_CUH__
#define __CUDA_UTILS_CUH__

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>

// Parameters for kernel launches
#define TILE_SIZE 256
#define MAX_BLOCKS 10240

// CUDA error handling macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA_Utils Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Macro utility for NVTX ranges
const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

// Push function for starting NVTX ranges
#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}

// Pop function for ending NVTX ranges
#define POP_RANGE nvtxRangePop();

// Kernels:

// AXPY kernel: y = a*x + y
// Receives:
// a: scalar multiplier
// x: input array
// y: input/output array
// N: size of arrays
// ITYPE: integer type for indexing (e.g., uint32_t, uint64_t)
// RTYPE: real type for computations (e.g., float, double, __nv_bfloat16)
template <typename ITYPE, typename RTYPE>
__global__ void axpy(const RTYPE a, const RTYPE* x, RTYPE* y, const ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        y[gid] += a * x[gid];
        gid += blockDim.x * gridDim.x;
    }
}

// Dot product kernel: r = a_i * b_i
// Receives:
// a: input array
// b: input array
// r: output scalar (in double precision for accuracy)
// N: size of arrays
// NOTE: the output is a size(1) pointer, as the atocmicAdd operation requires a pointer
// NOTE: the output is ALWAYS in double precision for accuracy
// NOTE: requires r to be initialized to zero before calling (cudaMemset)
// ITYPE: integer type for indexing (e.g., uint32_t, uint64_t)
// RTYPE: real type for computations (e.g., float, double, __nv_bfloat16)
template <typename ITYPE, typename RTYPE>
__global__ void dot_product(const RTYPE* a, const RTYPE* b, double* r, ITYPE N) {
    // Indexes
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    ITYPE tid = threadIdx.x;

    // Partials
    __shared__ float cache[TILE_SIZE];
    double value = 0.0;
    while (gid < N) {
        value += static_cast<double>(a[gid] * b[gid]);
        gid += blockDim.x * gridDim.x;
    }
    cache[tid] = value;
    __syncthreads();

    // Reduction in shared memory
    ITYPE i = blockDim.x / 2;
    while (i != 0) {
        if (tid < i) {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Atomic add to global result
    if (tid == 0) {
        atomicAdd(r, cache[0]);
    }
}

// Scale kernel: x = a*x
// Receives:
// a: scalar multiplier
// x: input/output array
// N: size of array
// ITYPE: integer type for indexing (e.g., uint32_t, uint64_t)
// RTYPE: real type for computations (e.g., float, double, __nv_bfloat16)
template <typename ITYPE, typename RTYPE>
__global__ void scale(const RTYPE a, RTYPE* x, ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        x[gid] *= a;
        gid += blockDim.x * gridDim.x;
    }
}

// Copy array kernel: y = x
// Receives:
// x: input array
// y: output array
// N: size of arrays
// ITYPE: integer type for indexing (e.g., uint32_t, uint64_t)
// RTYPE: real type for computations (e.g., float, double, __nv_bfloat16)
template <typename ITYPE, typename RTYPE>
__global__ void copy_array(const RTYPE* x, RTYPE* y, ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        y[gid] = x[gid];
        gid += blockDim.x * gridDim.x;
    }
}

// Array sqrt
// Receives:
// x: input/output array
// N: size of array
// ITYPE: integer type for indexing (e.g., uint32_t, uint64_t)
// RTYPE: real type for computations (e.g., float, double, __nv_bfloat16)
template <typename ITYPE, typename RTYPE>
__global__ void array_sqrt(RTYPE* x, ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    double value;
    while (gid < N) {
        value = static_cast<double>(x[gid]);
        x[gid] = static_cast<RTYPE>(std::sqrt(value));
        gid += blockDim.x * gridDim.x;
    }
}

// Array invert: x_i = 1/x_i
// Receives:
// x: input/output array
// N: size of array
// ITYPE: integer type for indexing (e.g., uint32_t, uint64_t)
// RTYPE: real type for computations (e.g., float, double, __nv_bfloat16)
template <typename ITYPE, typename RTYPE>
__global__ void array_invert(RTYPE* x, ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    double value;
    while (gid < N) {
        value = static_cast<double>(x[gid]);
        x[gid] = static_cast<RTYPE>(1.0 / value);
        gid += blockDim.x * gridDim.x;
    }
}

// Pointwise multiply: y_i = x_i * y_i
// Receives:
// x: input array
// y: input/output array
// N: size of arrays
// ITYPE: integer type for indexing (e.g., uint32_t, uint64_t)
// RTYPE: real type for computations (e.g., float, double, __nv_bfloat16)
template <typename ITYPE, typename RTYPE>
__global__ void pointwise_multiply(const RTYPE* x, RTYPE* y, ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        y[gid] *= x[gid];
        gid += blockDim.x * gridDim.x;
    }
}

template <typename ITYPE, typename RTYPE>
__global__ void set_array(RTYPE* x, RTYPE value, ITYPE N) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        x[gid] = value;
        gid += blockDim.x * gridDim.x;
    }
}

template <typename ITYPE, typename RTYPE_IN, typename RTYPE_OUT>
__global__ void convert_array(const ITYPE N, const RTYPE_IN* input, RTYPE_OUT* output) {
    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        output[gid] = static_cast<RTYPE_OUT>(input[gid]);
        gid += blockDim.x * gridDim.x;
    }
}

// Generic templated CUDA kernel launcher
// NOTE: this is a VERY basic implementation; it does not
//       handle usage of different streams, dynamic shared memory, etc.
//       It is recommended to extend this function as needed.
template <typename Kernel, typename... Args>
void launchKernel(
    Kernel k,
    dim3 grid,
    dim3 block,
    cudaStream_t kStream,
    Args&... args) {
        // Pointer of arguments
        void* argPtrs[] = { (void*)&args... };

        // Launch the kernel
        PUSH_RANGE("launchKernel", 0);
        CUDA_CHECK(cudaLaunchKernel(
            (const void*)k,
            grid,
            block,
            argPtrs,
            0,
            kStream
        ));
        cudaStreamSynchronize(kStream);
        POP_RANGE
    }

#endif //! __CUDA_UTILS_CUH__