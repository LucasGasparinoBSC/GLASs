/**
 * @file Kernels_Lv1.cuh
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

#ifndef __KERNELS_LV1_CUH__
#define __KERNELS_LV1_CUH__

#pragma once

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "DeviceUtils.hpp"

#if defined(USE_CUDA)
    // Parameters for kernel launches
    #define TILE_SIZE 256
    #define MAX_BLOCKS 10240
#elif defined(USE_HIP)
    // Parameters for kernel launches
    #define TILE_SIZE 256
    #define MAX_BLOCKS 10240
#endif

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
__global__ void axpy(const RTYPE a, const RTYPE* x, RTYPE* y, const ITYPE N);

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
__global__ void dot_product(const RTYPE* a, const RTYPE* b, double* r, ITYPE N);

// Scale kernel: x = a*x
// Receives:
// a: scalar multiplier
// x: input/output array
// N: size of array
// ITYPE: integer type for indexing (e.g., uint32_t, uint64_t)
// RTYPE: real type for computations (e.g., float, double, __nv_bfloat16)
template <typename ITYPE, typename RTYPE>
__global__ void scale(const RTYPE a, RTYPE* x, ITYPE N);

// Copy array kernel: y = x
// Receives:
// x: input array
// y: output array
// N: size of arrays
// ITYPE: integer type for indexing (e.g., uint32_t, uint64_t)
// RTYPE: real type for computations (e.g., float, double, __nv_bfloat16)
template <typename ITYPE, typename RTYPE>
__global__ void copy_array(const RTYPE* x, RTYPE* y, ITYPE N);

// Array sqrt
// Receives:
// x: input/output array
// N: size of array
// ITYPE: integer type for indexing (e.g., uint32_t, uint64_t)
// RTYPE: real type for computations (e.g., float, double, __nv_bfloat16)
template <typename ITYPE, typename RTYPE>
__global__ void array_sqrt(RTYPE* x, ITYPE N);

// Array invert: x_i = 1/x_i
// Receives:
// x: input/output array
// N: size of array
// ITYPE: integer type for indexing (e.g., uint32_t, uint64_t)
// RTYPE: real type for computations (e.g., float, double, __nv_bfloat16)
template <typename ITYPE, typename RTYPE>
__global__ void array_invert(RTYPE* x, ITYPE N);

// Pointwise multiply: y_i = x_i * y_i
// Receives:
// x: input array
// y: input/output array
// N: size of arrays
// ITYPE: integer type for indexing (e.g., uint32_t, uint64_t)
// RTYPE: real type for computations (e.g., float, double, __nv_bfloat16)
template <typename ITYPE, typename RTYPE>
__global__ void pointwise_multiply(const RTYPE* x, RTYPE* y, ITYPE N);

template <typename ITYPE, typename RTYPE>
__global__ void set_array(RTYPE* x, RTYPE value, ITYPE N);

template <typename ITYPE, typename RTYPE_IN, typename RTYPE_OUT>
__global__ void convert_array(const ITYPE N, const RTYPE_IN* input, RTYPE_OUT* output);

/*
// Kernel to fill a buffer with multiple args
template <typename ITYPE, typename RTYPE, typename... Ptrs>
__global__ void fill_buffer(RTYPE* buffer, const ITYPE nargs, Ptrs... args) {
    // Local array of device pointers
    RTYPE* tmps[] = { args... };

    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0) {
        for (ITYPE i = 0; i < nargs; i++) {
            buffer[i] = tmps[i][0];
        }
    }
}

// Opposite op: scatter from the buffer to the args
template <typename ITYPE, typename RTYPE, typename... Ptrs>
__global__ void scatter_buffer(RTYPE* buffer, const ITYPE nargs, Ptrs... args) {
    // Local array of device pointers
    RTYPE* tmps[] = { args... };

    ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0) {
        for (ITYPE i = 0; i < nargs; i++) {
            tmps[i][0] = buffer[i];
        }
    }
}

// Special launcher for fill_buffer
template <typename ITYPE, typename RTYPE, typename... Ptrs>
void launchFillBuffer(
    RTYPE* d_sbuffer,
    cudaStream_t kStream,
    Ptrs... d_args) {
        // Get number of arguments
        constexpr ITYPE nargs = sizeof...(d_args);

        // Alias for kernel call
        using FillKernel = void(*)(RTYPE*, const ITYPE, Ptrs...);
        FillKernel k = fill_buffer<ITYPE, RTYPE, Ptrs...>;

        // Build the args for cudaLaunchKernel
        const ITYPE nargs_cpy = nargs;
        void* argPtrs[] = {
            (void*)&d_sbuffer,
            (void*)&nargs_cpy,
            (void*)&d_args...
        };

        CUDA_CHECK(cudaLaunchKernel(
            (const void*)k,
            dim3(1),
            dim3(1),
            argPtrs,
            0,
            kStream
        ));
    }

// Special launcher for scatter_buffer
template <typename ITYPE, typename RTYPE, typename... Ptrs>
void launchScatterBuffer(
    RTYPE* d_rbuffer,
    cudaStream_t kStream,
    Ptrs... d_args) {
        // Get number of arguments
        constexpr ITYPE nargs = sizeof...(d_args);

        // Alias for kernel call
        using ScatterKernel = void(*)(RTYPE*, const ITYPE, Ptrs...);
        ScatterKernel k = scatter_buffer<ITYPE, RTYPE, Ptrs...>;

        // Build the args for cudaLaunchKernel
        const ITYPE nargs_cpy = nargs;
        void* argPtrs[] = {
            (void*)&d_rbuffer,
            (void*)&nargs_cpy,
            (void*)&d_args...
        };

        CUDA_CHECK(cudaLaunchKernel(
            (const void*)k,
            dim3(1),
            dim3(1),
            argPtrs,
            0,
            kStream
        ));
    }
*/

#endif //! __KERNELS_LV1_CUH__