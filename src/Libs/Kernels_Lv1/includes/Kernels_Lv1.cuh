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

// For CUDA, full implementation is here. For AMD, the implementation is in the .cu file to avoid HIP compilation issues with templates.
#if defined(USE_CUDA)
    template <typename ITYPE, typename RTYPE>
    __global__ void axpy(const RTYPE a, const RTYPE* x, RTYPE* y, const ITYPE N) {
        ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
        while (gid < N) {
            y[gid] += a * x[gid];
            gid += blockDim.x * gridDim.x;
        }
    }

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

    template <typename ITYPE, typename RTYPE>
    __global__ void scale(const RTYPE a, RTYPE* x, ITYPE N) {
        ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
        while (gid < N) {
            x[gid] *= a;
            gid += blockDim.x * gridDim.x;
        }
    }

    template <typename ITYPE, typename RTYPE>
    __global__ void copy_array(const RTYPE* x, RTYPE* y, ITYPE N) {
        ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
        while (gid < N) {
            y[gid] = x[gid];
            gid += blockDim.x * gridDim.x;
        }
    }

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

// Just declare the kernels for AMD, implementation is in the .cu file to avoid HIP compilation issues with templates.
#elif defined(USE_HIP)
    template <typename ITYPE, typename RTYPE>
    __global__ void axpy(const RTYPE a, const RTYPE* x, RTYPE* y, const ITYPE N);
    template <typename ITYPE, typename RTYPE>
    __global__ void dot_product(const RTYPE* a, const RTYPE* b, double* r, ITYPE N);
    template <typename ITYPE, typename RTYPE>
    __global__ void scale(const RTYPE a, RTYPE* x, ITYPE N);
    template <typename ITYPE, typename RTYPE>
    __global__ void copy_array(const RTYPE* x, RTYPE* y, ITYPE N);
    template <typename ITYPE, typename RTYPE>
    __global__ void array_sqrt(RTYPE* x, ITYPE N);
    template <typename ITYPE, typename RTYPE>
    __global__ void array_invert(RTYPE* x, ITYPE N);
    template <typename ITYPE, typename RTYPE>
    __global__ void pointwise_multiply(const RTYPE* x, RTYPE* y, ITYPE N);
    template <typename ITYPE, typename RTYPE>
    __global__ void set_array(RTYPE* x, RTYPE value, ITYPE N);
#endif


#endif //! __KERNELS_LV1_CUH__