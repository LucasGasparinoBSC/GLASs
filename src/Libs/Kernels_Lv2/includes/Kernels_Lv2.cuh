/**
 * @file Kernels_Lv2.cuh
 * @author Lucas Gasparino (lucas.gasparino3110@gmail.com)
 * @brief Kernel library for handling matrices and matrix-vector products
 * @details 
 * @version 0.1
 * @date 2026-04-13
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef __KERNELS_LV2_CUH__
#define __KERNELS_LV2_CUH__

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
#elif defined(USE_HIP)
    // Parameters for kernel launches
#endif

// For CUDA, full implementation is here. For AMD, the implementation is in the .cu file to avoid HIP compilation issues with templates.
#if defined(USE_CUDA)

    // Matrix manipulation kernels

    // Extract row
    template <typename ITYPE, typename RTYPE>
    __global__ void extractRow(const RTYPE* matrix, RTYPE* row, const ITYPE rowIndex, const ITYPE numCols) {
        ITYPE colIndex = blockIdx.x * blockDim.x + threadIdx.x;
        while (colIndex < numCols) {
            row[colIndex] = matrix[rowIndex * numCols + colIndex];
            colIndex += blockDim.x * gridDim.x;
        }
    }

    // Extract column
    template <typename ITYPE, typename RTYPE>
    __global__ void extractColumn(const RTYPE* matrix, RTYPE* column, const ITYPE colIndex, const ITYPE numRows, const ITYPE numCols) {
        ITYPE rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
        while (rowIndex < numRows) {
            column[rowIndex] = matrix[rowIndex * numCols + colIndex];
            rowIndex += blockDim.x * gridDim.x;
        }
    }

    // BLAS-1 kernels involving matrx column access

    // Guided dot product of a matrix column with a vector, using a list of entries to guide the computation (owned nodes, periodicity...)
    template <typename ITYPE, typename RTYPE>
    __global__ void guided_dotColumn(const ITYPE* listEntries, const RTYPE* matrix, const RTYPE* vector, double* result, const ITYPE colIndex, const ITYPE numRows, const ITYPE numCols) {
        ITYPE gid = blockIdx.x * blockDim.x + threadIdx.x;
        ITYPE tid = threadIdx.x;

        __shared__ double cache[TILE_SIZE];
        double value = 0.0;
        while (gid < numRows) {
            ITYPE idx = listEntries[gid];
            value += static_cast<double>(matrix[idx * numCols + colIndex] * vector[idx]);
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
        if (tid == 0) {
            atomicAdd(result, cache[0]);
        }
    }

// Just declare the kernels for AMD, implementation is in the .cu file to avoid HIP compilation issues with templates.
#elif defined(USE_HIP)
    template <typename ITYPE, typename RTYPE>
    __global__ void extractRow(const RTYPE* matrix, RTYPE* row, const ITYPE rowIndex, const ITYPE numCols);
    template <typename ITYPE, typename RTYPE>
    __global__ void extractColumn(const RTYPE* matrix, RTYPE* column, const ITYPE colIndex, const ITYPE numRows, const ITYPE numCols);
    template <typename ITYPE, typename RTYPE>
    __global__ void guided_dotColumn(const ITYPE* listEntries, const RTYPE* matrix, const RTYPE* vector, double* result, const ITYPE colIndex, const ITYPE numRows, const ITYPE numCols);
#endif


#endif //! __KERNELS_LV2_CUH__