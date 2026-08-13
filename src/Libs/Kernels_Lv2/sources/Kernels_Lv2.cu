/**
 * @file Kernels_Lv1.cu
 * @author Lucas Gasparino (lucas.gasparino3110@gmail.com)
 * @brief Specific template instantiations for device kernels in Kernels_Lv1
 * @version 0.1
 * @date 2025-10-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "Kernels_Lv2.cuh"

// Implementations for AMD are in the .cu file to avoid HIP compilation issues with templates. Check .cuh for CUDA implementations.
#if defined(USE_HIP)
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
#endif

template __global__ void extractRow<uint32_t, float>(const float* matrix, float* row, const uint32_t rowIndex, const uint32_t numCols);
template __global__ void extractRow<uint64_t, float>(const float* matrix, float* row, const uint64_t rowIndex, const uint64_t numCols);
template __global__ void extractRow<uint32_t, double>(const double* matrix, double* row, const uint32_t rowIndex, const uint32_t numCols);
template __global__ void extractRow<uint64_t, double>(const double* matrix, double* row, const uint64_t rowIndex, const uint64_t numCols);
template __global__ void extractRow<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16* matrix, DeviceUtils::bf16* row, const uint32_t rowIndex, const uint32_t numCols);
template __global__ void extractRow<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16* matrix, DeviceUtils::bf16* row, const uint64_t rowIndex, const uint64_t numCols);

template __global__ void extractColumn<uint32_t, float>(const float* matrix, float* column, const uint32_t colIndex, const uint32_t numRows, const uint32_t numCols);
template __global__ void extractColumn<uint64_t, float>(const float* matrix, float* column, const uint64_t colIndex, const uint64_t numRows, const uint64_t numCols);
template __global__ void extractColumn<uint32_t, double>(const double* matrix, double* column, const uint32_t colIndex, const uint32_t numRows, const uint32_t numCols);
template __global__ void extractColumn<uint64_t, double>(const double* matrix, double* column, const uint64_t colIndex, const uint64_t numRows, const uint64_t numCols);
template __global__ void extractColumn<uint32_t, DeviceUtils::bf16>(const DeviceUtils::bf16* matrix, DeviceUtils::bf16* column, const uint32_t colIndex, const uint32_t numRows, const uint32_t numCols);
template __global__ void extractColumn<uint64_t, DeviceUtils::bf16>(const DeviceUtils::bf16* matrix, DeviceUtils::bf16* column, const uint64_t colIndex, const uint64_t numRows, const uint64_t numCols);

template __global__ void guided_dotColumn<uint32_t, float>(const uint32_t* listEntries, const float* matrix, const float* vector, double* result, const uint32_t colIndex, const uint32_t numRows, const uint32_t numCols);
template __global__ void guided_dotColumn<uint64_t, float>(const uint64_t* listEntries, const float* matrix, const float* vector, double* result, const uint64_t colIndex, const uint64_t numRows, const uint64_t numCols);
template __global__ void guided_dotColumn<uint32_t, double>(const uint32_t* listEntries, const double* matrix, const double* vector, double* result, const uint32_t colIndex, const uint32_t numRows, const uint32_t numCols);
template __global__ void guided_dotColumn<uint64_t, double>(const uint64_t* listEntries, const double* matrix, const double* vector, double* result, const uint64_t colIndex, const uint64_t numRows, const uint64_t numCols);
template __global__ void guided_dotColumn<uint32_t, DeviceUtils::bf16>(const uint32_t* listEntries, const DeviceUtils::bf16* matrix, const DeviceUtils::bf16* vector, double* result, const uint32_t colIndex, const uint32_t numRows, const uint32_t numCols);
template __global__ void guided_dotColumn<uint64_t, DeviceUtils::bf16>(const uint64_t* listEntries, const DeviceUtils::bf16* matrix, const DeviceUtils::bf16* vector, double* result, const uint64_t colIndex, const uint64_t numRows, const uint64_t numCols);