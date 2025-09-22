#ifndef __CUDA_UTILS_CUH__
#define __CUDA_UTILS_CUH__

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <cstdint>
#include <cmath>

// Tile size for dot product shared memory
#define TILE_SIZE 256

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

// Kernels
template <typename ITYPE, typename RTYPE>
__global__ void axpy(const RTYPE a, RTYPE* x, RTYPE* y, ITYPE N);

template <typename ITYPE, typename RTYPE>
__global__ void dot_product(const RTYPE* x, const RTYPE* y, RTYPE* result, ITYPE N);

// Macro to instantiate the templated kernels for specific types
#define INSTANTIATE_KERNELS(ITYPE, RTYPE) \
    template __global__ void axpy<ITYPE, RTYPE>(const RTYPE a, RTYPE* x, RTYPE* y, ITYPE N); \
    template __global__ void dot_product<ITYPE, RTYPE>(const RTYPE* x, const RTYPE* y, RTYPE* result, ITYPE N);

// Kernel launcher
// Generic templated CUDA kernel launcher
template <typename Kernel, typename... Args>
void launchKernel(
    Kernel k,
    dim3 grid,
    dim3 block,
    Args&&... args) {
        // Pointer of arguments
        void* argPtrs[] = { (void*)(&args)... };

        // Launch the kernel
        PUSH_RANGE("launchKernel", 0);
        CUDA_CHECK(cudaLaunchKernel(
            (const void*)k,
            grid,
            block,
            argPtrs,
            0,
            0
        ));
        POP_RANGE
    }

#endif //! __CUDA_UTILS_CUH__