#ifndef DEVICEUTILS_HPP
#define DEVICEUTILS_HPP

#pragma once

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>

#if defined(USE_CUDA)
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_bf16.h>
    #include <nvtx3/nvToolsExt.h>
#elif defined(USE_HIP)
    #include <hip/hip_runtime.h>
    #include <hip/hip_bfloat16.h>
#endif

#if defined(USE_CUDA)
    // CUDA error handling macro
    #define CUDA_CHECK(call)                             \
        do                                               \
        {                                                \
            cudaError_t _err = (call);                   \
            if (_err != cudaSuccess)                     \
            {                                            \
                std::cerr << "CUDA Error: "              \
                          << cudaGetErrorString(_err)    \
                          << " (" << _err << ") at "     \
                          << __FILE__ << ":" << __LINE__ \
                          << std::endl;                  \
                std::exit(EXIT_FAILURE);                 \
            }                                            \
        } while (0)

    // Macro utility for NVTX ranges
    inline constexpr uint32_t colors[] = {0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff};
    inline constexpr int num_colors = sizeof(colors) / sizeof(uint32_t);

    // Push function for starting NVTX ranges
    #define PUSH_RANGE(name, cid)                                  \
        do {                                                   \
            int color_id = cid;                                \
            color_id = color_id % num_colors;                  \
            nvtxEventAttributes_t eventAttrib = {0};           \
            eventAttrib.version = NVTX_VERSION;                \
            eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
            eventAttrib.colorType = NVTX_COLOR_ARGB;           \
            eventAttrib.color = colors[color_id];              \
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
            eventAttrib.message.ascii = name;                  \
            nvtxRangePushEx(&eventAttrib);                     \
        } while (0)

    // Pop function for ending NVTX ranges
    #define POP_RANGE() nvtxRangePop()
#elif defined(USE_HIP)
    // HIP error handling macro
    #define HIP_CHECK(call)                              \
        do                                               \
        {                                                \
            hipError_t _err = (call);                    \
            if (_err != hipSuccess)                      \
            {                                            \
                std::cerr << "HIP Error: "               \
                          << hipGetErrorString(_err)     \
                          << " (" << _err << ") at "     \
                          << __FILE__ << ":" << __LINE__ \
                          << std::endl;                  \
                std::exit(EXIT_FAILURE);                 \
            }                                            \
        } while (0)

    // NVTX replacement, for now empty
    #define PUSH_RANGE(name, cid)
    #define POP_RANGE()
#endif

class DeviceUtils
{
    public:
        #if defined (USE_CUDA)
            using Stream_t = cudaStream_t;
            using Event_t = cudaEvent_t;
            using bf16 = __nv_bfloat16;
        #elif defined (USE_HIP)
            using Stream_t = hipStream_t;
            using Event_t = hipEvent_t;
            using bf16 = hip_bfloat16;
        #endif

        // Block instantion
        DeviceUtils() = delete; // Prevent instantiation of this class

        // Stream handling
        static void StreamCreate(Stream_t* stream);
        static void StreamDestroy(Stream_t stream);
        static void StreamSynchronize(Stream_t stream);

        // Kernel launch utility
        template <typename Kernel, typename... Args>
        static void launchKernel(Kernel k, dim3 grid, dim3 block, Stream_t kStream, Args &&...args)
        {
            void *argPtrs[] = {(void *)&args...};
            PUSH_RANGE("DeviceUtils::Kernel Launch", 0);
            #if defined(USE_CUDA)
                CUDA_CHECK(cudaLaunchKernel(
                    (const void*)k,
                    grid,
                    block,
                    argPtrs,
                    0,
                    kStream
                ));
            #elif defined(USE_HIP)
                HIP_CHECK(hipLaunchKernel(
                    (const void*)k,
                    grid,
                    block,
                    argPtrs,
                    0,
                    kStream
            ));
            #endif
            POP_RANGE();
        }

        // Data converters for bf16
        inline static bf16 floatToBf16(float val) {
            #if defined(USE_CUDA)
                return __float2bfloat16(val);
            #elif defined(USE_HIP)
                return hip_bfloat16(val);
            #endif
        }

        inline static float bf16ToFloat(bf16 val) {
            #if defined(USE_CUDA)
                return __bfloat162float(val);
            #elif defined(USE_HIP)
                uint32_t tmp = static_cast<uint32_t>(val.data) << 16; // Shift to upper 16 bits
                float out;
                std::memcpy(&out, &tmp, sizeof(float)); // Reinterpret bits as float
                return out;
            #endif
        }
};

#endif //! DEVICEUTILS_HPP