#include "DeviceUtils.hpp"

void DeviceUtils::StreamCreate(Stream_t* stream)
{
    #if defined(USE_CUDA)
        CUDA_CHECK(cudaStreamCreate(stream));
    #elif defined(USE_HIP)
        HIP_CHECK(hipStreamCreate(stream));
    #endif
}

void DeviceUtils::StreamDestroy(Stream_t stream)
{
    #if defined(USE_CUDA)
        CUDA_CHECK(cudaStreamDestroy(stream));
    #elif defined(USE_HIP)
        HIP_CHECK(hipStreamDestroy(stream));
    #endif
}

void DeviceUtils::StreamSynchronize(Stream_t stream)
{
    #if defined(USE_CUDA)
        CUDA_CHECK(cudaStreamSynchronize(stream));
    #elif defined(USE_HIP)
        HIP_CHECK(hipStreamSynchronize(stream));
    #endif
}