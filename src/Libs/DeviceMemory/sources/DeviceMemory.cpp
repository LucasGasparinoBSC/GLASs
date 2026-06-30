#include "DeviceMemory.hpp"

template <typename ITYPE, typename VTYPE>
VTYPE* DeviceMemory<ITYPE, VTYPE>::deviceMalloc(const ITYPE &numEntries)
{
    PUSH_RANGE("DeviceMemory::deviceMalloc", 1);
    VTYPE *dPtr = nullptr;
    #if defined(USE_CUDA)
        // CUDA malloc
        CUDA_CHECK( cudaMalloc((void**)&dPtr, numEntries * sizeof(VTYPE)) );
    #elif defined(USE_HIP)
        // HIP malloc
        HIP_CHECK( hipMalloc((void**)&dPtr, numEntries * sizeof(VTYPE)) );
    #endif
    POP_RANGE();
    return dPtr;
}

template <typename ITYPE, typename VTYPE>
VTYPE *DeviceMemory<ITYPE, VTYPE>::deviceCalloc(const ITYPE &numEntries)
{
    PUSH_RANGE("DeviceMemory::deviceCalloc", 1);
    VTYPE *dPtr = nullptr;
    #if defined(USE_CUDA)
        // CUDA malloc
        CUDA_CHECK(cudaMalloc((void **)&dPtr, numEntries * sizeof(VTYPE)));
        CUDA_CHECK(cudaMemset(dPtr, 0, numEntries * sizeof(VTYPE)));
    #elif defined(USE_HIP)
        // HIP malloc
        HIP_CHECK( hipMalloc((void**)&dPtr, numEntries * sizeof(VTYPE)) );
        HIP_CHECK( hipMemset(dPtr, 0, numEntries * sizeof(VTYPE)) );
    #endif
    POP_RANGE();
    return dPtr;
}

template <typename ITYPE, typename VTYPE>
void DeviceMemory<ITYPE, VTYPE>::deviceFree(VTYPE *d_ptr)
{
    PUSH_RANGE("DeviceMemory::deviceFree", 1);
    #if defined(USE_CUDA)
        // CUDA free
        CUDA_CHECK( cudaFree(d_ptr) );
    #elif defined(USE_HIP)
        // HIP free
        HIP_CHECK( hipFree(d_ptr) );
    #endif
    POP_RANGE();
}

template <typename ITYPE, typename VTYPE>
void DeviceMemory<ITYPE, VTYPE>::copyHostToDevice(const ITYPE &numEntries, const VTYPE *hPtr, VTYPE *dPtr)
{
    PUSH_RANGE("DeviceMemory::copyHostToDevice", 1);
    #if defined(USE_CUDA)
        // CUDA copy host to device
        CUDA_CHECK( cudaMemcpy(dPtr, hPtr, numEntries * sizeof(VTYPE), cudaMemcpyHostToDevice) );
    #elif defined(USE_HIP)
        // HIP copy host to device
        HIP_CHECK( hipMemcpy(dPtr, hPtr, numEntries * sizeof(VTYPE), hipMemcpyHostToDevice) );
    #endif
    POP_RANGE();
}

template <typename ITYPE, typename VTYPE>
void DeviceMemory<ITYPE, VTYPE>::copyDeviceToHost(const ITYPE &numEntries, const VTYPE *dPtr, VTYPE *hPtr)
{
    PUSH_RANGE("DeviceMemory::copyDeviceToHost", 1);
    #if defined(USE_CUDA)
        // CUDA copy device to host
        CUDA_CHECK( cudaMemcpy(hPtr, dPtr, numEntries * sizeof(VTYPE), cudaMemcpyDeviceToHost) );
    #elif defined(USE_HIP)
        // HIP copy device to host
        HIP_CHECK( hipMemcpy(hPtr, dPtr, numEntries * sizeof(VTYPE), hipMemcpyDeviceToHost) );
    #endif
    POP_RANGE();
}

template class DeviceMemory<uint32_t, uint32_t>;
template class DeviceMemory<uint64_t, uint32_t>;
template class DeviceMemory<uint32_t, uint64_t>;
template class DeviceMemory<uint64_t, uint64_t>;
template class DeviceMemory<uint32_t, float>;
template class DeviceMemory<uint64_t, float>;
template class DeviceMemory<uint32_t, double>;
template class DeviceMemory<uint64_t, double>;
template class DeviceMemory<uint32_t, DeviceUtils::bf16>;
template class DeviceMemory<uint64_t, DeviceUtils::bf16>;