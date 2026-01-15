#include "DeviceMemory.hpp"

template <typename ITYPE, typename VTYPE>
VTYPE* DeviceMemory<ITYPE, VTYPE>::deviceMalloc(const ITYPE &numEntries)
{
    VTYPE *dPtr = nullptr;
    #ifdef USE_GPU
        // CUDA malloc
        PUSH_RANGE("DeviceMemory::deviceMalloc", 0)
        CUDA_CHECK( cudaMalloc((void**)&dPtr, numEntries * sizeof(VTYPE)) );
        POP_RANGE
    #else
        // HIP malloc
        // TODO: Implement HIP malloc
        // hipMalloc((void**)&dPtr, numEntries * sizeof(VTYPE)); // Does not work yet!!!!!!!
        std::cerr << "HIP not implemented yet!" << std::endl;
        return(EXIT_FAILURE);
    #endif
    return dPtr;
}

template <typename ITYPE, typename VTYPE>
VTYPE *DeviceMemory<ITYPE, VTYPE>::deviceCalloc(const ITYPE &numEntries)
{
    VTYPE *dPtr = nullptr;
    #ifdef USE_GPU
        // CUDA malloc
        PUSH_RANGE("DeviceMemory::deviceCalloc", 0)
        CUDA_CHECK(cudaMalloc((void **)&dPtr, numEntries * sizeof(VTYPE)));
        CUDA_CHECK(cudaMemset(dPtr, 0, numEntries * sizeof(VTYPE)));
        POP_RANGE
    #else
        // HIP malloc
        // TODO: Implement HIP malloc
        // hipMalloc((void**)&dPtr, numEntries * sizeof(VTYPE)); // Does not work yet!!!!!!!
        std::cerr << "HIP not implemented yet!" << std::endl;
        return (EXIT_FAILURE);
    #endif
    return dPtr;
}

template <typename ITYPE, typename VTYPE>
void DeviceMemory<ITYPE, VTYPE>::deviceFree(VTYPE *d_ptr)
{
    #ifdef USE_GPU
        // CUDA free
        PUSH_RANGE("DeviceMemory::deviceFree", 0)
        CUDA_CHECK( cudaFree(d_ptr) );
        POP_RANGE
    #else
        // HIP free
        // TODO: Implement HIP free
        // hipFree(d_ptr); // Does not work yet!!!!!!!
        std::cerr << "HIP not implemented yet!" << std::endl;
    #endif
}

template <typename ITYPE, typename VTYPE>
void DeviceMemory<ITYPE, VTYPE>::copyHostToDevice(const ITYPE &numEntries, const VTYPE *hPtr, VTYPE *dPtr)
{
    #ifdef USE_GPU
        // CUDA copy host to device
        PUSH_RANGE("DeviceMemory::copyHostToDevice", 0)
        CUDA_CHECK( cudaMemcpy(dPtr, hPtr, numEntries * sizeof(VTYPE), cudaMemcpyHostToDevice) );
        POP_RANGE
    #else
        // HIP copy host to device
        // TODO: Implement HIP copy host to device
        // hipMemcpy(dPtr, hPtr, numEntries * sizeof(VTYPE), hipMemcpyHostToDevice); // Does not work yet!!!!!!!
        std::cerr << "HIP not implemented yet!" << std::endl;
    #endif
}

template <typename ITYPE, typename VTYPE>
void DeviceMemory<ITYPE, VTYPE>::copyDeviceToHost(const ITYPE &numEntries, const VTYPE *dPtr, VTYPE *hPtr)
{
    #ifdef USE_GPU
        // CUDA copy device to host
        PUSH_RANGE("DeviceMemory::copyDeviceToHost", 0)
        CUDA_CHECK( cudaMemcpy(hPtr, dPtr, numEntries * sizeof(VTYPE), cudaMemcpyDeviceToHost) );
        POP_RANGE
    #else
        // HIP copy device to host
        // TODO: Implement HIP copy device to host
        // hipMemcpy(hPtr, dPtr, numEntries * sizeof(VTYPE), hipMemcpyDeviceToHost); // Does not work yet!!!!!!!
        std::cerr << "HIP not implemented yet!" << std::endl;
    #endif
}

template class DeviceMemory<uint32_t, uint32_t>;
template class DeviceMemory<uint64_t, uint32_t>;
template class DeviceMemory<uint32_t, uint64_t>;
template class DeviceMemory<uint64_t, uint64_t>;
template class DeviceMemory<uint32_t, float>;
template class DeviceMemory<uint64_t, float>;
template class DeviceMemory<uint32_t, double>;
template class DeviceMemory<uint64_t, double>;
#ifdef USE_GPU
    template class DeviceMemory<uint32_t, __nv_bfloat16>;
    template class DeviceMemory<uint64_t, __nv_bfloat16>;
#else
    // HIP bfloat16 support to be added
#endif