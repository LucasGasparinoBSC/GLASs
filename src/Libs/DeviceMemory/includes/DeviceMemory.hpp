#ifndef DEVICEMEMORY_HPP
#define DEVICEMEMORY_HPP

#pragma once

#include "DeviceUtils.hpp"

template <typename ITYPE, typename VTYPE>
class DeviceMemory
{
    public:
        static VTYPE *deviceMalloc(const ITYPE &numEntries);
        static VTYPE *deviceCalloc(const ITYPE &numEntries);
        static void deviceFree(VTYPE *d_ptr);
        static void copyHostToDevice(const ITYPE &numEntries, const VTYPE *hPtr, VTYPE *dPtr);
        static void copyDeviceToHost(const ITYPE &numEntries, const VTYPE *dPtr, VTYPE *hPtr);
};

#endif //! DEVICEMEMORY_HPP