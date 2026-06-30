#include "EntryPoint.hpp"

// Explicit template instantiations for EntryPoint
template class EntryPoint<uint32_t, float>;
template class EntryPoint<uint64_t, float>;
template class EntryPoint<uint32_t, double>;
template class EntryPoint<uint64_t, double>;
#ifdef USE_GPU
    template class EntryPoint<uint32_t, DeviceUtils::bf16>;
    template class EntryPoint<uint64_t, DeviceUtils::bf16>;
#endif