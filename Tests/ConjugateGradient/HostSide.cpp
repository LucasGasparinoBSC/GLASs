#include "HostSide.hpp"

template <typename ITYPE, typename RTYPE>
void HostSide<ITYPE, RTYPE>::runSolver(ITYPE nrows, RTYPE* Adiag, ConjugateGradient<ITYPE, RTYPE>& solver) {
    #ifdef USE_GPU
        ITYPE blockSize = static_cast<ITYPE>(256);
        ITYPE maxBlocks = static_cast<ITYPE>(10240);
        ITYPE gridSize = (nrows + blockSize - 1) / blockSize;
        gridSize = (gridSize > maxBlocks) ? maxBlocks : gridSize;
        DeviceUtils::Stream_t kernelStream = solver.getKernelStream();
        auto MatVec = [=] __host__ (const RTYPE* x_in, RTYPE* x_out) {
            DeviceUtils::launchKernel(diagMatVec_device<ITYPE, RTYPE>, gridSize, blockSize, kernelStream, Adiag, x_in, x_out, nrows);
        };
    #else
        auto MatVec = [=] (const RTYPE* x_in, RTYPE* x_out) {
            diagMatVec_host(Adiag, x_in, x_out, nrows);
        };
    #endif
    solver.cgSolver(MatVec);
}

template class HostSide<uint32_t, float>;
template class HostSide<uint64_t, float>;
template class HostSide<uint32_t, double>;
template class HostSide<uint64_t, double>;
#ifdef USE_GPU
    template class HostSide<uint32_t, DeviceUtils::bf16>;
    template class HostSide<uint64_t, DeviceUtils::bf16>;
#endif