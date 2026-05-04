#ifndef AUXKERNELS_CUH
#define AUXKERNELS_CUH

#include "DeviceUtils.hpp"

namespace AuxKernels
{
    // Only handle internal nodes, from 1 to n-2, inclusive.
    template <typename ITYPE, typename RTYPE>
    __global__ void matvec_nohalo(const RTYPE* cl, const RTYPE* dl, const RTYPE* el, const RTYPE* x, RTYPE* y, const ITYPE n)
    {
        ITYPE i = blockIdx.x * blockDim.x + threadIdx.x + 1;
        while (i < n - 1)
        {
            y[i] = cl[i -1] * x[i - 1] + dl[i] * x[i] + el[i + 1] * x[i + 1];
            i += blockDim.x * gridDim.x;
        }
    }

    // Handle halos with received data
    template <typename ITYPE, typename RTYPE>
    __global__ void matvec_halo(const RTYPE* cl, const RTYPE* dl, const RTYPE* el, const RTYPE* x, const RTYPE* ghosts, RTYPE* y, const ITYPE n)
    {
        if (threadIdx.x == 0)
        {
            y[0] = dl[0] * x[0] + el[1] * x[1] + cl[0] * ghosts[0];
            y[n - 1] = cl[n - 2] * x[n - 2] + dl[n - 1] * x[n - 1] + el[n - 1] * ghosts[1];
        }
    }
}

#endif // AUXKERNELS_CUH