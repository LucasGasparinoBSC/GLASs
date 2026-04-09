#ifndef HOST_SIDE_HPP
#define HOST_SIDE_HPP

#ifdef USE_GPU
    #include "DeviceSide.cuh"
#endif
#include "ConjugateGradient.hpp"

template <typename ITYPE, typename RTYPE>
class HostSide
{
    private:
    public:
        static void runSolver(ITYPE nrows, RTYPE *Adiag, ConjugateGradient<ITYPE, RTYPE> &solver);
};

#endif //! HOST_SIDE_HPP