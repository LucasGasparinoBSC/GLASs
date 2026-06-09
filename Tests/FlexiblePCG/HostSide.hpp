#ifndef HOSTSIDE_HPP
#define HOSTSIDE_HPP

#pragma once

#include "ConjugateGradient.hpp"
#if defined (USE_GPU)
    #include "AuxKernels.cuh"
#endif

template <typename ITYPE, typename RTYPE>
class HostSide
{
    private:
    public:
        static void setLocalSizes(const ITYPE globalSize, const int rank, const int nranks, ITYPE& localSize, ITYPE* sizesPerRank);
        static void generate_matrix(const ITYPE N, RTYPE *c, RTYPE *d, RTYPE *e);
        static void generate_inicond(const ITYPE N, RTYPE *x0);
        static void generate_rhs(const ITYPE N, RTYPE *b, const int rank);
        static void computeLeftRightRanks(const int irank, const int nranks, int& leftRank, int& rightRank);
        static void matvec_nohalo(const RTYPE* cl, const RTYPE* dl, const RTYPE* el, const RTYPE* x, RTYPE* y, const ITYPE n);
        static void matvec_halo(const RTYPE *cl, const RTYPE *dl, const RTYPE *el, const RTYPE *x, const RTYPE *ghosts, RTYPE *y, const ITYPE n);
};

template <typename ITYPE, typename RTYPE>
class Matvec
{
    public:
        static void fillBuffer(const RTYPE *cl, const RTYPE *el, const RTYPE *x, RTYPE *buf, const ITYPE n);
        static void launch_matvec(MPI_Win& win, const int irank, const int nranks, const int leftRank, const int rightRank,
                                  const RTYPE* cl, const RTYPE* dl, const RTYPE* el, const RTYPE* x, RTYPE* ghosts, RTYPE* y, const ITYPE n);
};

#endif //! HOSTSIDE_HPP