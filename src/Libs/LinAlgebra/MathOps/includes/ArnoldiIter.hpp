#ifndef ARNOLDIITER_HPP
#define ARNOLDIITER_HPP

#pragma once

#ifdef USE_GPU
    #include "CUDA_Utils.cuh"
#else
    #define PUSH_RANGE(name, cid)
    #define POP_RANGE
#endif

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <functional>
#include <algorithm>

#include "EntryPoint.hpp"

template <typename ITYPE, typename RTYPE>
class ArnoldiIter : public EntryPoint<ITYPE, RTYPE>
{
    private:
    protected:
        ITYPE nRows;
        ITYPE maxIters;
        const ITYPE auxSize = 1;
        RTYPE *wOld, *d_wOld;
        RTYPE *wNew, *d_wNew;
        RTYPE *UpHess, *d_UpHess;
        RTYPE *Qkrylov, *d_Qkrylov;
        RTYPE *aux, *d_aux;
        double *tmpDot, *d_tmpDot;
        double *mpiTmp, *d_mpiTmp;
        bool flag_planned = false;
        bool flag_setup = false;

        // Alias to inherited communication utilities
        Comm_Utils& ArnoldiIter_comm = this->entrypoint_comm;

        void GramSchmidt(ITYPE kIter);
    public:
        // Constructors
        // Empty
        ArnoldiIter();

        // Param.
        ArnoldiIter(ITYPE nRows_in, ITYPE maxIters_in);

        // Destructor
        ~ArnoldiIter();

        // Planner
        void plan(ITYPE nRows_in, ITYPE maxIters_in);

        // Setup
        void setup(RTYPE* v0);
};

#endif // ARNOLDIITER_HPP