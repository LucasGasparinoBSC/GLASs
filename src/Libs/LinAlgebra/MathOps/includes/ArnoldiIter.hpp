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

#include "Comm_Utils.hpp"
#include "TensorUtils.hpp"

template <typename ITYPE, typename RTYPE>
class ArnoldiIter
{
    private:
    protected:
        using MatVecOp = std::function<void(const RTYPE *x_in, RTYPE *x_out)>;
        using ModVecOp = std::function<void(RTYPE *x_inout)>;
        ITYPE nRows;
        ITYPE maxIters;
        const ITYPE auxSize = 1;
        RTYPE *wOld;
        RTYPE *wNew;
        RTYPE* UpHess;
        RTYPE* Qkrylov;
        RTYPE* aux;
        double *tmpDot;
        double *mpiTmp;
        bool flag_planned = false;
        bool flag_setup = false;

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