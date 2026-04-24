#ifndef ARNOLDIITER_HPP
#define ARNOLDIITER_HPP

#pragma once

#ifdef USE_GPU
    #include "CUDA_Utils.cuh"
#else
    #define PUSH_RANGE(name, cid)
    #define POP_RANGE()
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
#include "TensorUtils.hpp"

template <typename ITYPE, typename RTYPE>
class ArnoldiIter : public EntryPoint<ITYPE, RTYPE>
{
    private:
    protected:
        ITYPE nRows;
        ITYPE maxIters;
        const ITYPE auxSize = 1;
        RTYPE *wOld, *d_wOld;  // previous basis vector q[i]
        RTYPE *wNew, *d_wNew;  // new vector being orthogonalized
        //RTYPE *UpHess, *d_UpHess;   // upper Hessenberg matrix (kIter+1 x kIter)
        double *UpHess, *d_UpHess;   // upper Hessenberg matrix (kIter+1 x kIter)
        RTYPE *Qkrylov, *d_Qkrylov;   // Krylov basis Q
        
        RTYPE *aux, *d_aux;   // auxiliary vector
        double *tmpDot, *d_tmpDot;  // temporary for dot product 
        double *mpiTmp, *d_mpiTmp;  // temporary for MPI reduction of dot product
        
        bool flag_planned = false;
        bool flag_setup = false;

        // Alias to inherited communication utilities
        Comm_Utils& ArnoldiIter_comm = this->entrypoint_comm;

        // Gram-Schmidt orthogonalization - private, called by arnoldiStep
        void GramSchmidt(ITYPE kIter);
    public:
        // Constructors
        // Empty
        ArnoldiIter();

        // Param.
        ArnoldiIter(ITYPE nRows_in, ITYPE maxIters_in);

        // Parameter constructor with MPI
        ArnoldiIter(MPI_Comm& c_comm, ITYPE nRows_in, ITYPE maxIters_in);   

        // Destructor
        ~ArnoldiIter();

        // Planner
        void plan(ITYPE nRows_in, ITYPE maxIters_in);

        // Setup
        void setup(RTYPE* v0);

        // Run one Arnoldi step at iteration j
        // matevc computes w = A*q[j]
        // returns h[j+1][j] -norm of new basis vector
        double arnoldiStep(ITYPE j, const typename EntryPoint<ITYPE,RTYPE>::MatVecOp& matvec);

        // Access to Krylov basis and upper Hessenberg matrix
        RTYPE* getQkrylov() { return this->Qkrylov; }
        //RTYPE* getUpHess() { return this->UpHess; }
        double* getUpHess() { return this->UpHess; }
};

#endif // ARNOLDIITER_HPP