#ifndef ITERSOLVERS_HPP
#define ITERSOLVERS_HPP

#pragma once

#ifdef USE_GPU
    #include "CUDA_Utils.cuh"
#else
    #define PUSH_RANGE(name,cid)
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
class IterSolvers : public EntryPoint<ITYPE, RTYPE>
{
    private:
    protected:
        bool flag_planned = false;
        bool flag_setup = false;
        ITYPE arrSize;
        const ITYPE auxSize = 1;
        ITYPE maxIters;
        ITYPE iter;
        double tol;
        // Vectors for the linear solver
        //    Host,   Device
        double *tmpDot, *d_tmpDot; // Temporary for dot products
        double *mpiTmp, *d_mpiTmp;  // Auxiliary single entry array for MPI_Allreduce
        RTYPE *x_sol, *d_x_sol; // Solution
        RTYPE *x0, *d_x0;       // Initial guess
        RTYPE *b, *d_b;         // RHS
        RTYPE *r0, *d_r0;       // Initial residual
        RTYPE *rk, *d_rk;       // Residual
        RTYPE *zk, *d_zk;       // Preconditioned residual
        RTYPE *Ax, *d_Ax;       // Matrix-vector product
        RTYPE *res0, *d_res0;   // Initial residual
        RTYPE *resk, *d_resk;   // Initial residual
        RTYPE *aux, *d_aux;     // Auxiliary single entry array

        // Optional comm_utils object (alias to inherited entrypoint_comm)
        Comm_Utils& IterSolvers_comm = this->entrypoint_comm;

        // CUDA-related variables
        #ifdef USE_GPU
            cudaStream_t kernelStream; // CUDA stream for kernel launches
            dim3 kernelGrid;
            dim3 kernelBlock;
            dim3 auxGrid;
            dim3 auxBlock;
        #endif

    public:
        // Empty constructor
        IterSolvers();

        // Param constructor
        IterSolvers(ITYPE arrSize, ITYPE maxIters, double tol);

        // Overload of param constructor with Comm_Utils obj
        IterSolvers(MPI_Comm& c_comm, ITYPE arrSize, ITYPE maxIters, double tol);

        // Destructor
        ~IterSolvers();

        // Solver plan
        void plan(ITYPE arrSize, ITYPE maxIters, double tol);

        // Solver setup
        void setup(RTYPE* inicond, RTYPE* rhs);

        // Get the solution back
        void getSolution(RTYPE* clientPtr);

        // Get the size of the arrays
        ITYPE getSize();

};

#endif //! ITERSOLVERS_HPP