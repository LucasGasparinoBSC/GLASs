#ifndef ITERSOLVERS_HPP
#define ITERSOLVERS_HPP

#pragma once

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
        //     Host,    Device
        double *tmpDot, *d_tmpDot; // Auxiliaries for performing allreduces
        double *mpiTmp, *d_mpiTmp; // Auxiliary single entry array for MPI_Allreduce
        double *res0,   *d_res0;   // Initial residual
        double *resk,   *d_resk;   // Initial residual
        double *aux,    *d_aux;    // Auxiliary single entry array
        RTYPE  *x_sol,  *d_x_sol;  // Solution
        RTYPE  *x0,     *d_x0;     // Initial guess
        RTYPE  *b,      *d_b;      // RHS
        RTYPE  *r0,     *d_r0;     // Initial residual
        RTYPE  *rk,     *d_rk;     // Residual
        RTYPE  *zk,     *d_zk;     // Preconditioned residual
        RTYPE  *Ax,     *d_Ax;     // Matrix-vector product

        // Comm_Utils object (alias to inherited entrypoint_comm)
        Comm_Utils& IterSolvers_comm = this->entrypoint_comm;

        // GPU-related variables
        #ifdef USE_GPU
            DeviceUtils::Stream_t kernelStream; // stream for kernel launches
            dim3 kernelGrid;                    // Grid of blocks (main kernels)
            dim3 kernelBlock;                   // Threadblock definition (main kernels)
            dim3 auxGrid;                       // Grid of blocks (auxiliary kernels)
            dim3 auxBlock;                      // Threadblock definition (auxiliary kernels)
        #endif

    public:
        // Empty constructor
        IterSolvers();

        // Param constructor with Comm_Utils obj
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

        #ifdef USE_GPU
            DeviceUtils::Stream_t getKernelStream();
        #endif

};

#endif //! ITERSOLVERS_HPP