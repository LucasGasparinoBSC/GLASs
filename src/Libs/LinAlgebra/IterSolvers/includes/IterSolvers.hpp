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

#include "Comm_Utils.hpp"
#include "TensorUtils.hpp"

template <typename ITYPE, typename RTYPE>
class IterSolvers
{
    private:
    protected:
        using MatVecOp = std::function<void(const RTYPE *x_in, RTYPE *x_out)>;
        using PrecondOp = std::function<void(const RTYPE *x_in, RTYPE *x_out)>;
        using ModVecOp = std::function<void(RTYPE *x_inout)>;
        bool flag_planned = false;
        bool flag_setup = false;
        ITYPE arrSize;
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

        // Optional comm_utils object
        Comm_Utils IterSolvers_comm;

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
        RTYPE* getSolution();

        // Get the size of the arrays
        ITYPE getSize();

};

#endif //! ITERSOLVERS_HPP