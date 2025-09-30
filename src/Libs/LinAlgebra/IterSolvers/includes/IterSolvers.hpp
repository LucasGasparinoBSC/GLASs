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

using MatVecOp = std::function<void(const float *x_in, float *x_out)>;
using ModVecOp = std::function<void(float *x_inout)>;

template <typename ITYPE, typename RTYPE>
class IterSolvers
{
    private:
    protected:
        bool flag_planned = false;
        bool flag_setup = false;
        ITYPE arrSize;
        ITYPE maxIters;
        ITYPE iter;
        double tol;
        // Vectors for the linear solver
        //    Host,   Device
        RTYPE *x_sol, *d_x_sol; // Solution
        RTYPE *x0, *d_x0;       // Initial guess
        RTYPE *r0, *d_r0;       // Initial residual
        RTYPE *p0, *d_p0;       // Search dir.
        RTYPE *rk, *d_rk;       // Residual
        RTYPE *zk, *d_zk;       // Preconditioned residual
        RTYPE *Ax, *d_Ax;       // Matrix-vector product
        RTYPE *b, *d_b;         // RHS
        RTYPE *res0, *d_res0;   // Initial residual
        RTYPE *resk, *d_resk;   // Initial residual
        RTYPE *aux, *d_aux;     // Auxiliary single entry array

    public:
        // Empty constructor
        IterSolvers();

        // Param constructor
        IterSolvers(ITYPE arrSize, ITYPE maxIters, double tol);

        // Destructor
        ~IterSolvers();

        // Solver plan
        void plan(ITYPE arrSize, ITYPE maxIters, double tol);

        // Solver setup
        void setup(RTYPE* inicond, RTYPE* rhs);

        // Pure virtual solve method to be implemented by derived classes
        virtual void solve(MatVecOp& matVec, ModVecOp& applyBC) = 0;

};

#endif //! ITERSOLVERS_HPP