#ifndef CONJUGATEGRADIENT_H
#define CONJUGATEGRADIENT_H

#pragma once

#include "IterSolvers.hpp"

// Conjugate Gradient solver class, child of IterSolvers
template <typename ITYPE, typename RTYPE>
class ConjugateGradient : public IterSolvers<ITYPE, RTYPE>
{
    private:
        RTYPE *p0,    *d_p0;    // Search dir.
        RTYPE *alpha, *d_alpha; // Alpha param
        RTYPE *beta,  *d_beta;  // Beta param

    public:
        // Interface for the base class types
        using Base = IterSolvers<ITYPE, RTYPE>;
        using typename Base::MatVecOp;
        using typename Base::PrecondOp;
        using typename Base::ModVecOp;

        // Empty constructor, calls parent empty constructor
        ConjugateGradient();

        // Param constructor, calls parent param constructor
        ConjugateGradient(ITYPE arrSize, ITYPE maxIters, double tol);

        // Destructor, calls parent destructor
        ~ConjugateGradient();

        // Implementations of the CGsolver:

        // non-preconditioned CG solver
        void cgSolver(const MatVecOp &matvec);

        // Flexible PCG
        void fpcgSolver(const MatVecOp &matvec, const PrecondOp &precond);
};

#endif