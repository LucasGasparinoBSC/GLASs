#ifndef CONJUGATEGRADIENT_HPP
#define CONJUGATEGRADIENT_HPP

#pragma once

#include "IterSolvers.hpp"

// Conjugate Gradient solver class, child of IterSolvers
template <typename ITYPE, typename RTYPE>
class ConjugateGradient : public IterSolvers<ITYPE, RTYPE>
{
    private:
        RTYPE  *p0,    *d_p0;    // Search dir.
        double *alpha, *d_alpha; // Alpha param
        double *beta,  *d_beta;  // Beta param

        // Comms variables
        const int nargs = static_cast<int>(2);
        double *sendbuf, *d_sendbuf;
        double *recvbuf, *d_recvbuf;
        double *dotTmp1, *d_dotTmp1;
        double *dotTmp2, *d_dotTmp2;

    public:
        // Interface for the base class types
        using Base = IterSolvers<ITYPE, RTYPE>;
        using typename Base::MatVecOp;
        using typename Base::ModVecOp;
        using typename Base::PrecondOp;
        using typename Base::HaloOp;

        // Empty constructor, calls parent empty constructor
        ConjugateGradient();

        // Param constructor with MPI_Comm
        ConjugateGradient(MPI_Comm& c_comm, ITYPE arrSize, ITYPE maxIters, double tol);

        // Destructor, calls parent destructor
        ~ConjugateGradient();

        // Implementations of the CGsolver:

        // non-preconditioned CG solver
        void cgSolver(const MatVecOp &matvec);

        // Flexible PCG
        void fpcgSolver(const MatVecOp &matvec, const PrecondOp &precond);
};

#endif //! CONJUGATEGRADIENT_HPP