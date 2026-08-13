#ifndef GMRES_HPP
#define GMRES_HPP

#pragma once

#include "IterSolvers.hpp"

// GMRES solver class, child of IterSolvers
template <typename ITYPE, typename RTYPE>
class GMRES : public IterSolvers<ITYPE, RTYPE>
{
    private:
        using Base = IterSolvers<ITYPE, RTYPE>;
        using typename Base::MatVecOp;
        using typename Base::ModVecOp;
        using typename Base::PrecondOp;

        double *beta, *d_beta; // Beta param (res. norm)
        RTYPE  *w_k,  *d_w_k;  // Auxiliary vector for the Arnoldi process
        RTYPE  *Q_k,  *d_Q_k;  // Krylov basis N x mIters+1
        RTYPE  *H_k,  *d_H_k;  // Upper Hessenberg matrix mIters+1 x mIters

        // Comms variables
        const int nargs = static_cast<int>(2);
        double *sendbuf, *d_sendbuf;
        double *recvbuf, *d_recvbuf;
        double *dotTmp1, *d_dotTmp1;
        double *dotTmp2, *d_dotTmp2;

        // Arnoldi iteration, shared for all GMRES members
        void arnoldiIteration(ITYPE kIter, const MatVecOp &matvec);

    public:

        // Empty constructor, calls parent empty constructor
        GMRES();

        // Param constructor with MPI_Comm
        GMRES(MPI_Comm& c_comm, ITYPE arrSize, ITYPE arrSizeList, ITYPE maxIters, double tol);

        // Destructor, calls parent destructor
        ~GMRES();

        // Implementations of the GMRES solver:

        // Simple GMRES
        void basicSolver(const MatVecOp &matvec);

};

#endif //! GMRES_HPP