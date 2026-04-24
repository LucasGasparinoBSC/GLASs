#ifndef GMRES_HPP
#define GMRES_HPP   
#pragma once

#include "IterSolvers.hpp"
#include "../../MathOps/includes/ArnoldiIter.hpp"

// GMRES solver class, child of IterSolvers
template <typename ITYPE, typename RTYPE>
class GMRES : public IterSolvers<ITYPE, RTYPE>
{
    private:
        // GMRES-specific variables
        ITYPE restart; // Number of Krylov vectors before restart

        double *g;     // Right-hand side for least squares problem
        double *cs;   // Cosines for Givens rotations
        double *sn;   // Sines for Givens rotations
        RTYPE  *qj_tmp; // temporary for halo exchange

        // Arnoldi iteration object
        ArnoldiIter<ITYPE, RTYPE> arnoldi;

        // Apply Givens rotation to the columns of the Hessenberg matrix and g-local vector, no MPI 
        void applyGivens(int j);

        // Perform back substitution to solve the least squares problem Hy = g, no MPI
        void backSubstitution(int m, double *y);

        // Update the solution vector x using the computed Krylov basis and the least squares solution, no MPI
        void updateSolution(int m, double *x);


    public:
        // Interface for the base class types
        using Base = IterSolvers<ITYPE, RTYPE>;
        using typename Base::MatVecOp;
        using typename Base::ModVecOp;
        using typename Base::PrecondOp;
        using typename Base::HaloOp;

        // Empty constructor, calls parent empty constructor
        GMRES();

        // Param constructor with MPI_Comm
        GMRES(MPI_Comm& c_comm, ITYPE arrSize, ITYPE maxIters, double tol, ITYPE restart);

        // Destructor, calls parent destructor
        ~GMRES();

        // Implementations of the GMRESSolver:
        // non-preconditioned GMRES solver
        void gmresSolver(const MatVecOp &matvec, const HaloOp &halo);
};

#endif //! GMRES_HPPs