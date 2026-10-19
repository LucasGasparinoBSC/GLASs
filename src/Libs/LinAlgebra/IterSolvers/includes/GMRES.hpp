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

        // Basic constants
        const RTYPE zero = static_cast<RTYPE>(0);
        const RTYPE one = static_cast<RTYPE>(1);
        const double zeroFP64 = static_cast<double>(0);
        const double oneFP64 = static_cast<double>(1);

        // GMRES variables
        // NOTE: H_k is column-major stored (transposed), so we need to access it accordingly
        // NOTE: Q_k is column-major stored (transposed), so we need to access it accordingly
        ITYPE HK_size;
        ITYPE QK_size;
        double *beta, *d_beta; // Norm of the rhs
        double *hk,   *d_hk;   // Scalar aux. for arnoldi
        double *norm, *d_norm; // Vector norm auxiliary
        double *cs,   *d_cs;   // Cosine values for Givens rotations
        double *sn,   *d_sn;   // Sine values for Givens rotations
        double *gk,   *d_gk;   // Norm tracker
        double *yk,   *d_yk;   // LSQ solution vector
        double *H_k,   *d_H_k;   // Hessenberg matrix
        RTYPE  *wk,   *d_wk;   // Auxiliary vector for Arnoldi/cgsr2p
        RTYPE  *qk,   *d_qk;   // Current Krylov vector
        RTYPE  *Q_k,  *d_Q_k;  // Krylov basis matrix
        bool arnoldiConverged; // Flag for Arnoldi convergence

        // Comms variables
        const int nargs = static_cast<int>(2);
        double *sendbuf, *d_sendbuf;
        double *recvbuf, *d_recvbuf;
        double *dotTmp1, *d_dotTmp1;
        double *dotTmp2, *d_dotTmp2;

        // Auxiliary methods:

        // Compute cosine and sine for Givens rotation
        void compute_givens_rotation(const double v1, const double v2, double &cs_k, double &sn_k);

        // Apply the givens rotation to the Hessenberg matrix
        void apply_givens_rotation(const ITYPE k);

        // Arnoldi process to compute Q_k and H_k
        // NOTE: passing preconditioner as an optional argument, if not provided, it will be ignored
        bool arnoldi(const MatVecOp &&MV_Op, const ITYPE k);
        bool arnoldi(const MatVecOp &&MV_Op, const PrecondOp &&Prec_Op, const ITYPE k);

        // CGSR-2p process to compute Q_k and H_k with preconditioner
        bool cgsr_2p(const MatVecOp &&MV_Op, const ITYPE k);
        bool cgsr_2p(const MatVecOp &&MV_Op, const PrecondOp &&Prec_Op, const ITYPE k);

    public:

        // Empty constructor, calls parent empty constructor
        GMRES();

        // Param constructor with MPI_Comm
        GMRES(MPI_Comm& c_comm, ITYPE arrSize, ITYPE arrSizeList, ITYPE maxIters, double tol);

        // Destructor, calls parent destructor
        ~GMRES();

};

#endif //! GMRES_HPP