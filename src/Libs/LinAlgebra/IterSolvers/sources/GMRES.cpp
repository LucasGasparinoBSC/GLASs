#include "GMRES.hpp"

// Arnoldi iter process
template <typename ITYPE, typename RTYPE>
void GMRES<ITYPE, RTYPE>::arnoldiIteration(ITYPE kIter, const MatVecOp& matvec) {
    PUSH_RANGE("GMRES::arnoldiIteration", 4);
    #if defined(USE_GPU)
        matvec(this->d_zk, this->d_Ax);

        for (ITYPE j = 0; j <= kIter; ++j) {
            // Extract Q_k[:][j] to w_k

            // h = dot(Ax, w_k)

            // Store H_k[j][k] = h

            // Ax = Ax - h * w_k
        }

    #else
        // Compute w = A * zk, where zk is the preconditioned residual (assumes right preconditioning)
        matvec(this->zk, this->Ax);

        // Modified Gram-Schmidt process to orthogonalize w against the existing Krylov basis Q_k
        for (ITYPE j = 0; j <= kIter; ++j) {
        }
    #endif
    POP_RANGE();
}

template <typename ITYPE, typename RTYPE>
// empty constructor for flexibility in initialization
GMRES<ITYPE, RTYPE>::GMRES() : IterSolvers<ITYPE, RTYPE>() {
    PUSH_RANGE("GMRES::Constructor(empty)", 4);
    this->beta = nullptr;
    this->d_beta = nullptr;
    this->w_k = nullptr;
    this->d_w_k = nullptr;
    this->Q_k = nullptr;
    this->d_Q_k = nullptr;
    this->H_k = nullptr;
    this->d_H_k = nullptr;

    this->sendbuf = nullptr;
    this->d_sendbuf = nullptr;
    this->recvbuf = nullptr;
    this->d_recvbuf = nullptr;
    this->dotTmp1 = nullptr;
    this->d_dotTmp1 = nullptr;
    this->dotTmp2 = nullptr;
    this->d_dotTmp2 = nullptr;
    POP_RANGE();
}

template <typename ITYPE, typename RTYPE>
// constructor with parameters and communicator
GMRES<ITYPE, RTYPE>::GMRES(MPI_Comm& c_comm, ITYPE arrSize, ITYPE arrSizeList, ITYPE maxIters, double tol) : IterSolvers<ITYPE, RTYPE>(c_comm, arrSize, arrSizeList, maxIters, tol) {
    if (this->IterSolvers_comm.getWorldRank() == 0) std::cout << "--| IterSolvers: using GMRES solver!" << std::endl;
    PUSH_RANGE("GMRES::Constructor(param+comm)", 4);
    // Allocate host memory using calloc (ensures init to 0)
    this->beta = (double *)calloc(this->auxSize, sizeof(double));
    this->w_k = (RTYPE *)calloc(this->arrSize, sizeof(RTYPE));
    this->Q_k = (RTYPE *)calloc(this->arrSize * (this->maxIters + 1), sizeof(RTYPE));
    this->H_k = (RTYPE *)calloc((this->maxIters + 1) * this->maxIters, sizeof(RTYPE));
    this->sendbuf = (double *)calloc(this->nargs, sizeof(double));
    this->recvbuf = (double *)calloc(this->nargs, sizeof(double));
    this->dotTmp1 = (double *)calloc(this->auxSize, sizeof(double));
    this->dotTmp2 = (double *)calloc(this->auxSize, sizeof(double));

    #ifdef USE_GPU
        // Allocate device arrays
        d_beta = DeviceMemory<ITYPE, double>::deviceCalloc(this->auxSize);
        d_w_k = DeviceMemory<ITYPE, RTYPE>::deviceCalloc(this->arrSize);
        d_Q_k = DeviceMemory<ITYPE, RTYPE>::deviceCalloc(this->arrSize * (this->maxIters + 1));
        d_H_k = DeviceMemory<ITYPE, RTYPE>::deviceCalloc((this->maxIters + 1) * this->maxIters);
        d_sendbuf = DeviceMemory<ITYPE, double>::deviceCalloc(this->nargs);
        d_recvbuf = DeviceMemory<ITYPE, double>::deviceCalloc(this->nargs);
        d_dotTmp1 = DeviceMemory<ITYPE, double>::deviceCalloc(this->auxSize);
        d_dotTmp2 = DeviceMemory<ITYPE, double>::deviceCalloc(this->auxSize);
    #endif

    // Start logfile
    this->logfile_name = "GLASs_ gmresSolver";
    if (this->IterSolvers_comm.getLibRank() == 0) {
        // Open the file for writing
        this->logfile.open(this->logfile_name + this->logfile_ext, std::ios::out);
        // Wrtie the header: "ITER ||rk|| cgTime(ms)"
        this->logfile << "ITER --- "
                      << " --- ||rk|| --- "
                      << " --- gmresTime(ms)" << std::endl;
        this->logfile << "-------------------------------" << std::endl;
        this->logfile.flush();
    }
    POP_RANGE();
}

template <typename ITYPE, typename RTYPE>
GMRES<ITYPE, RTYPE>::~GMRES() {
    if (this->IterSolvers_comm.getWorldRank() == 0) std::cout << "--| IterSolvers: destroying GMRES solver" << std::endl;

    // Close logfile
    if (this->IterSolvers_comm.getLibRank() == 0) this->logfile.close();

    PUSH_RANGE("GMRES::Destructor", 4);
    // Free host memory
    free(this->beta);
    free(this->w_k);
    free(this->Q_k);
    free(this->H_k);
    free(this->sendbuf);
    free(this->recvbuf);
    free(this->dotTmp1);
    free(this->dotTmp2);

    #ifdef USE_GPU
        // Free device memory
        DeviceMemory<ITYPE, double>::deviceFree(this->d_beta);
        DeviceMemory<ITYPE, RTYPE>::deviceFree(this->d_w_k);
        DeviceMemory<ITYPE, RTYPE>::deviceFree(this->d_Q_k);
        DeviceMemory<ITYPE, RTYPE>::deviceFree(this->d_H_k);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_sendbuf);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_recvbuf);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_dotTmp1);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_dotTmp2);
    #endif
    POP_RANGE();
}

//-------------------------//
// Solver implementations  //
//-------------------------//
// Non-preconditioned Conjugate Gradient solver
template <typename ITYPE, typename RTYPE>
void GMRES<ITYPE, RTYPE>::basicSolver(const MatVecOp& matvec) {
    double out_sqrtRes = static_cast<double>(0);
    double cgTime = this->IterSolvers_comm.timeFunction([&]
    {
        PUSH_RANGE("GMRES::basicSolver", 4);
        // Basic scalars
        this->iter = 0;
        RTYPE zero = static_cast<RTYPE>(0);
        RTYPE negOne = static_cast<RTYPE>(-1);
        double zero_fp64 = static_cast<double>(0);

        // GMRES iterations
        for (this->iter = 0; this->iter < this->maxIters; ++this->iter) {
            #ifdef USE_GPU
            #else
            #endif
        }
        POP_RANGE(); // 4
    });
    if (this->IterSolvers_comm.getLibRank() == 0)
    {
        // Write to logfile
        // out_sqrtRes in scientific format
        this->logfile << this->iter << " , " << std::scientific << out_sqrtRes << " , " << cgTime * 1000.0 << std::endl;
        this->logfile.flush();
    }
}

template class GMRES<uint32_t, float>;
template class GMRES<uint64_t, float>;
template class GMRES<uint32_t, double>;
template class GMRES<uint64_t, double>;
#ifdef USE_GPU
    template class GMRES<uint32_t, DeviceUtils::bf16>;
    template class GMRES<uint64_t, DeviceUtils::bf16>;
#endif
