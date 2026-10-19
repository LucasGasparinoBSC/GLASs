#include "GMRES.hpp"

// Auxiliary methods:

// Compute givens rotation coefficients for Givens rotation
template <typename ITYPE, typename RTYPE>
void GMRES<ITYPE, RTYPE>::compute_givens_rotation(const double v1, const double v2, double &cs_k, double &sn_k) {
    PUSH_RANGE("GMRES::compute_givens_rotation", 0);
    if (v2 == 0.0) {
        cs_k = 1.0;
        sn_k = 0.0;
    } else if (v1 == 0.0) {
        cs_k = 0.0;
        sn_k = -1.0;
    } else {
        double r = std::hypot(v1, v2);
        cs_k = v1 / r;
        sn_k = -v2 / r;
    }
    POP_RANGE();
}

// Apply Givens rotation to the Hessenberg matrix and gk
template <typename ITYPE, typename RTYPE>
void GMRES<ITYPE, RTYPE>::apply_givens_rotation(const ITYPE k) {

    PUSH_RANGE("GMRES::apply_givens_rotation", 1);

    // Create a pointer to the k-th row of H_k
    double* hrow = H_k + k * (this->maxIters + 1);

    // Apply previous Givens rotations to the new column of H_k
    // NOTE: for k=0, this loop will not execute, which is correct
    for (ITYPE i = 0; i < k; ++i) {
        double temp = cs[i] * hrow[i] - sn[i] * hrow[i + 1];
        hrow[i + 1] = sn[i] * hrow[i] + cs[i] * hrow[i + 1];
        hrow[i] = temp;
    }

    // Compute the new Givens rotation
    compute_givens_rotation(hrow[k], hrow[k + 1], cs[k], sn[k]);

    // Eliminate the sub-diagonal entry
    double temp = cs[k] * hrow[k] - sn[k] * hrow[k + 1];
    hrow[k + 1] = 0.0;
    hrow[k] = temp;

    // Update gk with the new Givens rotation
    gk[k + 1] = sn[k] * gk[k];
    gk[k] = cs[k] * gk[k];

    POP_RANGE();
}

// Arnoldi process to produce Q_k and H_k entries

// No preconditioner version
template <typename ITYPE, typename RTYPE>
bool GMRES<ITYPE, RTYPE>::arnoldi(const MatVecOp &&MV_Op, const ITYPE k) {
    PUSH_RANGE("GMRES::arnoldi", 0);
    #ifdef USE_GPU
        // w = A * qk
        MV_Op(d_qk, d_wk);

        // MGS
        for (ITYPE j = 0; j <= k; ++j) {
            // h_jk = dot(wk, Q_k[:, j])

            // Copy hk to host, then store it in H_k
            DeviceMemory<ITYPE, double>::deviceToHost(1, d_hk, hk);
            H_k[k * (this->maxIters + 1) + j] = hk[0];

            // wk = wk - h_jk * Q_k[:, j]
        }

        // h_(k+1)k = ||wk||

    #else
        // w = A * qk
        MV_Op(qk, wk);

        // MGS
        for (ITYPE j = 0; j <= k; ++j) {
            // h_jk = dot(wk, Q_k[:, j])

            // Store hk into H
            H_k[k * (this->maxIters + 1) + j] = hk[0];

            // wk = wk - h_jk * Q_k[:, j]
        }

        // h_(k+1)k = ||wk||
    #endif
    POP_RANGE();
}

template <typename ITYPE, typename RTYPE>
// empty constructor for flexibility in initialization
GMRES<ITYPE, RTYPE>::GMRES() : IterSolvers<ITYPE, RTYPE>() {
    PUSH_RANGE("GMRES::Constructor(empty)", 4);
    HK_size = 0;
    QK_size = 0;
    this->beta = nullptr;
    this->d_beta = nullptr;
    this->hk = nullptr;
    this->d_hk = nullptr;
    this->norm = nullptr;
    this->d_norm = nullptr;
    this->cs = nullptr;
    this->d_cs = nullptr;
    this->sn = nullptr;
    this->d_sn = nullptr;
    this->gk = nullptr;
    this->d_gk = nullptr;
    this->yk = nullptr;
    this->d_yk = nullptr;
    this->H_k = nullptr;
    this->d_H_k = nullptr;
    this->wk = nullptr;
    this->d_wk = nullptr;
    this->qk = nullptr;
    this->d_qk = nullptr;
    this->Q_k = nullptr;
    this->d_Q_k = nullptr;
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

    // Allocate memory for GMRES variables
    // Remember: H_k and Qk are transposed!
    HK_size = this->maxIters * (this->maxIters + 1); // H_k is m x m+1
    QK_size = (this->maxIters + 1) * this->arrSize;  // Q_k is m+1 x n
    this->beta = (double*)calloc(this->auxSize, sizeof(double));
    this->hk = (double*)calloc(this->auxSize, sizeof(double));
    this->norm = (double*)calloc(this->auxSize, sizeof(double));
    this->cs = (double*)calloc(this->maxIters, sizeof(double));
    this->sn = (double*)calloc(this->maxIters, sizeof(double));
    this->gk = (double*)calloc((this->maxIters + 1), sizeof(double));
    this->yk = (double*)calloc((this->maxIters + 1), sizeof(double));
    this->H_k = (double*)calloc(HK_size, sizeof(double));
    this->wk = (RTYPE*)calloc(this->arrSize, sizeof(RTYPE));
    this->qk = (RTYPE*)calloc(this->arrSize, sizeof(RTYPE));
    this->Q_k = (RTYPE*)calloc(QK_size, sizeof(RTYPE));
    this->sendbuf = (double*)calloc(this->nargs, sizeof(double));
    this->recvbuf = (double*)calloc(this->nargs, sizeof(double));
    this->dotTmp1 = (double*)calloc(this->auxSize, sizeof(double));
    this->dotTmp2 = (double*)calloc(this->auxSize, sizeof(double));
    #ifdef USE_GPU
        // Allocate device arrays
        d_beta = DeviceMemory<ITYPE, double>::deviceCalloc(this->auxSize);
        d_hk = DeviceMemory<ITYPE, double>::deviceCalloc(this->auxSize);
        d_norm = DeviceMemory<ITYPE, double>::deviceCalloc(this->auxSize);
        d_cs = DeviceMemory<ITYPE, double>::deviceCalloc(this->maxIters);
        d_sn = DeviceMemory<ITYPE, double>::deviceCalloc(this->maxIters);
        d_gk = DeviceMemory<ITYPE, double>::deviceCalloc((this->maxIters + 1));
        d_yk = DeviceMemory<ITYPE, double>::deviceCalloc((this->maxIters + 1));
        d_H_k = DeviceMemory<ITYPE, double>::deviceCalloc(HK_size);
        d_wk = DeviceMemory<ITYPE, RTYPE>::deviceCalloc(this->arrSize);
        d_qk = DeviceMemory<ITYPE, RTYPE>::deviceCalloc(this->arrSize);
        d_Q_k = DeviceMemory<ITYPE, RTYPE>::deviceCalloc(QK_size);
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
    free(this->hk);
    free(this->norm);
    free(this->cs);
    free(this->sn);
    free(this->gk);
    free(this->yk);
    free(this->H_k);
    free(this->wk);
    free(this->qk);
    free(this->Q_k);
    free(this->sendbuf);
    free(this->recvbuf);
    free(this->dotTmp1);
    free(this->dotTmp2);
    #ifdef USE_GPU
        // Free device memory
        DeviceMemory<ITYPE, double>::deviceFree(this->d_beta);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_hk);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_norm);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_cs);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_sn);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_gk);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_yk);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_H_k);
        DeviceMemory<ITYPE, RTYPE>::deviceFree(this->d_wk);
        DeviceMemory<ITYPE, RTYPE>::deviceFree(this->d_qk);
        DeviceMemory<ITYPE, RTYPE>::deviceFree(this->d_Q_k);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_sendbuf);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_recvbuf);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_dotTmp1);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_dotTmp2);
    #endif
    POP_RANGE();
}

template class GMRES<uint32_t, float>;
template class GMRES<uint64_t, float>;
template class GMRES<uint32_t, double>;
template class GMRES<uint64_t, double>;
#ifdef USE_GPU
    template class GMRES<uint32_t, DeviceUtils::bf16>;
    template class GMRES<uint64_t, DeviceUtils::bf16>;
#endif
