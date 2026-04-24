#include "ArnoldiIter.hpp"

template <typename ITYPE, typename RTYPE>
ArnoldiIter<ITYPE, RTYPE>::ArnoldiIter() {
    this->nRows = 0;
    this->maxIters = 0;
    this->wOld = nullptr;
    this->wNew = nullptr;
    this->UpHess = nullptr;
    this->Qkrylov = nullptr;
    this->aux = nullptr;
    this->tmpDot = nullptr;
    this->mpiTmp = nullptr;
    this->flag_planned = false;
    this->flag_setup = false;
}

// ArnoldiIter constructor without MPI
template <typename ITYPE, typename RTYPE>
ArnoldiIter<ITYPE, RTYPE>::ArnoldiIter(ITYPE nRows_in, ITYPE maxIters_in) {
    plan(nRows_in, maxIters_in);
}

// ArnoldiIter constructor with MPI
template <typename ITYPE, typename RTYPE>
ArnoldiIter<ITYPE, RTYPE>::ArnoldiIter(MPI_Comm& c_comm, ITYPE nRows_in, ITYPE maxIters_in) {
    // set up MPI communication utilities
    this->ArnoldiIter_comm.setup(c_comm);
    plan(nRows_in, maxIters_in);
}

// Desctrucor
template <typename ITYPE, typename RTYPE>
ArnoldiIter<ITYPE, RTYPE>::~ArnoldiIter() {
    if (this->wOld != nullptr) free(this->wOld);
    if (this->wNew != nullptr) free(this->wNew);
    if (this->UpHess != nullptr) free(this->UpHess);
    if (this->Qkrylov != nullptr) free(this->Qkrylov);
    if (this->aux != nullptr) free(this->aux);
    if (this->tmpDot != nullptr) free(this->tmpDot);
    if (this->mpiTmp != nullptr) free(this->mpiTmp);
}

// Plan -allocates memory
template <typename ITYPE, typename RTYPE>
void ArnoldiIter<ITYPE, RTYPE>::plan(ITYPE nRows_in, ITYPE maxIters_in) {
    this->nRows = nRows_in;
    this->maxIters = maxIters_in;

    // Allocate internal matrices
    this->wOld = (RTYPE *)calloc(this->nRows, sizeof(RTYPE));
    this->wNew = (RTYPE *)calloc(this->nRows, sizeof(RTYPE));
    //this->UpHess = (RTYPE*)calloc( (maxIters_in+1) * maxIters_in, sizeof(RTYPE) ); // Always on host, m+1 rows, m cols
    this->UpHess = (double*)calloc( (maxIters_in+1) * maxIters_in, sizeof(double) ); // Always on host, m+1 rows, m cols
    this->Qkrylov = (RTYPE*)calloc( nRows_in * (maxIters_in+1), sizeof(RTYPE) );
    this->aux = (RTYPE*)calloc( this->auxSize, sizeof(RTYPE) );
    this->tmpDot = (double*)calloc( this->auxSize, sizeof(double) );
    this->mpiTmp = (double*)calloc( this->auxSize, sizeof(double) );
    this->flag_planned = true;
}

// Setup - initializes first basis vector q[0] = v0/||v0|| and stores it as the first column in the matrix Q
template <typename ITYPE, typename RTYPE>
void ArnoldiIter<ITYPE, RTYPE>::setup(RTYPE* v0) {

    // Compute partial norm of v0
    double local = 0.0;
    for (ITYPE i = 0; i < this->nRows; i++)
        local += (double)v0[i] * (double)v0[i];

    // Reduce to get global norm
    MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_DOUBLE, MPI_SUM, this->ArnoldiIter_comm.getLibComm());
    double beta = std::sqrt(local);

    // q[0] = v0 / beta
    // stored in column 0 of Qkrylov
    // Qkrylov[i*(maxIters+1) + 0] = v0[i] / beta
    for (ITYPE i = 0; i < this->nRows; i++){

        //this->Qkrylov[i*(this->maxIters+1) + 0] = (RTYPE)(v0[i]/beta);
        this->Qkrylov[i*(this->maxIters+1) + 0] = (RTYPE)((double)v0[i] / beta);   
    }
    this->flag_setup = true;
}

// GramSchmidt - version with MPI
template <typename ITYPE, typename RTYPE>
void ArnoldiIter<ITYPE, RTYPE>::GramSchmidt(ITYPE kIter) {
    // For each previous basis vector q[j], j=0,...,kIter-1
    for (ITYPE j = 0; j < kIter; j++) {

        // Extract q[j] from column j of Qkrylov into wold
        TensorUtils<ITYPE, RTYPE>::extract_column(this->nRows, this->maxIters+1, this->Qkrylov, j, this->wOld);

        // h[j][kIter] = dot(wNew, q[j])  -local partial dot product)
        double local_dot = 0.0;
        for (ITYPE i = 0; i < this->nRows; i++)
            local_dot += (double)this->wNew[i] * (double)this->wOld[i];

        // Reduce to get global dot product
        MPI_Allreduce(MPI_IN_PLACE, &local_dot, 1, MPI_DOUBLE, MPI_SUM, this->ArnoldiIter_comm.getLibComm());

        // Store in upper Hessenberg matrix - column kIter, row j
        //this->UpHess[j * this->maxIters + (kIter-1)] = (RTYPE)local_dot;
        this->UpHess[j * this->maxIters + (kIter-1)] = local_dot;
        // wNew = wNew - h[j][kIter] * q[j]
        for (ITYPE i = 0; i < this->nRows; i++)
            //this->wNew[i] -= (RTYPE)(local_dot * this->wOld[i]);
            this->wNew[i] = (RTYPE)((double)this->wNew[i] - local_dot * (double)this->wOld[i]);
    }
}

// arnoldiStep - runs one Arnoldi step at iteration j
// called by GMRES inner loop
template <typename ITYPE, typename RTYPE>
double ArnoldiIter<ITYPE, RTYPE>::arnoldiStep(ITYPE j, const typename EntryPoint<ITYPE,RTYPE>::MatVecOp& matvec) {
    // 1. extract q[j] from column j of Qkrylov into wOld
    TensorUtils<ITYPE, RTYPE>::extract_column(this->nRows, this->maxIters+1, this->Qkrylov, j, this->wOld);

    // 2. Compute wNew = A*q[j]
    matvec(this->wOld, this->wNew);

    // 3. Gram-Schmidt orthogonalization
    GramSchmidt(j + 1);

    // 4. Compute the norm of the residual
    double local = 0.0;
    for (ITYPE i = 0; i < this->nRows; i++)
        local += (double)this->wNew[i] * (double)this->wNew[i];

    // Reduce to get global norm
    MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_DOUBLE, MPI_SUM, this->ArnoldiIter_comm.getLibComm());

    double h_new = std::sqrt(local);

    // Store h[j+1][j] in upper Hessenberg matrix
    //this->UpHess[(j+1)*this->maxIters + j] = (RTYPE)h_new;
    this->UpHess[(j+1)*this->maxIters + j] = h_new;
    // 5. q[j+1] = wNew / h[j+1][j]
    // stored in column j+1 of Qkrylov
    for (ITYPE i = 0; i < this->nRows; i++)
        //this->Qkrylov[i*(this->maxIters+1) + (j+1)] = (RTYPE)(this->wNew[i]/h_new);
        this->Qkrylov[i*(this->maxIters+1) + (j+1)] = (RTYPE)((double)this->wNew[i] / h_new);
    // Return h[j+1][j] - norm of new basis vector, needed by GMRES for given rotations
    return h_new;
}

template class ArnoldiIter<uint32_t, float>;
template class ArnoldiIter<uint64_t, float>;
template class ArnoldiIter<uint32_t, double>;
template class ArnoldiIter<uint64_t, double>;
#ifdef USE_GPU
    template class ArnoldiIter<uint32_t, __nv_bfloat16>;
    template class ArnoldiIter<uint64_t, __nv_bfloat16>;
#endif