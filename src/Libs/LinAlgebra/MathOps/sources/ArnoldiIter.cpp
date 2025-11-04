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

template <typename ITYPE, typename RTYPE>
ArnoldiIter<ITYPE, RTYPE>::ArnoldiIter(ITYPE nRows_in, ITYPE maxIters_in) {
    plan(nRows_in, maxIters_in);
}

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

template <typename ITYPE, typename RTYPE>
void ArnoldiIter<ITYPE, RTYPE>::plan(ITYPE nRows_in, ITYPE maxIters_in) {
    this->nRows = nRows_in;
    this->maxIters = maxIters_in;

    // Allocate internal matrices
    this->wOld = (RTYPE *)calloc(this->nRows, sizeof(RTYPE));
    this->wNew = (RTYPE *)calloc(this->nRows, sizeof(RTYPE));
    this->UpHess = (RTYPE*)calloc( (maxIters+1) * maxIters, sizeof(RTYPE) ); // Always on host, m+1 rows, m cols
    this->Qkrylov = (RTYPE*)calloc( nRows * (maxIters+1), sizeof(RTYPE) );
    this->aux = (RTYPE*)calloc( this->auxSize, sizeof(RTYPE) );
    this->tmpDot = (double*)calloc( this->auxSize, sizeof(double) );
    this->mpiTmp = (double*)calloc( this->auxSize, sizeof(double) );
    this->flag_planned = true;
}

template <typename ITYPE, typename RTYPE>
void ArnoldiIter<ITYPE, RTYPE>::GramSchmidt(ITYPE kIter) {
    for (ITYPE j = 0; j < kIter-1; ++j) {
    }
}

template class ArnoldiIter<uint32_t, float>;
template class ArnoldiIter<uint64_t, float>;
template class ArnoldiIter<uint32_t, double>;
template class ArnoldiIter<uint64_t, double>;
#ifdef USE_GPU
    template class ArnoldiIter<uint32_t, __nv_bfloat16>;
    template class ArnoldiIter<uint64_t, __nv_bfloat16>;
#endif