#include "ArnoldiIter.hpp"

template <typename ITYPE, typename RTYPE>
ArnoldiIter<ITYPE, RTYPE>::ArnoldiIter() {
    this->nRows = 0;
    this->maxIters = 0;
    this->q0 = nullptr;
    this->UpHess = nullptr;
    this->Qkrylov = nullptr;
    this->flag_planned = false;
    this->flag_setup = false;
}

template <typename ITYPE, typename RTYPE>
ArnoldiIter<ITYPE, RTYPE>::ArnoldiIter(ITYPE nRows_in, ITYPE maxIters_in) {
    plan(nRows_in, maxIters_in);
}

template <typename ITYPE, typename RTYPE>
ArnoldiIter<ITYPE, RTYPE>::~ArnoldiIter() {
    if (this->UpHess != nullptr) {
        free(this->UpHess);
    }
    #ifdef USE_GPU
        if (this->Qkrylov != nullptr) {
            CUDA_CHECK( cudaFree( this->Qkrylov ) );
        }
    #else
        if (this->Qkrylov != nullptr) {
            free(this->Qkrylov);
        }
    #endif
}

template <typename ITYPE, typename RTYPE>
void ArnoldiIter<ITYPE, RTYPE>::plan(ITYPE nRows_in, ITYPE maxIters_in) {
    this->nRows = nRows_in;
    this->maxIters = maxIters_in;

    // Allocate internam matrices
    this->UpHess = (RTYPE*)calloc( (maxIters+1) * maxIters, sizeof(RTYPE) ); // Always on host, m+1 rows, m cols
    #ifdef USE_GPU
        CUDA_CHECK( cudaMalloc( (void**)&(this->Qkrylov), nRows * (maxIters+1) * sizeof(RTYPE) ) );
        CUDA_CHECK( cudaMemset( this->Qkrylov, 0, nRows * (maxIters+1) * sizeof(RTYPE) ) );
    #else
        this->Qkrylov = (RTYPE*)calloc( nRows * (maxIters+1), sizeof(RTYPE) );
    #endif
    this->flag_planned = true;
}

template <typename ITYPE, typename RTYPE>
void ArnoldiIter<ITYPE, RTYPE>::GramSchmidt(ITYPE kIter) {
}

template class ArnoldiIter<uint32_t, float>;
template class ArnoldiIter<uint64_t, float>;
template class ArnoldiIter<uint32_t, double>;
template class ArnoldiIter<uint64_t, double>;
#ifdef USE_GPU
    template class ArnoldiIter<uint32_t, __nv_bfloat16>;
    template class ArnoldiIter<uint64_t, __nv_bfloat16>;
#endif