#include "ConjugateGradient.hpp"

template <typename ITYPE, typename RTYPE>
ConjugateGradient<ITYPE, RTYPE>::ConjugateGradient() : IterSolvers<ITYPE, RTYPE>() {
    std::cout << "--| IterSolvers: using PCG solver!" << std::endl;
    PUSH_RANGE("ConjugateGradient::Constructor(empty)", 0)
    this->p0 = nullptr;
    this->d_p0 = nullptr;
    this->alpha = nullptr;
    this->d_alpha = nullptr;
    this->beta = nullptr;
    this->d_beta = nullptr;
    POP_RANGE
}

template <typename ITYPE, typename RTYPE>
ConjugateGradient<ITYPE, RTYPE>::ConjugateGradient(ITYPE arrSize, ITYPE maxIters, double tol) : IterSolvers<ITYPE, RTYPE>(arrSize, maxIters, tol) {
    std::cout << "--| IterSolvers: using PCG solver!" << std::endl;
    PUSH_RANGE("ConjugateGradient::Constructor(param)", 0)
    // Allocate host memory using calloc (ensures init to 0)
    this->p0 = (RTYPE *)calloc(this->arrSize, sizeof(RTYPE));
    this->alpha = (RTYPE *)calloc(1, sizeof(RTYPE));
    this->beta = (RTYPE *)calloc(1, sizeof(RTYPE));
    POP_RANGE
#ifdef USE_GPU
    // Allocate device arrays
    PUSH_RANGE("ConjugateGradient::Constructor(param) -> device", 0)
    CUDA_CHECK(cudaMalloc((void **)&this->d_p0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc((void **)&this->d_alpha, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc((void **)&this->d_beta, 1 * sizeof(RTYPE)));
    POP_RANGE
#endif
}

template <typename ITYPE, typename RTYPE>
ConjugateGradient<ITYPE, RTYPE>::~ConjugateGradient() {
    std::cout << "--| IterSolvers: destroying PCG solver" << std::endl;
    PUSH_RANGE("ConjugateGradient::Destructor", 0)
    // Free host memory
    free(this->p0);
    free(this->alpha);
    free(this->beta);
    POP_RANGE
#ifdef USE_GPU
    // Free device memory
    PUSH_RANGE("ConjugateGradient::Destructor -> device", 0)
    CUDA_CHECK(cudaFree(this->d_p0));
    CUDA_CHECK(cudaFree(this->d_alpha));
    CUDA_CHECK(cudaFree(this->d_beta));
    POP_RANGE
#endif
}

template class ConjugateGradient<uint32_t, float>;
template class ConjugateGradient<uint64_t, float>;
template class ConjugateGradient<uint32_t, double>;
template class ConjugateGradient<uint64_t, double>;
#ifdef USE_GPU
template class ConjugateGradient<uint32_t, __nv_bfloat16>;
template class ConjugateGradient<uint64_t, __nv_bfloat16>;
#endif