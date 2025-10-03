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
    CUDA_CHECK(cudaMemset(this->d_p0, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc((void **)&this->d_alpha, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_alpha, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc((void **)&this->d_beta, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_beta, 0, 1 * sizeof(RTYPE)));
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

//-------------------------//
// Solver implementations  //
//-------------------------//

template <typename ITYPE, typename RTYPE>
void ConjugateGradient<ITYPE, RTYPE>::cgSolver(const MatVecOp& matvec) {
    PUSH_RANGE("ConjugateGradient::cgSolver", 1)
#ifdef USE_GPU
    // Initial step

    // -1. define launch grid for kernels and for size=1 auxiliaries
    dim3 block(TILE_SIZE, 1, 1);
    ITYPE numBlocks = (this->arrSize + TILE_SIZE - 1) / TILE_SIZE;
    numBlocks = std::min(numBlocks, (ITYPE)MAX_BLOCKS);
    dim3 grid(numBlocks, 1, 1);

    ITYPE auxSize = 1;
    dim3 auxBlock(1, 1, 1);
    dim3 auxGrid(1, 1, 1);

    //0. initialize solution to x0 and all else to zero
    PUSH_RANGE("cgSolver: x_sol = x0", 2)
    launchKernel(copy_array<ITYPE,RTYPE>, grid, block, this->d_x0, this->d_x_sol, this->arrSize); // x_sol = x0
    POP_RANGE

    PUSH_RANGE("cgSolver: zero arrays", 2)
    memset(this->res0, 0, 1 * sizeof(RTYPE));
    memset(this->resk, 0, 1 * sizeof(RTYPE));
    memset(this->aux, 0, 1 * sizeof(RTYPE));
    memset(this->alpha, 0, 1 * sizeof(RTYPE));
    memset(this->beta, 0, 1 * sizeof(RTYPE));
    memset(this->tmpDot, 0, 1 * sizeof(float));
    CUDA_CHECK(cudaMemset(this->d_r0, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_p0, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_rk, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_Ax, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_res0, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_resk, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_aux, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_alpha, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_beta, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(float)));
    POP_RANGE

    // Initial step
    PUSH_RANGE("cgSolver: initial step", 2)

    //1. compute initial residual r0 = b - A*x0

    PUSH_RANGE("cgSolver: matvec", 3)
    matvec(this->d_x0, this->d_Ax); // Ax = A*x0
    POP_RANGE

    PUSH_RANGE("cgSolver: r0 = b - Ax", 3)
    launchKernel(copy_array<ITYPE,RTYPE>, grid, block, this->d_b, this->d_r0, this->arrSize); // r0 = b
    RTYPE negOne = static_cast<RTYPE>(-1);
    launchKernel(axpy<ITYPE,RTYPE>, grid, block, negOne, this->d_Ax, this->d_r0, this->arrSize); // r0 = r0 - Ax
    POP_RANGE

    PUSH_RANGE("cgSolver: residual0", 3)
    CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(double)));
    launchKernel(dot_product<ITYPE,RTYPE>, grid, block, this->d_r0, this->d_r0, this->d_tmpDot, this->arrSize); // tmpDot = r0 . r0
    CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_tmpDot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
    printf("Initial residual norm: %e\n", sqrt(this->tmpDot[0]));
    POP_RANGE

    POP_RANGE

#else
#endif
    POP_RANGE
}

template class ConjugateGradient<uint32_t, float>;
template class ConjugateGradient<uint64_t, float>;
template class ConjugateGradient<uint32_t, double>;
template class ConjugateGradient<uint64_t, double>;
#ifdef USE_GPU
template class ConjugateGradient<uint32_t, __nv_bfloat16>;
template class ConjugateGradient<uint64_t, __nv_bfloat16>;
#endif