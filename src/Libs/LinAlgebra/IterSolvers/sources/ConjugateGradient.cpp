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

    // 0. define launch grid for kernels and for size=1 auxiliaries
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
    CUDA_CHECK(cudaMemset(this->d_r0, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_p0, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_rk, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_Ax, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_res0, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_resk, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_aux, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_alpha, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_beta, 0, 1 * sizeof(RTYPE)));
    POP_RANGE

    //1. r0 = b - A*x0
    PUSH_RANGE("cgSolver: Ax0", 2)
    matvec(this->d_x0, this->d_Ax); // Ax = A*x0
    POP_RANGE

    PUSH_RANGE("cgSolver: r0, p0", 2)
    launchKernel(copy_array<ITYPE,RTYPE>, grid, block, this->d_b, this->d_r0, this->arrSize); // r0 = b
    RTYPE fact = static_cast<RTYPE>(-1);
    launchKernel(axpy<ITYPE,RTYPE>, grid, block, fact, this->d_Ax, this->d_r0, this->arrSize); // r0 = r0 - Ax
    launchKernel(copy_array<ITYPE,RTYPE>, grid, block, this->d_r0, this->d_p0, this->arrSize); // p0 = r0
    launchKernel(copy_array<ITYPE,RTYPE>, grid, block, this->d_r0, this->d_rk, this->arrSize); // rk = r0
    POP_RANGE

    //2. res0 = ||r0||
    PUSH_RANGE("cgSolver: res0", 2)
    launchKernel(dot_product<ITYPE,RTYPE>, grid, block, this->d_r0, this->d_r0, this->d_res0, this->arrSize); // res0 = r0' * r0
    PUSH_RANGE("cgSolver: AllReduce", 3)
    // TODO: call an Allreduce routine here
    POP_RANGE
    launchKernel(copy_array<ITYPE,RTYPE>, auxGrid, auxBlock, this->d_res0, this->d_resk, auxSize); // resk = res0 = r0' * r0
    launchKernel(array_sqrt<ITYPE,RTYPE>, auxGrid, auxBlock, this->d_res0, auxSize); // res0 = sqrt(res0)
    CUDA_CHECK(cudaMemcpy(this->res0, this->d_res0, sizeof(RTYPE), cudaMemcpyDeviceToHost)); // Copy res0 to host
    CUDA_CHECK(cudaMemcpy(this->resk, this->d_resk, sizeof(RTYPE), cudaMemcpyDeviceToHost)); // Copy resk to host
    POP_RANGE

    // Debug: print the initial residual
    printf("--|ConjGrad: Initial residual (GPU): %e\n", static_cast<double>(this->res0[0]));

    // Iterations
    PUSH_RANGE("cgSolver: iterations", 2)
    this->iter = 0;
    while (this->iter < this->maxIters) {
        PUSH_RANGE("cgSolver: iteration", 3)
        this->iter++;

        //3. alpha = (rk' * rk) / (pk' * A * pk)

        PUSH_RANGE("cgSolver: Apk", 4)
        matvec(this->d_p0, this->d_Ax); // Ax = A*p0
        POP_RANGE

        PUSH_RANGE("cgSolver: pkApk", 4)
        launchKernel(dot_product<ITYPE,RTYPE>, grid, block, this->d_p0, this->d_Ax, this->d_alpha, this->arrSize); // alpha = p0' * Ax
        PUSH_RANGE("cgSolver: AllReduce", 5)
        // TODO: call an Allreduce routine here
        POP_RANGE
        POP_RANGE

        PUSH_RANGE("cgSolver: invert aux", 4)
        launchKernel(array_invert<ITYPE,RTYPE>, auxGrid, auxBlock, this->d_alpha, auxSize); // alphaInv = 1/alpha
        POP_RANGE

        PUSH_RANGE("cgSolver: alpha", 4)
        launchKernel(pointwise_multiply<ITYPE,RTYPE>, auxGrid, auxBlock, this->d_resk,this->d_alpha, auxSize); // alpha = resk * alphaInv
        CUDA_CHECK(cudaMemcpy(this->alpha, this->d_alpha, sizeof(RTYPE), cudaMemcpyDeviceToHost));
        POP_RANGE

        // Debug: print alpha
#ifdef DEBUG
        printf("Iter %d, alpha: %e\n", this->iter, static_cast<double>(this->alpha[0]));
#endif // DEBUG

        //4. xk+1 = xk + alpha * pk
        PUSH_RANGE("cgSolver: xk+1", 4)
        launchKernel(axpy<ITYPE,RTYPE>, grid, block, this->alpha[0], this->d_p0, this->d_x_sol, this->arrSize); // x_sol = x_sol + alpha * p0
        POP_RANGE

        //5. rk+1 = rk - alpha * A * pk
        PUSH_RANGE("cgSolver: rk+1", 4)
        RTYPE negAlpha = -this->alpha[0];
        launchKernel(axpy<ITYPE,RTYPE>, grid, block, negAlpha, this->d_Ax, this->d_rk, this->arrSize); // rk = rk - alpha * Ax
        POP_RANGE

        //6. resk+1 = ||rk+1||
        PUSH_RANGE("cgSolver: resk+1", 4)
        launchKernel(dot_product<ITYPE,RTYPE>, grid, block, this->d_rk, this->d_rk, this->d_aux, this->arrSize); // resk+1 = rk+1' * rk+1
        PUSH_RANGE("cgSolver: AllReduce", 5)
        // TODO: call an Allreduce routine here
        POP_RANGE
        CUDA_CHECK(cudaMemcpy(this->aux, this->d_aux, sizeof(RTYPE), cudaMemcpyDeviceToHost)); // Copy aux to host
        POP_RANGE

        //7. Check convergence: sqrt(rk+1' * rk+1) < tol * sqrt(r0' * r0) => sqrt(aux) < tol * res0
        double reskSQ = static_cast<double>(this->aux[0]);
        reskSQ = std::sqrt(reskSQ);
        if (reskSQ < this->tol * static_cast<double>(this->res0[0])) {
            // Converged
            printf("--| ConjGrad: Converged at iteration %d with residual %e\n", this->iter, reskSQ);
            POP_RANGE
            break;
        }

        printf("--| ConjGrad: Iteration %d, residual %e\n", this->iter, reskSQ);

        //8. beta = (rk+1' * rk+1) / (rk' * rk)
        PUSH_RANGE("cgSolver: beta", 4)
        this->beta[0] = this->aux[0] / this->resk[0];
        POP_RANGE

        // Debug: print beta
#ifdef DEBUG
        printf("Iter %d, beta: %e\n", this->iter, static_cast<double>(this->beta[0]));
#endif

        //9. pk+1 = rk+1 + beta * pk
        PUSH_RANGE("cgSolver: pk+1", 4)
        launchKernel(axpy<ITYPE,RTYPE>, grid, block, this->beta[0], this->d_p0, this->d_rk, this->arrSize); // pk+1 = rk+1 + beta * pk
        POP_RANGE

        //10. Update resk for next iteration
        PUSH_RANGE("cgSolver: update resk", 4)
        this->resk[0] = this->aux[0];
        CUDA_CHECK(cudaMemcpy(this->d_resk, this->d_aux, sizeof(RTYPE), cudaMemcpyDeviceToDevice)); // resk = aux
        POP_RANGE

        POP_RANGE
    }
    POP_RANGE
#else
    // Initial step

    //1. r0 = b - A*x0
    PUSH_RANGE("cgSolver: Ax0", 2)
    matvec(this->x0, this->Ax); // Ax = A*x0
    POP_RANGE
    PUSH_RANGE("cgSolver: r0, p0", 2)
    TensorUtils<ITYPE, RTYPE>::copy_array(this->arrSize, this->b, this->r0);                    // r0 = b
    TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, static_cast<RTYPE>(-1), this->Ax, this->r0); // r0 = b - Ax
    TensorUtils<ITYPE, RTYPE>::copy_array(this->arrSize, this->r0, this->p0);                   // p0 = r0
    POP_RANGE

    //2. res0 = ||r0||
    PUSH_RANGE("cgSolver: res0", 2)
    TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->r0, this->r0, this->res0); // res0 = r0' * r0
    this->res0[0] = std::sqrt(this->res0[0]);                                              // res0 = sqrt(res0)
    POP_RANGE

    // Debug: print the initial residual
    printf("Initial residual (CPU): %e\n", static_cast<double>(this->res0[0]));
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