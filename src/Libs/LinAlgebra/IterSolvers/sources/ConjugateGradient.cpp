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


// Non-preconditioned Conjugate Gradient solver
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
    memset(this->tmpDot, 0, 1 * sizeof(double));
    CUDA_CHECK(cudaMemset(this->d_r0, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_p0, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_rk, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_Ax, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_res0, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_resk, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_aux, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_alpha, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_beta, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(double)));
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

    //2. compute initial residual norm |r0| and initialize rk, rk.rk, p0
    PUSH_RANGE("cgSolver: residual0", 3)
    launchKernel(copy_array<ITYPE,RTYPE>, grid, block, this->d_r0, this->d_rk, this->arrSize); // rk = r0
    launchKernel(copy_array<ITYPE,RTYPE>, grid, block, this->d_r0, this->d_p0, this->arrSize); // p0 = r0
    CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(double)));
    launchKernel(dot_product<ITYPE,RTYPE>, grid, block, this->d_r0, this->d_r0, this->d_tmpDot, this->arrSize); // tmpDot = r0 . r0
    CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_tmpDot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
    // TODO: finish this part
    this->resk[0] = static_cast<RTYPE>(this->tmpDot[0]); // resk = rk.rk at k = 0
    this->res0[0] = static_cast<RTYPE>(std::sqrt(this->tmpDot[0])); // res0 = |r0|
    POP_RANGE

    POP_RANGE // 2: initial step

    // Iterate
    this->iter = 0;
    PUSH_RANGE("cgSolver: iterations", 2)
    while (this->iter < this->maxIters) {
        PUSH_RANGE("cgSolver: iteration", 3)
        this->iter++; // increment iteration counter then compute

        // 3. Compute alpha
        //   -- rk.rk should already be computed, so only need pk.Apk
        PUSH_RANGE("cgSolver: alpha", 4)
        matvec(this->d_p0, this->d_Ax); // Ax = A*p0
        CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(double)));
        launchKernel(dot_product<ITYPE,RTYPE>, grid, block, this->d_p0, this->d_Ax, this->d_tmpDot, this->arrSize); // tmpDot = p0 . Ax
        CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_tmpDot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        this->alpha[0] = this->resk[0] / static_cast<RTYPE>(this->tmpDot[0]);
        POP_RANGE // 4: alpha

        //4. Update solution xk = xk-1 + alpha*pk-1 and residual rk = rk-1 - alpha*Apk-1
        PUSH_RANGE("cgSolver: update step", 4)
        launchKernel(axpy<ITYPE,RTYPE>, grid, block, this->alpha[0], this->d_p0, this->d_x_sol, this->arrSize); // x_sol = x_sol + alpha*p0
        RTYPE negAlpha = -this->alpha[0];
        launchKernel(axpy<ITYPE,RTYPE>, grid, block, negAlpha, this->d_Ax, this->d_rk, this->arrSize); // rk = rk - alpha*Ax
        POP_RANGE // 4: update step

        //5. Compute new rk.rk
        PUSH_RANGE("cgSolver: residualk", 4)
        CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(double)));
        launchKernel(dot_product<ITYPE,RTYPE>, grid, block, this->d_rk, this->d_rk, this->d_tmpDot, this->arrSize); // tmpDot = rk . rk
        CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_tmpDot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        this->aux[0] = static_cast<RTYPE>(this->tmpDot[0]); // aux = rk.rk
        POP_RANGE // 4: residualk

        // Check convergence
        if ( std::sqrt(this->tmpDot[0]) < this->tol * static_cast<double>(this->res0[0]) ) {
            // Converged
            printf("--| cgSolver: converged at iteration %u with residual %e\n", this->iter, static_cast<double>(std::sqrt(this->tmpDot[0])));
            POP_RANGE // 3: iteration
            break;
        }

        //6. Compute beta and update search direction pk = rk + beta*pk-1
        PUSH_RANGE("cgSolver: beta", 4)
        this->beta[0] = this->aux[0] / this->resk[0]; // beta = (rk.rk)new / (rk.rk)old
        launchKernel(scale<ITYPE,RTYPE>, grid, block, this->beta[0], this->d_p0, this->arrSize); // p0 = beta*p0
        const RTYPE one = static_cast<RTYPE>(1);
        launchKernel(axpy<ITYPE,RTYPE>, grid, block, one, this->d_rk, this->d_p0, this->arrSize); // p0 = rk + beta*p0
        this->resk[0] = this->aux[0]; // resk = (rk.rk)new
        POP_RANGE // 4: beta

        POP_RANGE // 3: iteration
    }
    POP_RANGE // 2: iterations

#else
    //0. initialize solution to x0 and all else to zero
    TensorUtils<ITYPE, RTYPE>::copy_array(this->arrSize, this->x0, this->x_sol); // x_sol = x0
    memset(this->res0, 0, 1 * sizeof(RTYPE));
    memset(this->resk, 0, 1 * sizeof(RTYPE));
    memset(this->aux, 0, 1 * sizeof(RTYPE));
    memset(this->alpha, 0, 1 * sizeof(RTYPE));
    memset(this->beta, 0, 1 * sizeof(RTYPE));
    memset(this->tmpDot, 0, 1 * sizeof(double));
    memset(this->r0, 0, this->arrSize * sizeof(RTYPE));
    memset(this->p0, 0, this->arrSize * sizeof(RTYPE));
    memset(this->rk, 0, this->arrSize * sizeof(RTYPE));
    memset(this->Ax, 0, this->arrSize * sizeof(RTYPE));

    // Initial step

    //1. compute initial residual r0 = b - A*x0
    matvec(this->x0, this->Ax); // Ax = A*x0
    TensorUtils<ITYPE, RTYPE>::copy_array(this->arrSize, this->b, this->r0); // r0 = b
    TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, static_cast<RTYPE>(-1), this->Ax, this->r0); // r0 = b - Ax

    // 2. compute initial residual norm |r0| and initialize rk, rk.rk, p0
    TensorUtils<ITYPE, RTYPE>::copy_array(this->arrSize, this->r0, this->rk); // rk = r0
    TensorUtils<ITYPE, RTYPE>::copy_array(this->arrSize, this->r0, this->p0); // p0 = r0
    TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->r0, this->r0, this->res0); // res0 = r0 . r0
    this->resk[0] = this->res0[0]; // resk = res0
    this->res0[0] = std::sqrt(this->res0[0]); // res0 = |r0|

    // Iterate
    this->iter = 0;
    while (this->iter < this->maxIters) {
        this->iter++; // increment iteration counter then compute

        //3. Compute alpha
        //   -- rk.rk should already be computed, so only need pk.Apk
        matvec(this->p0, this->Ax); // Ax = A*p0
        TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->p0, this->Ax, this->alpha); // alpha = p0 . Ax
        this->alpha[0] = this->resk[0] / this->alpha[0];
        printf("Iter %u, alpha = %e\n", this->iter, static_cast<double>(this->alpha[0]));

        //4. Update solution xk = xk-1 + alpha*pk-1 and residual rk = rk-1 - alpha*Apk-1
        TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, this->alpha[0], this->p0, this->x_sol); // x_sol = x_sol + alpha*p0
        TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, -this->alpha[0], this->Ax, this->rk); // rk = rk - alpha*Ax

        //5. Compute new rk.rk
        TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->rk, this->rk, this->aux); // aux = rk . rk
        this->tmpDot[0] = static_cast<double>(this->aux[0]);

        // Check convergence
        if ( std::sqrt(this->tmpDot[0]) < this->tol * static_cast<double>(this->res0[0]) ) {
            printf("--| cgSolver: converged at iteration %u with residual %e\n", this->iter, static_cast<double>(std::sqrt(this->tmpDot[0])));
            break;
        }

        //6. Compute beta and update search direction pk = rk + beta*pk-1
        this->beta[0] = this->aux[0] / this->resk[0]; // beta = (rk.rk)new / (rk.rk)old
        TensorUtils<ITYPE, RTYPE>::scale(this->arrSize, this->beta[0], this->p0); // p0 = beta*p0
        TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, static_cast<RTYPE>(1), this->rk, this->p0); // p0 = rk + beta*p0
        this->resk[0] = this->aux[0]; // resk = (rk.rk)new
    }

#endif
    POP_RANGE // 1: cgSolver
}

template <typename ITYPE, typename RTYPE>
void ConjugateGradient<ITYPE, RTYPE>::fpcgSolver(const MatVecOp &matvec, const PrecondOp &precond)
{
    PUSH_RANGE("ConjugateGradient::fpcgSolver", 1)
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

    // 0. initialize solution to x0 and all else to zero
    PUSH_RANGE("cgSolver: x_sol = x0", 2)
    launchKernel(copy_array<ITYPE, RTYPE>, grid, block, this->d_x0, this->d_x_sol, this->arrSize); // x_sol = x0
    POP_RANGE

    PUSH_RANGE("cgSolver: zero arrays", 2)
    memset(this->res0, 0, 1 * sizeof(RTYPE));
    memset(this->resk, 0, 1 * sizeof(RTYPE));
    memset(this->aux, 0, 1 * sizeof(RTYPE));
    memset(this->alpha, 0, 1 * sizeof(RTYPE));
    memset(this->beta, 0, 1 * sizeof(RTYPE));
    memset(this->tmpDot, 0, 1 * sizeof(double));
    CUDA_CHECK(cudaMemset(this->d_r0, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_p0, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_rk, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_zk, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_Ax, 0, this->arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_res0, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_resk, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_aux, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_alpha, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_beta, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(double)));
    POP_RANGE

    // Initial step
    PUSH_RANGE("cgSolver: initial step", 2)

    // 1. compute initial residual r0 = b - A*x0
    PUSH_RANGE("cgSolver: matvec", 3)
    matvec(this->d_x0, this->d_Ax); // Ax = A*x0
    POP_RANGE

    PUSH_RANGE("cgSolver: r0 = b - Ax", 3)
    launchKernel(copy_array<ITYPE, RTYPE>, grid, block, this->d_b, this->d_r0, this->arrSize); // r0 = b
    RTYPE negOne = static_cast<RTYPE>(-1);
    launchKernel(axpy<ITYPE, RTYPE>, grid, block, negOne, this->d_Ax, this->d_r0, this->arrSize); // r0 = r0 - Ax
    POP_RANGE

    // 2. Precondition M z0 = r0
    PUSH_RANGE("cgSolver: precond", 3)
    precond(this->d_r0, this->d_zk); // zk = M^-1 * r0
    POP_RANGE

    // 3. compute initial residual norm |r0| and initialize rk, rk.rk, p0
    PUSH_RANGE("cgSolver: residual0", 3)
    launchKernel(copy_array<ITYPE, RTYPE>, grid, block, this->d_r0, this->d_rk, this->arrSize); // rk = r0
    launchKernel(copy_array<ITYPE, RTYPE>, grid, block, this->d_r0, this->d_p0, this->arrSize); // p0 = r0
    CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(double)));
    launchKernel(dot_product<ITYPE, RTYPE>, grid, block, this->d_r0, this->d_r0, this->d_tmpDot, this->arrSize); // tmpDot = r0 . r0
    CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_tmpDot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
    this->res0[0] = static_cast<RTYPE>(std::sqrt(this->tmpDot[0])); // res0 = |r0|
    CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(double)));
    launchKernel(dot_product<ITYPE, RTYPE>, grid, block, this->d_rk, this->d_zk, this->d_tmpDot, this->arrSize); // tmpDot = rk . zk
    CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_tmpDot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
    this->resk[0] = static_cast<RTYPE>(this->tmpDot[0]); // resk = rk.zk at k = 0
    POP_RANGE

    POP_RANGE // 2: initial step

    // Iterate
    this->iter = 0;
    PUSH_RANGE("cgSolver: iterations", 2)
    while (this->iter < this->maxIters)
    {
        PUSH_RANGE("cgSolver: iteration", 3)
        this->iter++; // increment iteration counter then compute

        // 4. Compute alpha
        //   -- rk.zk should already be computed, so only need pk.Apk
        PUSH_RANGE("cgSolver: alpha", 4)
        matvec(this->d_p0, this->d_Ax); // Ax = A*p0
        CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(double)));
        launchKernel(dot_product<ITYPE, RTYPE>, grid, block, this->d_p0, this->d_Ax, this->d_tmpDot, this->arrSize); // tmpDot = p0 . Ax
        CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_tmpDot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        this->alpha[0] = this->resk[0] / static_cast<RTYPE>(this->tmpDot[0]);
        POP_RANGE // 4: alpha

        // 5. Update solution xk = xk-1 + alpha*pk-1 and residual rk = rk-1 - alpha*Apk-1
        PUSH_RANGE("cgSolver: update step", 4)
        launchKernel(axpy<ITYPE, RTYPE>, grid, block, this->alpha[0], this->d_p0, this->d_x_sol, this->arrSize); // x_sol = x_sol + alpha*p0
        RTYPE negAlpha = -this->alpha[0];
        launchKernel(axpy<ITYPE, RTYPE>, grid, block, negAlpha, this->d_Ax, this->d_rk, this->arrSize); // rk = rk - alpha*Ax
        POP_RANGE                                                                                       // 4: update step

        // 6. Compute new rk.rk
        PUSH_RANGE("cgSolver: residualk", 4)
        CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(double)));
        launchKernel(dot_product<ITYPE, RTYPE>, grid, block, this->d_rk, this->d_rk, this->d_tmpDot, this->arrSize); // tmpDot = rk . rk
        CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_tmpDot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        this->aux[0] = static_cast<RTYPE>(this->tmpDot[0]); // aux = rk.rk
        POP_RANGE                                           // 4: residualk

        // Check convergence
        if (std::sqrt(this->tmpDot[0]) < this->tol * static_cast<double>(this->res0[0]))
        {
            // Converged
            printf("--| cgSolver: converged at iteration %u with residual %e\n", this->iter, static_cast<double>(std::sqrt(this->tmpDot[0])));
            POP_RANGE // 3: iteration
            break;
        }

        // 7. Compute aux = rk+1 . zk
        PUSH_RANGE("cgSolver: rk+1.zk ", 4)
        CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(double)));
        launchKernel(dot_product<ITYPE, RTYPE>, grid, block, this->d_rk, this->d_zk, this->d_tmpDot, this->arrSize); // tmpDot = rk . zk
        CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_tmpDot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        this->aux[0] = static_cast<RTYPE>(this->tmpDot[0]); // aux = rk+1 . zk
        POP_RANGE // 4: rk+1.zk

        // 8. Precondition M * zk+1 = rk+1
        PUSH_RANGE("cgSolver: precond", 4)
        precond(this->d_rk, this->d_zk); // zk = M^-1 * rk
        POP_RANGE // 4: precond

        // 9. beta = (rk+1.zk+1)
        PUSH_RANGE("cgSolver: beta prep", 4)
        CUDA_CHECK(cudaMemset(this->d_tmpDot, 0, 1 * sizeof(double)));
        launchKernel(dot_product<ITYPE, RTYPE>, grid, block, this->d_rk, this->d_zk, this->d_tmpDot, this->arrSize); // tmpDot = rk . zk
        CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_tmpDot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        this->beta[0] = static_cast<RTYPE>(this->tmpDot[0]); // beta = rk+1.zk+1
        POP_RANGE // 4: beta prep

        // 10. Compute beta = rk+1 . (zk+1 - zk) / rk . zk
        RTYPE tmp = this->beta[0];
        this->beta[0] = this->beta[0] - this->aux[0] / this->resk[0]; // beta = (rk+1.zk+1 - rk.zk) / rk.zk
        this->resk[0] = tmp;

        // 11. Update search direction pk = rk + beta*pk-1
        PUSH_RANGE("cgSolver: Update pk", 4)
        const RTYPE one = static_cast<RTYPE>(1);
        launchKernel(scale<ITYPE, RTYPE>, grid, block, this->beta[0], this->d_p0, this->arrSize); // p0 = beta*p0
        launchKernel(axpy<ITYPE, RTYPE>, grid, block, one, this->d_zk, this->d_p0, this->arrSize); // p0 = zk + beta*p0
        POP_RANGE // 4: Update pk

        POP_RANGE // 3: iteration
    }
    POP_RANGE // 2: iterations

#else
    // 0. initialize solution to x0 and all else to zero
    TensorUtils<ITYPE, RTYPE>::copy_array(this->arrSize, this->x0, this->x_sol); // x_sol = x0
    memset(this->res0, 0, 1 * sizeof(RTYPE));
    memset(this->resk, 0, 1 * sizeof(RTYPE));
    memset(this->aux, 0, 1 * sizeof(RTYPE));
    memset(this->alpha, 0, 1 * sizeof(RTYPE));
    memset(this->beta, 0, 1 * sizeof(RTYPE));
    memset(this->tmpDot, 0, 1 * sizeof(double));
    memset(this->r0, 0, this->arrSize * sizeof(RTYPE));
    memset(this->p0, 0, this->arrSize * sizeof(RTYPE));
    memset(this->rk, 0, this->arrSize * sizeof(RTYPE));
    memset(this->zk, 0, this->arrSize * sizeof(RTYPE));
    memset(this->Ax, 0, this->arrSize * sizeof(RTYPE));

    // Initial step

    // 1. compute initial residual r0 = b - A*x0
    matvec(this->x0, this->Ax);                                                                 // Ax = A*x0
    TensorUtils<ITYPE, RTYPE>::copy_array(this->arrSize, this->b, this->r0);                    // r0 = b
    TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, static_cast<RTYPE>(-1), this->Ax, this->r0); // r0 = b - Ax

    // 2. Precondition M * zk = rk
    precond(this->r0, this->zk); // zk = M^-1 * r0

    // 3. compute initial residual norm |r0| and initialize rk, rk.rk, p0
    TensorUtils<ITYPE, RTYPE>::copy_array(this->arrSize, this->r0, this->rk);              // rk = r0
    TensorUtils<ITYPE, RTYPE>::copy_array(this->arrSize, this->zk, this->p0);              // p0 = zk
    TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->rk, this->zk, this->resk);  // resk = rk . zk
    TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->r0, this->r0, this->res0); // res0 = r0 . r0
    this->res0[0] = std::sqrt(this->res0[0]);                                              // res0 = |r0|

    // Iterate
    this->iter = 0;
    while (this->iter < this->maxIters)
    {
        this->iter++; // increment iteration counter then compute

        // 4. Compute alpha
        //    -- rk.zk should already be computed, so only need pk.Apk
        matvec(this->p0, this->Ax);                                                             // Ax = A*p0
        TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->p0, this->Ax, this->alpha); // alpha = p0 . Ax
        this->alpha[0] = this->resk[0] / this->alpha[0];
        printf("Iter %u, alpha = %e\n", this->iter, static_cast<double>(this->alpha[0]));

        // 5. Update solution xk = xk-1 + alpha*pk-1 and residual rk = rk-1 - alpha*Apk-1
        TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, this->alpha[0], this->p0, this->x_sol); // x_sol = x_sol + alpha*p0
        TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, -this->alpha[0], this->Ax, this->rk);   // rk = rk - alpha*Ax

        // 6. Compute new rk.rk
        TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->rk, this->rk, this->aux); // aux = rk . rk
        this->tmpDot[0] = static_cast<double>(this->aux[0]);

        // Check convergence
        if (std::sqrt(this->tmpDot[0]) < this->tol * static_cast<double>(this->res0[0]))
        {
            printf("--| cgSolver: converged at iteration %u with residual %e\n", this->iter, static_cast<double>(std::sqrt(this->tmpDot[0])));
            break;
        }

        // 7. Compute rk+1 . zk
        TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->rk, this->zk, this->aux); // aux = rk . zk

        // 8. precondition M * zk+1 = rk+1
        precond(this->rk, this->zk); // zk = M^-1 * rk

        // 9. Compute beta = rk+1 . zk+1
        TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->rk, this->zk, this->beta); // beta = rk . zk

        // 10. Finish beta = rk+1 . (zk+1 - zk) / rk . zk
        RTYPE tmp = this->beta[0];
        this->beta[0] = ( this->beta[0] - this->aux[0] ) / this->resk[0];
        this->resk[0] = tmp; // resk = (rk.zk)new

        // 11. update search direction pk = zk + beta*pk-1
        const RTYPE one = static_cast<RTYPE>(1);
        TensorUtils<ITYPE, RTYPE>::scale(this->arrSize, this->beta[0], this->p0); // p0 = beta*p0
        TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, one, this->zk, this->p0);  // p0 = zk + beta*p0
    }

#endif
        POP_RANGE // 1: cgSolver
}

template class ConjugateGradient<uint32_t, float>;
template class ConjugateGradient<uint64_t, float>;
template class ConjugateGradient<uint32_t, double>;
template class ConjugateGradient<uint64_t, double>;
#ifdef USE_GPU
template class ConjugateGradient<uint32_t, __nv_bfloat16>;
template class ConjugateGradient<uint64_t, __nv_bfloat16>;
#endif