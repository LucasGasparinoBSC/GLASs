#include "ConjugateGradient.hpp"

template <typename ITYPE, typename RTYPE>
ConjugateGradient<ITYPE, RTYPE>::ConjugateGradient() : IterSolvers<ITYPE, RTYPE>() {
    PUSH_RANGE("ConjugateGradient::Constructor(empty)", 4);
    this->p0 = nullptr;
    this->d_p0 = nullptr;
    this->alpha = nullptr;
    this->d_alpha = nullptr;
    this->beta = nullptr;
    this->d_beta = nullptr;
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
ConjugateGradient<ITYPE, RTYPE>::ConjugateGradient(MPI_Comm& c_comm, ITYPE arrSize, ITYPE maxIters, double tol) : IterSolvers<ITYPE, RTYPE>(c_comm, arrSize, maxIters, tol) {
    if (this->IterSolvers_comm.getWorldRank() == 0) std::cout << "--| IterSolvers: using PCG solver!" << std::endl;
    PUSH_RANGE("ConjugateGradient::Constructor(param+comm)", 4);
    // Allocate host memory using calloc (ensures init to 0)
    this->p0 = (RTYPE *)calloc(this->arrSize, sizeof(RTYPE));
    this->alpha = (double *)calloc(this->auxSize, sizeof(double));
    this->beta  = (double *)calloc(this->auxSize, sizeof(double));
    this->sendbuf = (double *)calloc(this->nargs, sizeof(double));
    this->recvbuf = (double *)calloc(this->nargs, sizeof(double));
    this->dotTmp1 = (double *)calloc(this->auxSize, sizeof(double));
    this->dotTmp2 = (double *)calloc(this->auxSize, sizeof(double));
    #ifdef USE_GPU
        // Allocate device arrays
        d_p0 = DeviceMemory<ITYPE, RTYPE>::deviceCalloc(this->arrSize);
        d_alpha = DeviceMemory<ITYPE, double>::deviceCalloc(this->auxSize);
        d_beta  = DeviceMemory<ITYPE, double>::deviceCalloc(this->auxSize);
        d_sendbuf = DeviceMemory<ITYPE, double>::deviceCalloc(this->nargs);
        d_recvbuf = DeviceMemory<ITYPE, double>::deviceCalloc(this->nargs);
        d_dotTmp1 = DeviceMemory<ITYPE, double>::deviceCalloc(this->auxSize);
        d_dotTmp2 = DeviceMemory<ITYPE, double>::deviceCalloc(this->auxSize);
    #endif

    // Start logfile
    this->logfile_name = "GLASs_cgSolver";
    if (this->IterSolvers_comm.getLibRank() == 0) {
        // Open the file for writing
        this->logfile.open(this->logfile_name + this->logfile_ext, std::ios::out);
        // Wrtie the header: "ITER ||rk|| cgTime(ms)"
        this->logfile << "ITER --- "
                      << " --- ||rk|| --- "
                      << " --- cgTime(ms)" << std::endl;
        this->logfile << "-------------------------------" << std::endl;
        this->logfile.flush();
    }
    POP_RANGE();
}

template <typename ITYPE, typename RTYPE>
ConjugateGradient<ITYPE, RTYPE>::~ConjugateGradient() {
    if (this->IterSolvers_comm.getWorldRank() == 0) std::cout << "--| IterSolvers: destroying PCG solver" << std::endl;

    // Close logfile
    if (this->IterSolvers_comm.getLibRank() == 0) this->logfile.close();

    PUSH_RANGE("ConjugateGradient::Destructor", 4);
    // Free host memory
    free(this->p0);
    free(this->alpha);
    free(this->beta);
    free(this->sendbuf);
    free(this->recvbuf);
    free(this->dotTmp1);
    free(this->dotTmp2);
    #ifdef USE_GPU
        // Free device memory
        DeviceMemory<ITYPE, RTYPE>::deviceFree(this->d_p0);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_alpha);
        DeviceMemory<ITYPE, double>::deviceFree(this->d_beta);
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
void ConjugateGradient<ITYPE, RTYPE>::cgSolver(const MatVecOp& matvec) {
    double out_sqrtRes = static_cast<double>(0);
    double cgTime = this->IterSolvers_comm.timeFunction([&]
    {
        PUSH_RANGE("ConjugateGradient::cgSolver", 4);
        // Basic scalars
        this->iter = 0;
        RTYPE zero = static_cast<RTYPE>(0);
        RTYPE negOne = static_cast<RTYPE>(-1);
        double zero_fp64 = static_cast<double>(0);
        #ifdef USE_GPU
            // Initial step
            PUSH_RANGE("cgSolver: initial step", 5);
            {
                // Init vars.
                PUSH_RANGE("cgSolver: init vars", 6);
                DeviceUtils::launchKernel(copy_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_x0, this->d_x_sol, this->arrSize); // x_sol = x0
                POP_RANGE(); // 6

                // Matvec
                PUSH_RANGE("cgSolver: matvec", 6);
                matvec(this->d_x_sol, this->d_Ax); // Ax = A*x0
                POP_RANGE(); // 6

                // r0 = b - Ax0
                PUSH_RANGE("cgSolver: r0 = b - Ax", 6);
                DeviceUtils::launchKernel(copy_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_b, this->d_rk, this->arrSize);    // rk = b
                DeviceUtils::launchKernel(axpy<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, negOne, this->d_Ax, this->d_rk, this->arrSize); // rk += (-1)*Ax0
                POP_RANGE(); // 6

                // Initial p0
                PUSH_RANGE("cgSolver: p0 = r0", 6);
                DeviceUtils::launchKernel(copy_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_rk, this->d_p0, this->arrSize); // p0 = rk
                POP_RANGE(); // 6

                // Initial residual norm res0 = |r0|
                PUSH_RANGE("cgSolver: res0 = |r0|", 6);
                DeviceUtils::launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_resk, zero_fp64, this->auxSize);
                DeviceUtils::launchKernel(dot_product<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_rk, this->d_rk, this->d_resk, this->arrSize); // resk = rk . rk (partial)
                if (this->IterSolvers_comm.getLibSize() > 1)
                {
                    PUSH_RANGE("cgSolver: comms", 7);
                    DeviceUtils::launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, zero_fp64, this->auxSize);
                    this->IterSolvers_comm.Allreduce_Sum(this->d_resk, this->d_mpiTmp, 1);
                    DeviceUtils::launchKernel(copy_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, this->d_resk, this->auxSize); // resk = dot(rk,rk)
                    POP_RANGE(); // 7
                }
                // res0 = sqrt(resk)
                DeviceUtils::launchKernel(copy_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_resk, this->d_res0, this->auxSize);
                DeviceUtils::launchKernel(array_sqrt<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_res0, this->auxSize);
                DeviceUtils::launchKernel(scale<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->tol, this->d_res0, this->auxSize); // res0 = tol*res0
                DeviceMemory<ITYPE, double>::copyDeviceToHost(this->auxSize, this->d_res0, this->res0); // copy res0 to host for convergence check
                POP_RANGE(); // 6
            }
            POP_RANGE(); // 5

            // Iterations
            PUSH_RANGE("cgSolver: iterations", 5);
            while (this->iter < this->maxIters)
            {
                PUSH_RANGE("cgSolver: iteration", 6);
                {
                    // Increment iter
                    this->iter++;

                    // Matvec
                    PUSH_RANGE("cgSolver: matvec", 7);
                    matvec(this->d_p0, this->d_Ax); // Ax = A*p0
                    POP_RANGE(); // 7

                    // Compute alpha = (rk.rk) / (pk.Apk)
                    PUSH_RANGE("cgSolver: alpha", 7);
                    DeviceUtils::launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_alpha, zero_fp64, this->auxSize);
                    DeviceUtils::launchKernel(dot_product<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_p0, this->d_Ax, this->d_alpha, this->arrSize); // alpha = pk.Apk (partial)
                    if (this->IterSolvers_comm.getLibSize() > 1)
                    {
                        PUSH_RANGE("cgSolver: comms", 8);
                        DeviceUtils::launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, zero_fp64, this->auxSize);
                        this->IterSolvers_comm.Allreduce_Sum(this->d_alpha, this->d_mpiTmp, 1);
                        DeviceUtils::launchKernel(copy_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, this->d_alpha, this->auxSize); // alpha = pk.Apk
                        POP_RANGE(); // 8
                    }
                    // Invert alpha and multiply with resk
                    DeviceUtils::launchKernel(array_invert<ITYPE,double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_alpha, this->auxSize); // alpha = 1/alpha
                    DeviceUtils::launchKernel(pointwise_multiply<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_resk, this->d_alpha, this->auxSize); // alpha = resk * alpha
                    DeviceMemory<ITYPE, double>::copyDeviceToHost(this->auxSize, this->d_alpha, this->alpha); // copy alpha to host for axpy
                    POP_RANGE(); // 7

                    // Update x_sol
                    PUSH_RANGE("cgSolver: update x_sol", 7);
                    DeviceUtils::launchKernel(axpy<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, (RTYPE)this->alpha[0], this->d_p0, this->d_x_sol, this->arrSize); // x_sol += alpha*p0
                    POP_RANGE(); // 7

                    // Update rk+1 = rk - alpha*Apk
                    PUSH_RANGE("cgSolver: update rk", 7);
                    this->alpha[0] = -this->alpha[0]; // alpha = -alpha for axpy
                    DeviceUtils::launchKernel(axpy<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, (RTYPE)this->alpha[0], this->d_Ax, this->d_rk, this->arrSize); // rk += (-alpha)*Apk
                    POP_RANGE(); // 7

                    // Compute the new residual norm resk = |rk|
                    PUSH_RANGE("cgSolver: resk = |rk|", 7);
                    DeviceUtils::launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_beta, zero_fp64, this->auxSize);
                    DeviceUtils::launchKernel(dot_product<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_rk, this->d_rk, this->d_beta, this->arrSize); // beta = rk+1 . rk+1 (partial)
                    if (this->IterSolvers_comm.getLibSize() > 1)
                    {
                        PUSH_RANGE("cgSolver: comms", 8);
                        DeviceUtils::launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, zero_fp64, this->auxSize);
                        this->IterSolvers_comm.Allreduce_Sum(this->d_beta, this->d_mpiTmp, 1);
                        DeviceUtils::launchKernel(copy_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, this->d_beta, this->auxSize); // beta = dot(rk,rk)
                        POP_RANGE(); // 8
                    }
                    POP_RANGE(); // 7

                    // Check convergence
                    PUSH_RANGE("cgSolver: check convergence", 7);
                    DeviceMemory<ITYPE, double>::copyDeviceToHost(this->auxSize, this->d_beta, this->beta); // copy beta to host for convergence check
                    out_sqrtRes = std::sqrt(this->beta[0]);
                    POP_RANGE(); // 7
                    if (out_sqrtRes <= this->res0[0])
                    {
                        POP_RANGE(); // 6
                        break; // Converged, exit iteration loop
                    }

                    // Compute beta = (rk+1.rk+1) / (rk.rk) = aux/resk
                    PUSH_RANGE("cgSolver: beta", 7);
                    DeviceUtils::launchKernel(copy_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_beta, this->d_aux, this->auxSize);          // aux = beta (store new rk.rk for next stage)
                    DeviceUtils::launchKernel(array_invert<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_resk, this->auxSize);                     // resk = 1/resk
                    DeviceUtils::launchKernel(pointwise_multiply<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_resk, this->d_beta, this->auxSize); // beta = beta * resk
                    POP_RANGE(); // 7

                    // Update pk+1 = rk+1 + beta*pk
                    PUSH_RANGE("cgSolver: update pk", 7);
                    DeviceMemory<ITYPE, double>::copyDeviceToHost(this->auxSize, this->d_beta, this->beta);                                                                  // copy beta to host for axpy
                    DeviceUtils::launchKernel(scale<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, (RTYPE)this->beta[0], this->d_p0, this->auxSize);     // p0 = beta*p0
                    DeviceUtils::launchKernel(axpy<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, (RTYPE)1, this->d_rk, this->d_p0, this->arrSize); // p0 += rk
                    POP_RANGE(); // 7

                    // resk is now aux
                    PUSH_RANGE("cgSolver: update resk", 7);
                    DeviceUtils::launchKernel(copy_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_aux, this->d_resk, this->auxSize); // resk = aux
                    POP_RANGE(); // 7
                }
                POP_RANGE(); // 6
            }
            POP_RANGE(); // 5
        #else
        #endif
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

/*
template <typename ITYPE, typename RTYPE>
void ConjugateGradient<ITYPE, RTYPE>::fpcgSolver(const MatVecOp &matvec, const PrecondOp &precond)
{
    double out_sqrtRes;
    double cgTime = this->IterSolvers_comm.timeFunction([&] {
    PUSH_RANGE("ConjugateGradient::fpcgSolver", 1)
#ifdef USE_GPU
    // Initial step

    // 0. initialize solution to x0 and all else to zero
    PUSH_RANGE("cgSolver: x_sol = x0", 2)
    launchKernel(copy_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_x0, this->d_x_sol, this->arrSize); // x_sol = x0
    POP_RANGE

    PUSH_RANGE("cgSolver: zero arrays", 2)
    memset(this->res0, 0, 1 * sizeof(RTYPE));
    memset(this->resk, 0, 1 * sizeof(RTYPE));
    memset(this->aux, 0, 1 * sizeof(RTYPE));
    memset(this->alpha, 0, 1 * sizeof(RTYPE));
    memset(this->beta, 0, 1 * sizeof(RTYPE));
    memset(this->tmpDot, 0, 1 * sizeof(double));

    RTYPE zero = static_cast<RTYPE>(0);
    double zero_fp64 = static_cast<double>(0);
    launchKernel(set_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_r0, zero, this->arrSize);
    launchKernel(set_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_p0, zero, this->arrSize);
    launchKernel(set_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_rk, zero, this->arrSize);
    launchKernel(set_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_zk, zero, this->arrSize);
    launchKernel(set_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_Ax, zero, this->arrSize);
    launchKernel(set_array<ITYPE, RTYPE>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_res0, zero, this->auxSize);
    launchKernel(set_array<ITYPE, RTYPE>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_resk, zero, this->auxSize);
    launchKernel(set_array<ITYPE, RTYPE>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_aux, zero, this->auxSize);
    launchKernel(set_array<ITYPE, RTYPE>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_alpha, zero, this->auxSize);
    launchKernel(set_array<ITYPE, RTYPE>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_beta, zero, this->auxSize);
    launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_tmpDot, zero_fp64, this->auxSize);
    POP_RANGE

    // Initial step
    PUSH_RANGE("cgSolver: initial step", 2)

    // 1. compute initial residual r0 = b - A*x0
    PUSH_RANGE("cgSolver: matvec", 3)
    matvec(this->d_x0, this->d_Ax); // Ax = A*x0
    POP_RANGE

    PUSH_RANGE("cgSolver: r0 = b - Ax", 3)
    launchKernel(copy_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_b, this->d_r0, this->arrSize); // r0 = b
    RTYPE negOne = static_cast<RTYPE>(-1);
    launchKernel(axpy<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, negOne, this->d_Ax, this->d_r0, this->arrSize); // r0 = r0 - Ax
    POP_RANGE

    // 2. Precondition M z0 = r0
    PUSH_RANGE("cgSolver: precond", 3)
    precond(this->d_r0, this->d_zk); // zk = M^-1 * r0
    POP_RANGE

    // 3. compute initial residual norm |r0| and initialize rk, rk.zk, p0
    PUSH_RANGE("cgSolver: residual0", 3)
    launchKernel(copy_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_r0, this->d_rk, this->arrSize); // rk = r0
    launchKernel(copy_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_zk, this->d_p0, this->arrSize); // p0 = zk
    launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_dotTmp1, zero_fp64, this->auxSize);
    launchKernel(dot_product<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_r0, this->d_r0, this->d_dotTmp1, this->arrSize); // tmpDot = r0 . r0
    launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_dotTmp2, zero_fp64, this->auxSize);
    launchKernel(dot_product<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_rk, this->d_zk, this->d_dotTmp2, this->arrSize); // tmpDot = rk . zk
    if (this->IterSolvers_comm.isParallel) {
        PUSH_RANGE("cgSolver: comms", 4)
        launchFillBuffer<ITYPE,double>(this->d_sendbuf, this->kernelStream, this->d_dotTmp1, this->d_dotTmp2);
        this->IterSolvers_comm.Allreduce_Sum(this->d_sendbuf, this->d_recvbuf, this->nargs);
        launchScatterBuffer<ITYPE,double>(this->d_recvbuf, this->kernelStream, this->d_dotTmp1, this->d_dotTmp2);
        POP_RANGE
    }
    CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_dotTmp1, 1 * sizeof(double), cudaMemcpyDeviceToHost));
    this->res0[0] = static_cast<RTYPE>(std::sqrt(this->tmpDot[0])); // res0 = |r0|
    CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_dotTmp2, 1 * sizeof(double), cudaMemcpyDeviceToHost));
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
        launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_tmpDot, zero_fp64, this->auxSize);
        launchKernel(dot_product<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_p0, this->d_Ax, this->d_tmpDot, this->arrSize); // tmpDot = p0 . Ax
        if (this->IterSolvers_comm.isParallel) {
            int count = static_cast<int>(1);
            PUSH_RANGE("cgSolver: comms", 4)
            launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, zero_fp64, this->auxSize);
            this->IterSolvers_comm.Allreduce_Sum(this->d_tmpDot, this->d_mpiTmp, count);
            launchKernel(copy_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, this->d_tmpDot, this->auxSize);
            POP_RANGE
        }
        CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_tmpDot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        this->alpha[0] = this->resk[0] / static_cast<RTYPE>(this->tmpDot[0]);
        POP_RANGE // 4: alpha

        // 5. Update solution xk = xk-1 + alpha*pk-1 and residual rk = rk-1 - alpha*Apk-1
        PUSH_RANGE("cgSolver: update step", 4)
        launchKernel(axpy<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->alpha[0], this->d_p0, this->d_x_sol, this->arrSize); // x_sol = x_sol + alpha*p0
        RTYPE negAlpha = -this->alpha[0];
        launchKernel(axpy<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, negAlpha, this->d_Ax, this->d_rk, this->arrSize); // rk = rk - alpha*Ax
        POP_RANGE                                                                                                                                   // 4: update step

        // 6. Compute new rk.rk
        PUSH_RANGE("cgSolver: residualk", 4)
        launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_tmpDot, zero_fp64, this->auxSize);
        launchKernel(dot_product<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_rk, this->d_rk, this->d_tmpDot, this->arrSize); // tmpDot = rk . rk
        if (this->IterSolvers_comm.isParallel) {
            PUSH_RANGE("cgSolver: comms", 4)
            int count = static_cast<int>(1);
            launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, zero_fp64, this->auxSize);
            this->IterSolvers_comm.Allreduce_Sum(this->d_tmpDot, this->d_mpiTmp, count);
            launchKernel(copy_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, this->d_tmpDot, this->auxSize);
            POP_RANGE
        }
        CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_tmpDot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        this->aux[0] = static_cast<RTYPE>(this->tmpDot[0]); // aux = rk.rk
        POP_RANGE                                           // 4: residualk

        // Check convergence
        out_sqrtRes = std::sqrt(this->tmpDot[0]);
        if (out_sqrtRes < this->tol * static_cast<double>(this->res0[0]))
        {
            // Converged
            POP_RANGE // 3: iteration
            break;
        }

        // 7. Compute aux = rk+1 . zk
        PUSH_RANGE("cgSolver: rk+1.zk ", 4)
        launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_dotTmp1, zero_fp64, this->auxSize);
        launchKernel(dot_product<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_rk, this->d_zk, this->d_dotTmp1, this->arrSize); // tmpDot = rk . zk
        POP_RANGE // 4: rk+1.zk

        // 8. Precondition M * zk+1 = rk+1
        PUSH_RANGE("cgSolver: precond", 4)
        precond(this->d_rk, this->d_zk); // zk = M^-1 * rk
        POP_RANGE                        // 4: precond

        // 9. Compute rk+1 . zk+1
        PUSH_RANGE("cgSolver: rk+1.zk+1", 4)
        launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_dotTmp2, zero_fp64, this->auxSize);
        launchKernel(dot_product<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_rk, this->d_zk, this->d_dotTmp2, this->arrSize); // tmpDot = rk . zk
        POP_RANGE // 4: rk+1.zk+1

        // Comms
        if (this->IterSolvers_comm.isParallel)
        {
            PUSH_RANGE("cgSolver: comms", 4)
            launchFillBuffer<ITYPE, double>(this->d_sendbuf, this->kernelStream, this->d_dotTmp1, this->d_dotTmp2);
            this->IterSolvers_comm.Allreduce_Sum(this->d_sendbuf, this->d_recvbuf, this->nargs);
            launchScatterBuffer<ITYPE, double>(this->d_recvbuf, this->kernelStream, this->d_dotTmp1, this->d_dotTmp2);
            POP_RANGE
        }

        // 10. Compute beta
        PUSH_RANGE("cgSolver: beta", 4)
        CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_dotTmp1, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        this->aux[0] = static_cast<RTYPE>(this->tmpDot[0]); // aux = rk+1 . zk
        CUDA_CHECK(cudaMemcpy(this->tmpDot, this->d_dotTmp2, 1 * sizeof(double), cudaMemcpyDeviceToHost));
        this->beta[0] = static_cast<RTYPE>(this->tmpDot[0]); // beta = rk+1 . zk+1
        RTYPE tmp = this->beta[0];
        this->beta[0] =  ( this->beta[0] - this->aux[0] )  / this->resk[0]; // beta = (rk+1.zk+1 - rk.zk) / rk.zk
        this->resk[0] = tmp;
        POP_RANGE // 4: beta


        // 11. Update search direction pk = rk + beta*pk-1
        PUSH_RANGE("cgSolver: Update pk", 4)
        const RTYPE one = static_cast<RTYPE>(1);
        launchKernel(scale<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->beta[0], this->d_p0, this->arrSize); // p0 = beta*p0
        launchKernel(axpy<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, one, this->d_zk, this->d_p0, this->arrSize); // p0 = zk + beta*p0
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

    // comm buffers
    double *source_buf = (double *)calloc(2, sizeof(double));
    double *target_buf = (double *)calloc(2, sizeof(double));
    int count = static_cast<int>(2);

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
    this->resk[0] = 0;
    TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->rk, this->zk, this->resk);  // resk = rk . zk
    TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->r0, this->r0, this->res0); // res0 = r0 . r0
    if (this->IterSolvers_comm.isParallel) {
        source_buf[0] = static_cast<double>(this->res0[0]);
        source_buf[1] = static_cast<double>(this->resk[0]);
        target_buf[0] = static_cast<double>(0);
        target_buf[1] = static_cast<double>(0);
        this->IterSolvers_comm.Allreduce_Sum(source_buf, target_buf, count);
        this->res0[0] = static_cast<RTYPE>(target_buf[0]);
        this->resk[0] = static_cast<RTYPE>(target_buf[1]);
    }
    this->res0[0] = std::sqrt(this->res0[0]);            // res0 = |r0|

    // Iterate
    this->iter = 0;
    while (this->iter < this->maxIters)
    {
        this->iter++; // increment iteration counter then compute

        // 4. Compute alpha
        //    -- rk.zk should already be computed, so only need pk.Apk
        matvec(this->p0, this->Ax);                                                             // Ax = A*p0
        TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->p0, this->Ax, this->alpha); // alpha = p0 . Ax
        if (this->IterSolvers_comm.isParallel) {
            int count = static_cast<int>(1);
            this->tmpDot[0] = static_cast<double>(this->alpha[0]);
            this->mpiTmp[0] = static_cast<double>(0);
            this->IterSolvers_comm.Allreduce_Sum(this->tmpDot, this->mpiTmp, count);
            this->alpha[0] = static_cast<RTYPE>(this->mpiTmp[0]);
        }
        this->alpha[0] = this->resk[0] / this->alpha[0];

        // 5. Update solution xk = xk-1 + alpha*pk-1 and residual rk = rk-1 - alpha*Apk-1
        TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, this->alpha[0], this->p0, this->x_sol); // x_sol = x_sol + alpha*p0
        TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, -this->alpha[0], this->Ax, this->rk);   // rk = rk - alpha*Ax

        // 6. Compute new rk.rk
        TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->rk, this->rk, this->aux); // aux = rk . rk
        if (this->IterSolvers_comm.isParallel) {
            int count = static_cast<int>(1);
            this->tmpDot[0] = static_cast<double>(this->aux[0]);
            this->mpiTmp[0] = static_cast<double>(0);
            this->IterSolvers_comm.Allreduce_Sum(this->tmpDot, this->mpiTmp, count);
            this->aux[0] = static_cast<RTYPE>(this->mpiTmp[0]);
        }
        this->tmpDot[0] = static_cast<double>(this->aux[0]);

        // Check convergence
        out_sqrtRes = std::sqrt(this->tmpDot[0]);
        if (out_sqrtRes < this->tol * static_cast<double>(this->res0[0]))
        {
            break;
        }

        // 7. Compute rk+1 . zk (partial)
        TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->rk, this->zk, this->aux); // aux = rk . zk

        // 8. precondition M * zk+1 = rk+1
        precond(this->rk, this->zk); // zk = M^-1 * rk

        // 9. Compute beta = rk+1 . zk+1 (partial)
        TensorUtils<ITYPE, RTYPE>::dot_product(this->arrSize, this->rk, this->zk, this->beta); // beta = rk . zk

        // 10. Comms for aux and beta
        if (this->IterSolvers_comm.isParallel) {
            source_buf[0] = static_cast<double>(this->aux[0]);
            source_buf[1] = static_cast<double>(this->beta[0]);
            target_buf[0] = static_cast<double>(0);
            target_buf[1] = static_cast<double>(0);
            this->IterSolvers_comm.Allreduce_Sum(source_buf, target_buf, count);
            this->aux[0] = static_cast<RTYPE>(target_buf[0]);
            this->beta[0] = static_cast<RTYPE>(target_buf[1]);
        }

        // 11. Finish beta = rk+1 . (zk+1 - zk) / rk . zk
        RTYPE tmp = this->beta[0]; // Store rk+1.zk+1
        this->beta[0] = ( this->beta[0] - this->aux[0] ) / this->resk[0];
        this->resk[0] = tmp; // resk = (rk.zk)new

        // 12. update search direction pk = zk + beta*pk-1
        const RTYPE one = static_cast<RTYPE>(1);
        TensorUtils<ITYPE, RTYPE>::scale(this->arrSize, this->beta[0], this->p0); // p0 = beta*p0
        TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, one, this->zk, this->p0);  // p0 = zk + beta*p0
    }

    // Free comm buffers
    free(source_buf);
    free(target_buf);

#endif
    });
    POP_RANGE // 1: cgSolver
    if (this->IterSolvers_comm.getLibRank() == 0)
    {
        // Write to logfile
        // out_sqrtRes in scientific format
        this->logfile << this->iter << " , " << std::scientific << out_sqrtRes << " , " << cgTime * 1000.0 << std::endl;
        this->logfile.flush();
    }
}
*/

template class ConjugateGradient<uint32_t, float>;
template class ConjugateGradient<uint64_t, float>;
template class ConjugateGradient<uint32_t, double>;
template class ConjugateGradient<uint64_t, double>;
#ifdef USE_GPU
    template class ConjugateGradient<uint32_t, DeviceUtils::bf16>;
    template class ConjugateGradient<uint64_t, DeviceUtils::bf16>;
#endif
