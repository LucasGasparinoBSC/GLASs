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
            // Impelement
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

// Non-preconditioned Conjugate Gradient solver
template <typename ITYPE, typename RTYPE>
void ConjugateGradient<ITYPE, RTYPE>::fpcgSolver(const MatVecOp& matvec, const PrecondOp& precond) {
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

                // Preconditioning: z0 = M^-1 r0
                PUSH_RANGE("cgSolver: preconditioning", 6);
                precond(this->d_rk, this->d_zk); // z0 = M^-1 r0
                POP_RANGE(); // 6

                // Initial p0
                PUSH_RANGE("cgSolver: p0 = z0", 6);
                DeviceUtils::launchKernel(copy_array<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_zk, this->d_p0, this->arrSize); // p0 = zk
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

                // Preconditioned residual rk.zk
                PUSH_RANGE("cgSolver: resk = rk.zk", 6);
                DeviceUtils::launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_resk, zero_fp64, this->auxSize);
                DeviceUtils::launchKernel(dot_product<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_rk, this->d_zk, this->d_resk, this->arrSize); // resk = rk.zk
                if (this->IterSolvers_comm.getLibSize() > 1)
                {
                    PUSH_RANGE("cgSolver: comms", 7);
                    DeviceUtils::launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, zero_fp64, this->auxSize);
                    this->IterSolvers_comm.Allreduce_Sum(this->d_resk, this->d_mpiTmp, 1);
                    DeviceUtils::launchKernel(copy_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, this->d_resk, this->auxSize); // resk = dot(rk.zk)
                    POP_RANGE(); // 7
                }
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

                    // Compute alpha = (rk.zk) / (pk.Apk)
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
                    // Invert alpha and multiply with resk (rk.zk)
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

                    // NOTE: this needs optimization!
                    // Flexible beta: beta = (rk+1.zk+1 - rk.zk) / (rk.zk) = (resk+1 - aux) / resk
                    // Compute aux = rk+1.zk
                    PUSH_RANGE("cgSolver: rk+1.zk", 7);
                    DeviceUtils::launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_aux, zero_fp64, this->auxSize);
                    DeviceUtils::launchKernel(dot_product<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_rk, this->d_zk, this->d_aux, this->arrSize); // aux = rk+1.zk (partial)
                    if (this->IterSolvers_comm.getLibSize() > 1)
                    {
                        PUSH_RANGE("cgSolver: comms", 8);
                        DeviceUtils::launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, zero_fp64, this->auxSize);
                        this->IterSolvers_comm.Allreduce_Sum(this->d_aux, this->d_mpiTmp, 1);
                        DeviceUtils::launchKernel(copy_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, this->d_aux, this->auxSize); // aux = rk+1.zk
                        POP_RANGE(); // 8
                    }
                    POP_RANGE(); // 7

                    // Update zk+1 = M^-1 rk+1
                    PUSH_RANGE("cgSolver: update zk", 7);
                    precond(this->d_rk, this->d_zk);
                    POP_RANGE(); // 7

                    // Compute beta = (rk+1.zk+1)
                    PUSH_RANGE("cgSolver: beta flexible", 7);
                    DeviceUtils::launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_beta, zero_fp64, this->auxSize);
                    DeviceUtils::launchKernel(dot_product<ITYPE, RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_rk, this->d_zk, this->d_beta, this->arrSize); // beta = rk+1.zk+1 (partial)
                    if (this->IterSolvers_comm.getLibSize() > 1)
                    {
                        PUSH_RANGE("cgSolver: comms", 8);
                        DeviceUtils::launchKernel(set_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, zero_fp64, this->auxSize);
                        this->IterSolvers_comm.Allreduce_Sum(this->d_beta, this->d_mpiTmp, 1);
                        DeviceUtils::launchKernel(copy_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_mpiTmp, this->d_beta, this->auxSize); // beta = rk+1.zk+1
                        POP_RANGE(); // 8
                    }
                    DeviceUtils::launchKernel(copy_array<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_beta, this->d_aux, this->auxSize); // aux = rk+1.zk+1 (store for next stage)
                    POP_RANGE(); // 7

                    // beta = (beta-aux)/resk
                    PUSH_RANGE("cgSolver: beta flexible update", 7);
                    DeviceUtils::launchKernel(array_invert<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_resk, this->auxSize);                     // resk = 1/resk
                    DeviceUtils::launchKernel(axpy<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, negOne, this->d_aux, this->d_beta, this->auxSize);        // beta = beta - aux
                    DeviceUtils::launchKernel(pointwise_multiply<ITYPE, double>, this->auxGrid, this->auxBlock, this->kernelStream, this->d_resk, this->d_beta, this->auxSize); // beta = beta * (1/resk)
                    POP_RANGE(); // 7

                    // Update pk+1 = zk+1 + beta*pk
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

template class ConjugateGradient<uint32_t, float>;
template class ConjugateGradient<uint64_t, float>;
template class ConjugateGradient<uint32_t, double>;
template class ConjugateGradient<uint64_t, double>;
#ifdef USE_GPU
    template class ConjugateGradient<uint32_t, DeviceUtils::bf16>;
    template class ConjugateGradient<uint64_t, DeviceUtils::bf16>;
#endif
