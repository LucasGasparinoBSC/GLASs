#include "IterSolvers.hpp"

// Empty constructor
template <typename ITYPE, typename RTYPE>
IterSolvers<ITYPE, RTYPE>::IterSolvers() {
    PUSH_RANGE("IterSolvers::Constructor(empty)", 3);
    this->flag_planned = false;
    this->flag_setup = false;
    this->arrSize = 0;
    this->maxIters = 0;
    this->iter = 0;
    this->tol = -1e-6;
    this->tmpDot = nullptr;
    this->d_tmpDot = nullptr;
    this->mpiTmp = nullptr;
    this->d_mpiTmp = nullptr;
    this->x_sol = nullptr;
    this->d_x_sol = nullptr;
    this->x0 = nullptr;
    this->d_x0 = nullptr;
    this->r0 = nullptr;
    this->d_r0 = nullptr;
    this->rk = nullptr;
    this->d_rk = nullptr;
    this->zk = nullptr;
    this->d_zk = nullptr;
    this->Ax = nullptr;
    this->d_Ax = nullptr;
    this->b = nullptr;
    this->d_b = nullptr;
    this->res0 = nullptr;
    this->d_res0 = nullptr;
    this->resk = nullptr;
    this->d_resk = nullptr;
    this->aux = nullptr;
    this->d_aux = nullptr;
    POP_RANGE();
}

// Param constructor
template <typename ITYPE, typename RTYPE>
IterSolvers<ITYPE, RTYPE>::IterSolvers(MPI_Comm& c_comm, ITYPE arrSize, ITYPE maxIters, double tol)
{

    PUSH_RANGE("IterSolvers::Constructor", 3);
    // Store the comms object
    IterSolvers_comm.setup(c_comm);
    if (IterSolvers_comm.getWorldRank() == 0) std::cout << "--| IterSolvers: initializing solver" << std::endl;

    // Plan the solver (arrSize is per-rank!!!!)
    plan(arrSize, maxIters, tol);
    if (IterSolvers_comm.getWorldRank() == 0) std::cout << "--| IterSolvers: solvers initialized!" << std::endl;
    POP_RANGE();
}

// Destructor
template <typename ITYPE, typename RTYPE>
IterSolvers<ITYPE, RTYPE>::~IterSolvers()
{
    if (IterSolvers_comm.getWorldRank() == 0) std::cout << "--| IterSolvers: destroying solver object..." << std::endl;

    // Free host memory
    PUSH_RANGE("IterSolvers::Destructor", 3);
    if (tmpDot)
        free(tmpDot);
    if (mpiTmp)
        free(mpiTmp);
    if (x_sol)
        free(x_sol);
    if (r0)
        free(r0);
    if (rk)
        free(rk);
    if (zk)
        free(zk);
    if (Ax)
        free(Ax);
    if (res0)
        free(res0);
    if (resk)
        free(resk);
    if (aux)
        free(aux);

    #ifdef USE_GPU
        // Free device memory
        DeviceMemory<ITYPE, double>::deviceFree(d_tmpDot);
        DeviceMemory<ITYPE, double>::deviceFree(d_mpiTmp);
        DeviceMemory<ITYPE, double>::deviceFree(d_res0);
        DeviceMemory<ITYPE, double>::deviceFree(d_resk);
        DeviceMemory<ITYPE, double>::deviceFree(d_aux);
        DeviceMemory<ITYPE, RTYPE>::deviceFree(d_x_sol);
        DeviceMemory<ITYPE, RTYPE>::deviceFree(d_r0);
        DeviceMemory<ITYPE, RTYPE>::deviceFree(d_rk);
        DeviceMemory<ITYPE, RTYPE>::deviceFree(d_zk);
        DeviceMemory<ITYPE, RTYPE>::deviceFree(d_Ax);
    #endif
    POP_RANGE();

    flag_planned = false;
    flag_setup = false;
}

// Param constructor
template <typename ITYPE, typename RTYPE>
void IterSolvers<ITYPE, RTYPE>::plan(ITYPE arrSize, ITYPE maxIters, double tol)
{
    if (IterSolvers_comm.getWorldRank() == 0) std::cout << "--| IterSolvers: planning solver" << std::endl;

    // Allocate host memory using calloc (ensures init to 0)
    PUSH_RANGE("IterSolvers::plan", 3);
    this->arrSize = arrSize;
    this->maxIters = maxIters;
    this->iter = 0;
    this->tol = tol;
    x_sol = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    r0 = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    rk = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    zk = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    Ax = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    res0 = (double *)calloc(auxSize, sizeof(double));
    resk = (double *)calloc(auxSize, sizeof(double));
    aux  = (double *)calloc(auxSize, sizeof(double));
    tmpDot = (double*)calloc(auxSize, sizeof(double));
    mpiTmp = (double*)calloc(auxSize, sizeof(double));

    #ifdef USE_GPU
        // Allocate device arrays
        d_x_sol = DeviceMemory<ITYPE, RTYPE>::deviceCalloc(arrSize);
        d_r0 = DeviceMemory<ITYPE, RTYPE>::deviceCalloc(arrSize);
        d_rk = DeviceMemory<ITYPE, RTYPE>::deviceCalloc(arrSize);
        d_zk = DeviceMemory<ITYPE, RTYPE>::deviceCalloc(arrSize);
        d_Ax = DeviceMemory<ITYPE, RTYPE>::deviceCalloc(arrSize);
        d_x_sol = DeviceMemory<ITYPE, RTYPE>::deviceCalloc(arrSize);
        d_res0 = DeviceMemory<ITYPE, double>::deviceCalloc(auxSize);
        d_resk = DeviceMemory<ITYPE, double>::deviceCalloc(auxSize);
        d_aux  = DeviceMemory<ITYPE, double>::deviceCalloc(auxSize);
        d_tmpDot = DeviceMemory<ITYPE, double>::deviceCalloc(auxSize);
        d_mpiTmp = DeviceMemory<ITYPE, double>::deviceCalloc(auxSize);

        // Define kernel and aux launch grids
        kernelBlock = dim3(TILE_SIZE, 1, 1);
        ITYPE numBlocks = (this->arrSize + TILE_SIZE - 1) / TILE_SIZE;
        numBlocks = std::min(numBlocks, (ITYPE)MAX_BLOCKS);
        kernelGrid = dim3(numBlocks, 1, 1);
        auxBlock = dim3(this->auxSize, 1, 1);
        auxGrid = dim3(this->auxSize, 1, 1);

        // Create CUDA stream for kernel launches
        CUDA_CHECK(cudaStreamCreate(&kernelStream));
    #endif
    POP_RANGE();

    flag_planned = true;
    if (IterSolvers_comm.getWorldRank() == 0) std::cout << "--| IterSolvers: solvers planned!" << std::endl;
}

template <typename ITYPE, typename RTYPE>
void IterSolvers<ITYPE, RTYPE>::setup(RTYPE *inicond, RTYPE *rhs) {
    // Assign initial condition and RHS
    PUSH_RANGE("IterSolvers::setup", 3);
    #ifdef USE_GPU
        this->d_x0 = inicond;
        this->d_b = rhs;
    #else
        this->x0 = inicond;
        this->b = rhs;
    #endif
    POP_RANGE();
    flag_setup = true;
}

// Shallow copy of the solution to the client pointer
// NOTE: this means the clientPtr will point to the internal solver memory!
//       This is done to avoid unnecessary copies of large arrays.
//       Client must be aware of this and not free the pointer!
//       For GPU scenarios, clientPtr needs to exist in device memory!
//       For OpenACC cleint data, wrap call in #pragma acc host_data use_device(clientPtr)
template <typename ITYPE, typename RTYPE>
void IterSolvers<ITYPE, RTYPE>::getSolution(RTYPE* clientPtr) {
    PUSH_RANGE("IterSolvers::getSolution", 3);
    #ifdef USE_GPU
        DeviceUtils::launchKernel(copy_array<ITYPE, RTYPE>, kernelGrid, kernelBlock, kernelStream, this->d_x_sol, clientPtr, this->arrSize);
    #else
        TensorUtils<ITYPE,RTYPE>::copy_array(this->arrSize, this->x_sol, clientPtr);
    #endif
    POP_RANGE();
}

template <typename ITYPE, typename RTYPE>
ITYPE IterSolvers<ITYPE, RTYPE>::getSize() {
    return this->arrSize;
}

template class IterSolvers<uint32_t, float>;
template class IterSolvers<uint64_t, float>;
template class IterSolvers<uint32_t, double>;
template class IterSolvers<uint64_t, double>;
#ifdef USE_GPU
    template class IterSolvers<uint32_t, DeviceUtils::bf16>;
    template class IterSolvers<uint64_t, DeviceUtils::bf16>;
#endif