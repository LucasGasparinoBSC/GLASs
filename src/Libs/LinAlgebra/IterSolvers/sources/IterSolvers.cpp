#include "IterSolvers.hpp"

// Empty constructor
template <typename ITYPE, typename RTYPE>
IterSolvers<ITYPE, RTYPE>::IterSolvers() {
    std::cout << "--| IterSolvers: instantiating solver..." << std::endl;
    PUSH_RANGE("IterSolvers::Constructor(empty)", 0)
    this->flag_planned = false;
    this->flag_setup = false;
    this->arrSize = 0;
    this->maxIters = 0;
    this->iter = 0;
    this->tol = -1e-6;
    this->tmpDot = nullptr;
    this->d_tmpDot = nullptr;
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
    POP_RANGE
    std::cout << "--| IterSolvers: solver instantiated!" << std::endl;
}

// Param constructor
template <typename ITYPE, typename RTYPE>
IterSolvers<ITYPE, RTYPE>::IterSolvers(ITYPE arrSize, ITYPE maxIters, double tol)
{
    std::cout << "--| IterSolvers: initializing solver" << std::endl;
    PUSH_RANGE("IterSolvers::Constructor(param)", 0)
    plan(arrSize, maxIters, tol);
    POP_RANGE
    std::cout << "--| IterSolvers: solvers initialized!" << std::endl;
}

// Destructor
template <typename ITYPE, typename RTYPE>
IterSolvers<ITYPE, RTYPE>::~IterSolvers()
{
    std::cout << "--| IterSolvers: destroying solver object..." << std::endl;

    // Free host memory
    PUSH_RANGE("IterSolvers::Destructor", 0);
    if (tmpDot)
        free(tmpDot);
    if (x_sol)
        free(x_sol);
    if (x0)
        free(x0);
    if (r0)
        free(r0);
    if (rk)
        free(rk);
    if (zk)
        free(zk);
    if (Ax)
        free(Ax);
    if (b)
        free(b);
    if (res0)
        free(res0);
    if (resk)
        free(resk);
    if (aux)
        free(aux);
    POP_RANGE

#ifdef USE_GPU
    // Free device memory
    PUSH_RANGE("IterSolvers::Destructor -> device", 0);
    CUDA_CHECK(cudaFree(d_tmpDot));
    CUDA_CHECK(cudaFree(d_x_sol));
    CUDA_CHECK(cudaFree(d_x0));
    CUDA_CHECK(cudaFree(d_r0));
    CUDA_CHECK(cudaFree(d_rk));
    CUDA_CHECK(cudaFree(d_zk));
    CUDA_CHECK(cudaFree(d_Ax));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_res0));
    CUDA_CHECK(cudaFree(d_resk));
    CUDA_CHECK(cudaFree(d_aux));
    POP_RANGE
#endif

    flag_planned = false;
    flag_setup = false;
}

// Param constructor
template <typename ITYPE, typename RTYPE>
void IterSolvers<ITYPE, RTYPE>::plan(ITYPE arrSize, ITYPE maxIters, double tol)
{
    std::cout << "--| IterSolvers: planning solver" << std::endl;

    // Allocate host memory using calloc (ensures init to 0)
    PUSH_RANGE("IterSolvers::plan", 1)
    this->arrSize = arrSize;
    this->maxIters = maxIters;
    this->iter = 0;
    this->tol = tol;
    x_sol = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    x0 = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    r0 = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    rk = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    zk = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    Ax = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    b = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    res0 = (RTYPE *)calloc(1, sizeof(RTYPE));
    resk = (RTYPE *)calloc(1, sizeof(RTYPE));
    aux = (RTYPE *)calloc(1, sizeof(RTYPE));
    tmpDot = (double*)calloc(1, sizeof(double));
    POP_RANGE

#ifdef USE_GPU
    // Allocate device arrays
    PUSH_RANGE("IterSolvers::plan -> device", 1);
    CUDA_CHECK(cudaMalloc(&d_x_sol, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(d_x_sol, 0, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_x0, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(d_x0, 0, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_r0, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(d_r0, 0, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_rk, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(d_rk, 0, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_zk, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(d_zk, 0, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_Ax, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(d_Ax, 0, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_b, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(d_b, 0, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_res0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(d_res0, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_resk, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(d_resk, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_aux, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMemset(d_aux, 0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_tmpDot, 1 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_tmpDot, 0, 1 * sizeof(double)));
    POP_RANGE
#endif

    flag_planned = true;
    std::cout << "--| IterSolvers: solvers planned!" << std::endl;
}

template <typename ITYPE, typename RTYPE>
void IterSolvers<ITYPE, RTYPE>::setup(RTYPE *inicond, RTYPE *rhs)
{
    std::cout << "--| IterSolvers: setting solver up..." << std::endl;

    // Setup host
    PUSH_RANGE("IterSolvers::setup", 0);
    x0 = inicond;
    b = rhs;
    POP_RANGE

#ifdef USE_GPU
    // Setup device
    PUSH_RANGE("IterSolvers::setup -> device", 0);
    CUDA_CHECK(cudaMemcpy(d_x0, x0, arrSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, arrSize * sizeof(float), cudaMemcpyHostToDevice));
    POP_RANGE
#endif

    flag_setup = true;
    std::cout << "--| IterSolvers: solvers set up!" << std::endl;
}

template class IterSolvers<uint32_t, float>;
template class IterSolvers<uint64_t, float>;
template class IterSolvers<uint32_t, double>;
template class IterSolvers<uint64_t, double>;
#ifdef USE_GPU
template class IterSolvers<uint32_t, __nv_bfloat16>;
template class IterSolvers<uint64_t, __nv_bfloat16>;
#endif