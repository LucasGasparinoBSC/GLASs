#include "IterSolvers.hpp"

// Empty constructor
template <typename ITYPE, typename RTYPE>
IterSolvers<ITYPE, RTYPE>::IterSolvers() {
    PUSH_RANGE("IterSolvers::Constructor(empty)", 0)
    this->flag_planned = false;
    this->flag_setup = false;
    this->arrSize = 0;
    this->maxIters = 0;
    this->iter = 0;
    this->tol = -1e-6;
    this->x_sol = nullptr;
    this->d_x_sol = nullptr;
    this->x0 = nullptr;
    this->d_x0 = nullptr;
    this->r0 = nullptr;
    this->d_r0 = nullptr;
    this->p0 = nullptr;
    this->d_p0 = nullptr;
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
    this->alpha = nullptr;
    this->d_alpha = nullptr;
    this->beta = nullptr;
    this->d_beta = nullptr;
    this->aux = nullptr;
    this->d_aux = nullptr;
    POP_RANGE
}

// Param constructor
template <typename ITYPE, typename RTYPE>
IterSolvers<ITYPE, RTYPE>::IterSolvers(ITYPE arrSize, ITYPE maxIters, double tol) : arrSize(arrSize), maxIters(maxIters), iter(0), tol(tol)
{
    std::cout << "--| IterSolvers: initializing solver" << std::endl;

    // Allocate host memory using calloc (ensures init to 0)
    PUSH_RANGE("IterSolvers::Constructor(param)", 0)
    x_sol = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    x0 = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    r0 = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    p0 = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    rk = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    zk = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    Ax = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    b = (RTYPE *)calloc(arrSize, sizeof(RTYPE));
    res0 = (RTYPE *)calloc(1, sizeof(RTYPE));
    resk = (RTYPE *)calloc(1, sizeof(RTYPE));
    alpha = (RTYPE *)calloc(1, sizeof(RTYPE));
    beta = (RTYPE *)calloc(1, sizeof(RTYPE));
    aux = (RTYPE *)calloc(1, sizeof(RTYPE));
    POP_RANGE

    // Allocate device arrays
#ifdef USE_GPU
    PUSH_RANGE("IterSolvers::Constructor -> device", 0);
    CUDA_CHECK(cudaMalloc(&d_x_sol, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_x0, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_r0, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_p0, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_rk, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_zk, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_Ax, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_b, arrSize * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_res0, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_resk, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_alpha, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_beta, 1 * sizeof(RTYPE)));
    CUDA_CHECK(cudaMalloc(&d_aux, 1 * sizeof(RTYPE)));
    POP_RANGE
#endif

    flag_planned = true;
    std::cout << "--| IterSolvers: solvers initialized!" << std::endl;
}

template class IterSolvers<uint32_t, float>;
template class IterSolvers<uint64_t, float>;
template class IterSolvers<uint32_t, double>;
template class IterSolvers<uint64_t, double>;
#ifdef USE_GPU
template class IterSolvers<uint32_t, __nv_bfloat16>;
template class IterSolvers<uint64_t, __nv_bfloat16>;
#endif