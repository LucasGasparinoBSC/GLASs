/*
#include "ConjugateGradient_wrapper.hpp"

using CG_u32_f = ConjugateGradient<uint32_t, float>;
using CG_u64_f = ConjugateGradient<uint64_t, float>;
using CG_u32_d = ConjugateGradient<uint32_t, double>;
using CG_u64_d = ConjugateGradient<uint64_t, double>;

// ---- uint32_t / float ----

// Constructor
void *cg_create_u32_f(uint32_t arrSize, uint32_t maxIters, double tol)
{
    return new CG_u32_f(arrSize, maxIters, tol);
}

// Parallel constructor
void *cg_create_u32_pf(int fcomm, uint32_t arrSize, uint32_t maxIters, double tol)
{
    MPI_Comm comm = MPI_Comm_f2c(fcomm);
    return new CG_u32_f(comm, arrSize, maxIters, tol);
}

// Destructor
void cg_destroy_u32_f(void *solver)
{
    delete static_cast<CG_u32_f*>(solver);
}

// Setup method
void cg_setup_u32_f(void *solver, const float *inicond, const float *rhs)
{
    auto* cg = static_cast<CG_u32_f*>(solver);
    cg->setup(const_cast<float*>(inicond), const_cast<float*>(rhs));
}

// Call CG solver
void cg_solve_u32_f(void *solver, matvec_f32 matvec, halocomm_f32 halocomm, void* user_data)
{
    auto* cg = static_cast<CG_u32_f*>(solver);
    typename CG_u32_f::MatVecOp cpp_matvec = [matvec, user_data](const float *x_in, float *x_out) {
        matvec(x_in, x_out, user_data);
    };
    typename CG_u32_f::HaloOp cpp_halocomm = [halocomm, user_data](float *x_inout) {
        halocomm(x_inout, user_data);
    };
    cg->cgSolver(cpp_matvec, cpp_halocomm);
}

// Call FPCG solver
void fpcg_solve_u32_f(void *solver, matvec_f32 matvec, precond_f32 precond, void* user_data)
{
    auto* cg = static_cast<CG_u32_f*>(solver);
    typename CG_u32_f::MatVecOp cpp_matvec = [matvec, user_data](const float *x_in, float *x_out) {
        matvec(x_in, x_out, user_data);
    };
    typename CG_u32_f::PrecondOp cpp_precond = [precond, user_data](const float *r_in, float *z_out) {
        precond(r_in, z_out, user_data);
    };
    cg->fpcgSolver(cpp_matvec, cpp_precond);
}

// Get solution
void cg_get_solution_u32_f(void *solver, float* sol) {
    auto* cg = static_cast<CG_u32_f*>(solver);
    cg->getSolution(sol);
}

// ---- uint32_t / double ----

// Constructor
void *cg_create_u32_d(uint32_t arrSize, uint32_t maxIters, double tol)
{
    return new CG_u32_d(arrSize, maxIters, tol);
}

// Parallel constructor
void *cg_create_u32_pd(int fcomm, uint32_t arrSize, uint32_t maxIters, double tol)
{
    MPI_Comm comm = MPI_Comm_f2c(fcomm);
    return new CG_u32_d(comm, arrSize, maxIters, tol);
}

// Destructor
void cg_destroy_u32_d(void *solver)
{
    delete static_cast<CG_u32_d*>(solver);
}

// Setup method
void cg_setup_u32_d(void *solver, const double *inicond, const double *rhs)
{
    auto* cg = static_cast<CG_u32_d*>(solver);
    cg->setup(const_cast<double*>(inicond), const_cast<double*>(rhs));
}

// Call CG solver
void cg_solve_u32_d(void *solver, matvec_f64 matvec, halocomm_f64 halocomm, void* user_data)
{
    auto* cg = static_cast<CG_u32_d*>(solver);
    typename CG_u32_d::MatVecOp cpp_matvec = [matvec, user_data](const double *x_in, double *x_out) {
        matvec(x_in, x_out, user_data);
    };
    typename CG_u32_d::HaloOp cpp_halocomm = [halocomm, user_data](double *x_inout) {
        halocomm(x_inout, user_data);
    };
    cg->cgSolver(cpp_matvec, cpp_halocomm);
}

// Call FPCG solver
void fpcg_solve_u32_d(void *solver, matvec_f64 matvec, precond_f64 precond, void* user_data)
{
    auto* cg = static_cast<CG_u32_d*>(solver);
    typename CG_u32_d::MatVecOp cpp_matvec = [matvec, user_data](const double *x_in, double *x_out) {
        matvec(x_in, x_out, user_data);
    };
    typename CG_u32_d::PrecondOp cpp_precond = [precond, user_data](const double *r_in, double *z_out) {
        precond(r_in, z_out, user_data);
    };
    cg->fpcgSolver(cpp_matvec, cpp_precond);
}

// Get solution
void cg_get_solution_u32_d(void *solver, double* sol) {
    auto* cg = static_cast<CG_u32_d*>(solver);
    cg->getSolution(sol);
}
*/