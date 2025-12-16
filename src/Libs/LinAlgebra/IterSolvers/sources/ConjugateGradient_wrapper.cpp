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
void *cg_create_u32_pf(MPI_Comm comm, uint32_t arrSize, uint32_t maxIters, double tol)
{
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

// Call solver
void cg_solve_u32_f(void *solver, matvec_f32 matvec, void* user_data)
{
    auto* cg = static_cast<CG_u32_f*>(solver);
    typename CG_u32_f::MatVecOp cpp_matvec = [matvec, user_data](const float *x_in, float *x_out) {
        matvec(x_in, x_out, user_data);
    };
    cg->cgSolver(cpp_matvec);
}

// Get solution
void cg_get_solution_u32_f(void *solver, float* sol) {
    auto* cg = static_cast<CG_u32_f*>(solver);
    cg->getSolution(sol);
}
