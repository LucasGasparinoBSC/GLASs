#include "ConjugateGradient_wrapper.hpp"

using CG_u32_f = ConjugateGradient<uint32_t, float>;
using CG_u64_f = ConjugateGradient<uint32_t, float>;
using CG_u32_d = ConjugateGradient<uint32_t, double>;
using CG_u46_d = ConjugateGradient<uint64_t, double>;

// ---- uint32_t / float ----

// Constructor
void *cg_create_u32_f(uint32_t arrSize, uint32_t maxIters, double tol)
{
    return new CG_u32_f(arrSize, maxIters, tol);
}

// Destructor
void cg_destroy_u32_f(void *solver)
{
    delete static_cast<CG_u32_f*>(solver);
}