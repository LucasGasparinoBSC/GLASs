#ifndef CONJUGATEGRADIENT_WRAPPER_H
#define CONJUGATEGRADIENT_WRAPPER_H

#pragma once

#include "ConjugateGradient.hpp"

#ifdef __cplusplus
extern "C" {
#endif

    // Matvec definitions
    typedef void (*matvec_f32)(const float *x_in, float *x_out, void *user_data);

    // ---- uint32_t / float ----
    void *cg_create_u32_f(uint32_t arrSize, uint32_t maxIters, double tol);                 // Serial version
    void *cg_create_u32_pf(MPI_Comm comm, uint32_t arrSize, uint32_t maxIters, double tol); // Parallel version
    void cg_destroy_u32_f(void *solver);
    void cg_setup_u32_f(void *solver, const float* inicond, const float* rhs);
    void cg_solve_u32_f(void *solver, matvec_f32 matvec, void* user_data);
    void cg_get_solution_u32_f(void *solver, float* sol);

#ifdef __cplusplus
}
#endif

#endif