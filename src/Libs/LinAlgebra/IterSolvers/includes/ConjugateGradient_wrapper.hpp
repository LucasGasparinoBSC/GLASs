#ifndef CONJUGATEGRADIENT_WRAPPER_H
#define CONJUGATEGRADIENT_WRAPPER_H

#pragma once

#include "ConjugateGradient.hpp"

#ifdef __cplusplus
extern "C" {
#endif

    // Matvec definitions
    typedef void (*matvec_f32)(const float *x_in, float *x_out, void *user_data);
    typedef void (*matvec_f64)(const double *x_in, double *x_out, void *user_data);
    typedef void (*precond_f32)(const float *r_in, float *z_out, void *user_data);
    typedef void (*precond_f64)(const double *r_in, double *z_out, void *user_data);
    typedef void (*halocomm_f32)(float *x_inout, void *user_data);
    typedef void (*halocomm_f64)(double *x_inout, void *user_data);

    // ---- uint32_t / float ----
    void *cg_create_u32_f(uint32_t arrSize, uint32_t maxIters, double tol);                 // Serial version
    void *cg_create_u32_pf(int fcomm, uint32_t arrSize, uint32_t maxIters, double tol); // Parallel version
    void cg_destroy_u32_f(void *solver);
    void cg_setup_u32_f(void *solver, const float* inicond, const float* rhs);
    void cg_solve_u32_f(void *solver, matvec_f32 matvec, halocomm_f32 halocomm, void *user_data);
    void fpcg_solve_u32_f(void *solver, matvec_f32 matvec, precond_f32 precond, void *user_data);
    void cg_get_solution_u32_f(void *solver, float* sol);

    // ---- uint32_t / double ----
    void *cg_create_u32_d(uint32_t arrSize, uint32_t maxIters, double tol);                 // Serial version
    void *cg_create_u32_pd(int fcomm, uint32_t arrSize, uint32_t maxIters, double tol); // Parallel version
    void cg_destroy_u32_d(void *solver);
    void cg_setup_u32_d(void *solver, const double* inicond, const double* rhs);
    void cg_solve_u32_d(void *solver, matvec_f64 matvec, halocomm_f64 halocomm, void *user_data);
    void fpcg_solve_u32_d(void *solver, matvec_f64 matvec, precond_f64 precond, void* user_data);
    void cg_get_solution_u32_d(void *solver, double* sol);

#ifdef __cplusplus
}
#endif

#endif