#ifndef CONJUGATEGRADIENT_WRAPPER_H
#define CONJUGATEGRADIENT_WRAPPER_H

#pragma once

#include "ConjugateGradient.hpp"

#ifdef __cplusplus
extern "C" {
#endif

    // ---- uint32_t / float ----
    void *cg_create_u32_f(uint32_t arrSize, uint32_t maxIters, double tol);
    void cg_destroy_u32_f(void *solver);
    void cg_solve_u32_f(void *solver);
    float *cg_get_solution_u32_f(void *solver);

#ifdef __cplusplus
}
#endif

#endif