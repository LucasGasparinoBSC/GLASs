#include "ConjugateGradient.hpp"

int main() {
    uint32_t narr = 100;
    uint32_t maxit = 1000;
    double tol = 1e-6;

    // Create solver instance for CG
    ConjugateGradient<uint32_t, float> cgSolver(narr, maxit, tol);

    // Generate a test x0 and rhs
    float* x0 = (float*)calloc(narr, sizeof(float));
    float* b = (float*)calloc(narr, sizeof(float));
    for (uint32_t i = 0; i < narr; i++) {
        x0[i] = 0.0f;
        b[i] = 1.0f; // Simple RHS
    }

    // Call the setup
    cgSolver.setup(x0, b);
}