#include "ConjugateGradient.hpp"

int main() {
    uint32_t narr = 100;
    uint32_t maxit = 1000;
    double tol = 1e-6;

    // Create solver instance for CG
    ConjugateGradient<uint32_t, float> cgSolver(narr, maxit, tol);
}