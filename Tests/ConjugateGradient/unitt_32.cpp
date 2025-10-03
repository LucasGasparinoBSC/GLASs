#ifdef USE_GPU
#include "unittKernel.cuh"
#endif
#include "ConjugateGradient.hpp"

int main() {

    // Problem definitions
    uint32_t arrSize = 100;
    uint32_t mIters = 10;
    double tol = 1e-10;

    // Instantiate the solver
    ConjugateGradient<uint32_t, float> solver(arrSize, mIters, tol);
    return 0;
}