#ifdef USE_GPU
#include "unittKernel.cuh"
#endif
#include "ConjugateGradient.hpp"

// Test MatVec for CPU DMV
void host_diagMatVec_32(const float* A, const float* x_in, float* x_out, uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        x_out[i] = A[i] * x_in[i];
    }
}

int main() {

    // Problem definitions
    uint32_t arrSize = 100;
    uint32_t mIters = 10;
    double tol = 1e-10;

    // Instantiate and setup the solver
    float* x0 = new float[arrSize];
    float* b  = new float[arrSize];
    for (uint32_t i = 0; i < arrSize; i++) {
        x0[i] = 0.0f;
        b[i]  = static_cast<float>(i+1);
    }
    ConjugateGradient<uint32_t, float> solver(arrSize, mIters, tol);
    solver.setup(x0, b);

    // Testing: define a simple diagonal matrix
    float* A = (float*)calloc(arrSize, sizeof(float));
    for (uint32_t i = 0; i < arrSize; i++) {
        A[i] = static_cast<float>(4);
    }
#ifdef USE_GPU
    // If testing on GPUs, copy matrix to device
    float* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, arrSize * arrSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A, arrSize * arrSize * sizeof(float), cudaMemcpyHostToDevice));
#endif

    // run the solver
#ifdef USE_GPU
    runSolver_32(arrSize, d_A, solver);
#else
    auto MatVec = [=] __host__ (const float* x_in, float* x_out) {
        host_diagMatVec_32(A, x_in, x_out, arrSize);
    };
    solver.cgSolver(MatVec);
#endif
    return 0;
}