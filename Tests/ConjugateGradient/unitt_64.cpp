#ifdef USE_GPU
#include "unittKernel.cuh"
#endif
#include "ConjugateGradient.hpp"

// Test MatVec for CPU DMV
void host_diagMatVec_64(const double* A, const double* x_in, double* x_out, uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        x_out[i] = A[i] * x_in[i];
    }
}

// Test MatVec for GPU-ACC DMV
#ifdef USE_GPU
void acc_diagMatVec_64(const double *A, const double *x_in, double *x_out, uint32_t N)
{
    #pragma acc parallel loop deviceptr(A, x_in, x_out)
    for (uint32_t i = 0; i < N; i++)
    {
        x_out[i] = A[i] * x_in[i];
    }
}
#endif

int main() {

    // Problem definitions
#ifdef USE_GPU
    uint32_t arrSize = 10;
#else
    uint32_t arrSize = 200;
#endif
    uint32_t mIters = 5;
    double tol = 1e-5;

    // Instantiate and setup the solver
    double* x0 = new double[arrSize];
    double* b  = new double[arrSize];
    for (uint32_t i = 0; i < arrSize; i++) {
        x0[i] = 0.001;
        b[i]  = static_cast<double>(1);
    }
    ConjugateGradient<uint32_t, double> solver(arrSize, mIters, tol);
    solver.setup(x0, b);

    // Testing: define a simple diagonal matrix
    double* A = (double*)calloc(arrSize, sizeof(double));
    for (uint32_t i = 0; i < arrSize; i++) {
        A[i] = static_cast<double>(4);
    }
#ifdef USE_GPU
    // If testing on GPUs, copy matrix to device
    double* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, arrSize * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A, arrSize * sizeof(double), cudaMemcpyHostToDevice));
#endif

    // run the solver
#ifdef USE_GPU
    // Run the ACC version
    // Define a lambda function for the MatVec operation
    auto MatVecACC = [=] (const double* x_in, double* x_out) {
        acc_diagMatVec_64(d_A, x_in, x_out, arrSize);
    };
    solver.cgSolver(MatVecACC);

    printf("\n");

    // Run the full CUDA kernel version
    runSolver_64(arrSize, d_A, solver);
#else
    auto MatVec = [=] (const double* x_in, double* x_out) {
        host_diagMatVec_64(A, x_in, x_out, arrSize);
    };
    solver.cgSolver(MatVec);
#endif
    return 0;
}