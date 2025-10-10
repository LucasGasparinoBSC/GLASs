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

// Test MatVec for GPU-ACC DMV
#ifdef USE_GPU
void acc_diagMatVec_32(const float *A, const float *x_in, float *x_out, uint32_t N)
{
    #pragma acc parallel loop deviceptr(A, x_in, x_out)
    for (uint32_t i = 0; i < N; i++)
    {
        x_out[i] = A[i] * x_in[i];
    }
}
#endif

int main() {

    // Initialize MPI environment
    MPI_Init(NULL, NULL);
    int client_rank, client_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &client_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &client_size);

    // Problem definitions
#ifdef USE_GPU
    uint32_t arrSize = 5000000;
    #else
    uint32_t arrSize = 20000000;
    #endif
    uint32_t mIters = 5;
    double tol = 1e-5;

    // Determine local sizes
    double tmp = static_cast<double>(arrSize) / static_cast<double>(client_size);
    uint32_t arrSize_l = static_cast<uint32_t>(std::floor(tmp));
    double intPart;
    double decPart = std::modf(tmp, &intPart);

    // Instantiate and setup the solver
    float* x0 = new float[arrSize];
    float* b  = new float[arrSize];
    for (uint32_t i = 0; i < arrSize; i++) {
        x0[i] = 0.001f;
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
    CUDA_CHECK(cudaMalloc(&d_A, arrSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A, arrSize * sizeof(float), cudaMemcpyHostToDevice));
#endif

    // run the solver
#ifdef USE_GPU
    // Run the ACC version
    // Define a lambda function for the MatVec operation
    auto MatVecACC = [=] (const float* x_in, float* x_out) {
        acc_diagMatVec_32(d_A, x_in, x_out, arrSize);
    };
    solver.cgSolver(MatVecACC);

    printf("\n");

    // Run the full CUDA kernel version
    runSolver_32(arrSize, d_A, solver);
#else
    auto MatVec = [=] (const float* x_in, float* x_out) {
        host_diagMatVec_32(A, x_in, x_out, arrSize);
    };
    solver.cgSolver(MatVec);
#endif

    // Get the solution
    float* x_sol = solver.getSolution();

    // check if x is approximately correct (bi / Ai)
    for (uint32_t i = 0; i < arrSize; i++) {
        if (std::abs(x_sol[i] - b[i]/A[i]) > 1e-3) {
            printf("Error at index %u: x_sol = %f, expected = %f\n", i, x_sol[i], b[i]/A[i]);
            return -1;
        }
    }
    printf("Test passed: solution is approximately correct.\n");

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}