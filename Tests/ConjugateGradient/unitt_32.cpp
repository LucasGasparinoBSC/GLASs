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

    // World communicator data
    int world_rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    // Client communicator (by duplication)
    MPI_Comm client_comm;
    MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &client_comm));
    int client_rank, client_size;
    MPI_CHECK(MPI_Comm_rank(client_comm, &client_rank));
    MPI_CHECK(MPI_Comm_size(client_comm, &client_size));

    // Problem definitions
    #ifdef USE_GPU
        uint32_t arrSize = 20000000;
    #else
        uint32_t arrSize = 20000000;
    #endif
    uint32_t mIters = 5;
    double tol = 1e-5;

    // Determine local sizes
    double ratio = static_cast<double>(arrSize) / static_cast<double>(client_size);
    uint32_t arrSize_loc;
    uint32_t arrSize_perRank[client_size];
    {
        uint32_t aux = static_cast<uint32_t>(ratio);
        if ( (ratio - static_cast<double>(aux)) > 0.5 ) aux++;
        for (int i = 0; i < client_size; i++) arrSize_perRank[i] = aux;
        arrSize_perRank[client_size-1] = arrSize - (client_size-1)*aux;
        arrSize_loc = arrSize_perRank[client_rank];
    }

    if (world_rank == 0) printf("Running test with %d MPI ranks, array size %d per rank\n", client_size, arrSize_loc);

    // Generate data for test
    float* x0 = (float*)calloc(arrSize_loc, sizeof(float));
    float* b  = (float*)calloc(arrSize_loc, sizeof(float));
    for (uint32_t i = 0; i < arrSize_loc; i++) {
        x0[i] = 0.001f;
        b[i]  = static_cast<float>( client_rank * arrSize_loc + i + 1 ); // b = [1, 2, 3, ..., arrSize]
    }

    #ifdef USE_GPU
        float* d_x0 = nullptr;
        float* d_b  = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_x0, arrSize_loc * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_b,  arrSize_loc * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x0, x0, arrSize_loc * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b,  b,  arrSize_loc * sizeof(float), cudaMemcpyHostToDevice));
    #endif

    // Define a simple diagonal matrix
    float *A = (float *)calloc(arrSize_loc, sizeof(float));
    for (uint32_t i = 0; i < arrSize_loc; i++)
    {
        A[i] = static_cast<float>(4);
    }

    // Library interaction: plan and solve
    ConjugateGradient<uint32_t, float> Solver(client_comm, arrSize_loc, mIters, tol);

    #ifdef USE_GPU
        Solver.setup(d_x0, d_b);
    #else
        Solver.setup(x0, b);
    #endif

    // Call the solver
    #ifdef USE_GPU
    // CUDA kernel version
        float* d_A;
        cudaMalloc((void**)&d_A, arrSize_loc * sizeof(float));
        cudaMemcpy(d_A, A, arrSize_loc * sizeof(float), cudaMemcpyHostToDevice);
        runSolver_32(arrSize_loc, d_A, Solver);
        runSolver_32(arrSize_loc, d_A, Solver);
        runSolver_32(arrSize_loc, d_A, Solver);
        runSolver_32(arrSize_loc, d_A, Solver);
        runSolver_32(arrSize_loc, d_A, Solver);
        runSolver_32(arrSize_loc, d_A, Solver);
        runSolver_32(arrSize_loc, d_A, Solver);
        runSolver_32(arrSize_loc, d_A, Solver);
        runSolver_32(arrSize_loc, d_A, Solver);
        runSolver_32(arrSize_loc, d_A, Solver);
    #else
        auto MatVec = [=] (const float* x_in, float* x_out) {
            host_diagMatVec_32(A, x_in, x_out, arrSize_loc);
        };
        for (int irun = 0; irun < 10; irun++) {
            Solver.cgSolver(MatVec);
        }
    #endif

    // Get the solution
    float* result = (float*)calloc(arrSize_loc, sizeof(float));
    #ifdef USE_GPU
        float* d_result;
        CUDA_CHECK(cudaMalloc((void**)&d_result, arrSize_loc * sizeof(float)));
        Solver.getSolution(d_result);
        CUDA_CHECK(cudaMemcpy(result, d_result, arrSize_loc * sizeof(float), cudaMemcpyDeviceToHost));
    #else
        Solver.getSolution(result);
    #endif

    // Check
    for (uint32_t i = 0; i < arrSize_loc; i++) {
        float exact = b[i] / 4.0f;
        if ( std::abs(result[i] - exact) > 1e-5 ) {
            printf("Rank %d: Error at entry %d, got %f, expected %f\n", client_rank, i, result[i], exact);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
