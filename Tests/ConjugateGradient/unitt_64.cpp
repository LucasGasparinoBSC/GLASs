#ifdef USE_GPU
#include "unittKernel.cuh"
#endif
#include "ConjugateGradient.hpp"

// Test MatVec for CPU DMV
void host_diagMatVec_64(const double *A, const double *x_in, double *x_out, uint32_t N)
{
    for (uint32_t i = 0; i < N; i++)
    {
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

int main()
{

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
    uint32_t arrSize = 20;
#endif
    uint32_t mIters = 5;
    double tol = 1e-5;

    // Determine local sizes
    double ratio = static_cast<double>(arrSize) / static_cast<double>(client_size);
    uint32_t arrSize_loc;
    uint32_t arrSize_perRank[client_size];
    {
        uint32_t aux = static_cast<uint32_t>(ratio);
        if ((ratio - static_cast<double>(aux)) > 0.5)
            aux++;
        for (int i = 0; i < client_size; i++)
            arrSize_perRank[i] = aux;
        arrSize_perRank[client_size - 1] = arrSize - (client_size - 1) * aux;
        arrSize_loc = arrSize_perRank[client_rank];
    }

    if (world_rank == 0)
        printf("Running test with %d MPI ranks, array size %d per rank\n", client_size, arrSize_loc);

    // Generate data for test
    double *x0 = (double *)calloc(arrSize_loc, sizeof(double));
    double *b = (double *)calloc(arrSize_loc, sizeof(double));
    for (uint32_t i = 0; i < arrSize_loc; i++)
    {
        x0[i] = 0.001f;
        b[i] = static_cast<double>(client_rank * arrSize_loc + i + 1); // b = [1, 2, 3, ..., arrSize]
    }

#ifdef USE_GPU
    double *d_x0 = nullptr;
    double *d_b = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_x0, arrSize_loc * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, arrSize_loc * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x0, x0, arrSize_loc * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, arrSize_loc * sizeof(double), cudaMemcpyHostToDevice));
#endif

    // Define a simple diagonal matrix
    double *A = (double *)calloc(arrSize_loc, sizeof(double));
    for (uint32_t i = 0; i < arrSize_loc; i++)
    {
        A[i] = static_cast<double>(4);
    }

    // Library interaction: plan and solve
    ConjugateGradient<uint32_t, double> Solver(client_comm, arrSize_loc, mIters, tol);

#ifdef USE_GPU
    Solver.setup(d_x0, d_b);
#else
    Solver.setup(x0, b);
#endif

// Call the solver
#ifdef USE_GPU
    // CUDA kernel version
    double *d_A;
    cudaMalloc((void **)&d_A, arrSize_loc * sizeof(double));
    cudaMemcpy(d_A, A, arrSize_loc * sizeof(double), cudaMemcpyHostToDevice);
    runSolver_64(arrSize_loc, d_A, Solver);
#else
    auto MatVec = [=](const double *x_in, double *x_out)
    {
        host_diagMatVec_64(A, x_in, x_out, arrSize_loc);
    };

    auto HaloComm = [=](double *x_inout) {
        // No operation for this test
    };
    Solver.cgSolver(MatVec, HaloComm);
#endif

    // Get the solution
    double *result = (double *)calloc(arrSize_loc, sizeof(double));
#ifdef USE_GPU
    double *d_result;
    CUDA_CHECK(cudaMalloc((void **)&d_result, arrSize_loc * sizeof(double)));
    Solver.getSolution(d_result);
    CUDA_CHECK(cudaMemcpy(result, d_result, arrSize_loc * sizeof(double), cudaMemcpyDeviceToHost));
#else
    Solver.getSolution(result);
#endif

    // Check
    for (uint32_t i = 0; i < arrSize_loc; i++)
    {
        double exact = b[i] / 4.0f;
        if (std::abs(result[i] - exact) > 1e-5)
        {
            printf("Rank %d: Error at entry %d, got %f, expected %f\n", client_rank, i, result[i], exact);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}