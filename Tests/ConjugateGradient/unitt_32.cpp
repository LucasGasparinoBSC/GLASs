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
    uint32_t arrSize = 50;
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
        if ( (ratio - static_cast<double>(aux)) > 0.5 ) aux++;
        for (int i = 0; i < client_size; i++) arrSize_perRank[i] = aux;
        arrSize_perRank[client_size-1] = arrSize - (client_size-1)*aux;
        arrSize_loc = arrSize_perRank[client_rank];
    }

    // Instantiate and setup the solver
    float* x0 = new float[arrSize_loc];
    float* b  = new float[arrSize_loc];
    for (uint32_t i = 0; i < arrSize_loc; i++) {
        x0[i] = 0.001f;
        b[i]  = static_cast<float>( client_rank * arrSize_loc + i + 1 ); // b = [1, 2, 3, ..., arrSize]
        printf("Rank %d: b[%u] = %f\n", client_rank, i, b[i]);
    }

    // Testing: define a simple diagonal matrix
    float *A = (float *)calloc(arrSize_loc, sizeof(float));
    for (uint32_t i = 0; i < arrSize_loc; i++)
    {
        A[i] = static_cast<float>(4);
    }

    // Library interaction: plan and solve

   // Finalize MPI environment
    MPI_Finalize();
    return 0;
}