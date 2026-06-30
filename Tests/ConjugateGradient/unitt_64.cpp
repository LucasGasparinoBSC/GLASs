#include "HostSide.hpp"

int main() {

    // Initialize MPI environment
    MPI_Init(NULL, NULL);

    // World communicator data
    int world_rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    // Client communicator (by duplication)
    MPI_Comm client_comm;
    int client_rank, client_size;
    MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &client_comm));
    MPI_CHECK(MPI_Comm_rank(client_comm, &client_rank));
    MPI_CHECK(MPI_Comm_size(client_comm, &client_size));

    // Problem definitions
    //uint32_t arrSize = 8*1280000;
    uint32_t arrSize = (uint32_t)(20000000);
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
    double* x0 = (double*)calloc(arrSize_loc, sizeof(double));
    double* b  = (double*)calloc(arrSize_loc, sizeof(double));
    for (uint32_t i = 0; i < arrSize_loc; i++) {
        x0[i] = 0.001f;
        b[i]  = static_cast<double>( client_rank * arrSize_loc + i + 1 ); // b = [1, 2, 3, ..., arrSize]
    }

    #ifdef USE_GPU
        double* d_x0 = DeviceMemory<uint32_t,double>::deviceCalloc(arrSize_loc);
        double* d_b = DeviceMemory<uint32_t,double>::deviceCalloc(arrSize_loc);
        DeviceMemory<uint32_t,double>::copyHostToDevice(arrSize_loc, x0, d_x0);
        DeviceMemory<uint32_t,double>::copyHostToDevice(arrSize_loc, b,  d_b);
    #endif

    // Define a simple diagonal matrix
    double *A = (double *)calloc(arrSize_loc, sizeof(double));
    for (uint32_t i = 0; i < arrSize_loc; i++)
    {
        A[i] = static_cast<double>(4);
    }

    #if defined(USE_GPU)
        double* d_A = DeviceMemory<uint32_t,double>::deviceCalloc(arrSize_loc);
        DeviceMemory<uint32_t,double>::copyHostToDevice(arrSize_loc, A, d_A);
    #endif

    // Plan the solver
    ConjugateGradient<uint32_t, double> Solver(client_comm, arrSize_loc, mIters, tol);

    // Setup the solver
    #if defined(USE_GPU)
        Solver.setup(d_x0, d_b);
    #else
        Solver.setup(x0, b);
    #endif

    // Run the solver
    #if defined(USE_GPU)
	for (uint32_t irun = 0; irun < 200; irun++) {
            HostSide<uint32_t, double>::runSolver(arrSize_loc, d_A, Solver);
	}
    #else
        HostSide<uint32_t, double>::runSolver(arrSize_loc, A, Solver);
    #endif

    // Get the solution
    double* result = (double*)calloc(arrSize_loc, sizeof(double));
    #ifdef USE_GPU
        double* d_result = DeviceMemory<uint32_t,double>::deviceCalloc(arrSize_loc);
        Solver.getSolution(d_result);
        DeviceMemory<uint32_t,double>::copyDeviceToHost(arrSize_loc, d_result, result);
    #else
        Solver.getSolution(result);
    #endif

    // Check
    for (uint32_t i = 0; i < arrSize_loc; i++) {
        double exact = b[i] / 4.0f;
        if ( std::abs(result[i] - exact) > 1e-5 ) {
            printf("Rank %d: Error at entry %d, got %f, expected %f\n", client_rank, i, result[i], exact);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
