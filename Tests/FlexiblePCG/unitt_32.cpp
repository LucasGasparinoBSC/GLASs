#ifdef USE_GPU
    #include "unittKernel.cuh"
#endif
#include "ConjugateGradient.hpp"
#include "halo_ops.hpp"
#include "host_ops.hpp"

int main() {

    // Initialize MPI environment
    MPI_Init(NULL, NULL);

    // Create a commutils object
    MPI_Comm wcomm = MPI_COMM_WORLD;
    Comm_Utils client_commObj(wcomm);

    // Define problem size
    const uint32_t N = 12000; // Global problem size (glob nrows)

    // Set local sizes
    uint32_t N_loc = 0;
    uint32_t* arrSize_perRank = (uint32_t *)calloc(client_commObj.getLibSize(), sizeof(uint32_t));
    setLocalSizes(N, client_commObj.getLibRank(), client_commObj.getLibSize(), N_loc, arrSize_perRank);
    if (client_commObj.getLibRank() == 0) {
        for (int r = 0; r < client_commObj.getLibSize(); r++) {
            printf("Rank %d: Local size = %u\n", r, arrSize_perRank[r]);
        }
    }

    // Generate tridiagonal matrix
    float *cl = (float *)calloc(N_loc, sizeof(float));
    float *dl = (float *)calloc(N_loc, sizeof(float));
    float *el = (float *)calloc(N_loc, sizeof(float));
    generate_matrix<uint32_t, float>(N_loc, cl, dl, el);

    // Generate initial condition
    float *x0 = (float *)calloc(N_loc, sizeof(float));
    generate_inicond<uint32_t, float>(N_loc, x0);

    // Generate RHS vector
    float *b = (float *)calloc(N_loc, sizeof(float));
    for (uint32_t i = 0; i < N_loc; i++) {
        uint32_t val = client_commObj.getLibRank() * N_loc + i + 1;
        b[i] = static_cast<float>(val);
    }

    // Create ConjugateGradient object
    uint32_t maxIters = 1000;
    double tol = static_cast<double>(1e-8);
    MPI_Comm client_comm = client_commObj.getLibComm();
    ConjugateGradient<uint32_t, float> solver(client_comm, N_loc, maxIters, tol);

    // Setup solver
    solver.setup(x0, b);

    // Solve using Flexible PCG
    int nranks = client_commObj.getLibSize();
    auto matvec = [=](const float *x_in, float *x_out) {
        // call tridiagonal matvec
        host_tridiagMatVec<uint32_t, float>(cl, dl, el, x_in, x_out, N_loc);
        // If parallel, do halo exchange
        if (client_commObj.isParallel && nranks > 1) {
            halo_exchange<uint32_t, float>(client_commObj, N_loc, cl, el, x_in, x_out);
        }
    };

    auto precond = [=](const float *r_in, float *r_out) {
        // Diagonal preconditioner
        host_diagPrecond<uint32_t, float>(dl, r_in, r_out, N_loc);
    };

    double startSample = MPI_Wtime();
    for (int run = 0; run < 5000; run++) {
        solver.fpcgSolver(matvec, precond);
    }
    double endSample = MPI_Wtime();
    double avgSampleTime_p = (endSample - startSample) / 5000.0;
    double avgSampleTime = 0.0;
    if (client_commObj.isParallel && client_commObj.getLibSize() > 1) {
        MPI_CHECK(MPI_Reduce(&avgSampleTime_p, &avgSampleTime, 1, MPI_DOUBLE, MPI_MAX, 0, client_comm));
    } else {
        avgSampleTime = avgSampleTime_p;
    }
    if (client_commObj.getLibRank() == 0) {
        printf("Average time per FPCG solve over 5000 runs: %f (ms)\n", avgSampleTime * 1000.0);
    }


    // Retrieve solution
    float *x_sol = (float *)calloc(N_loc, sizeof(float));
    solver.getSolution(x_sol);

    // Rank 0 writes to a .dat file, using MPI_Get to retrieve data from other ranks
    // Create a MPI window
    MPI_Win win;
    MPI_CHECK(MPI_Win_create(x_sol, N_loc * sizeof(float), sizeof(float), MPI_INFO_NULL, client_comm, &win));

    if (client_commObj.getLibRank() == 0) {
        // Open file
        FILE *fout = fopen("unitt_32_fpcg_solution.dat", "w");
        // Write own data
        for (uint32_t i = 0; i < N_loc; i++) {
            fprintf(fout, "%f\n", x_sol[i]);
        }
        // Get data from other ranks using MPI_Get
        for (int r = 1; r < client_commObj.getLibSize(); r++) {
            uint32_t r_size = arrSize_perRank[r];
            float *r_buf = (float *)calloc(r_size, sizeof(float));
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, r, 0, win));
            MPI_CHECK(MPI_Get(r_buf, r_size, MPI_FLOAT, r, 0, r_size, MPI_FLOAT, win));
            MPI_CHECK(MPI_Win_unlock(r, win));
            for (uint32_t i = 0; i < r_size; i++) {
                fprintf(fout, "%f\n", r_buf[i]);
            }
            free(r_buf);
        }
        fclose(fout);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
