#ifdef USE_GPU
#include "unittKernel.cuh"
#endif
#include "ConjugateGradient.hpp"
#include "halo_ops.hpp"
#include "host_ops.hpp"

int main()
{

    // Initialize MPI environment
    MPI_Init(NULL, NULL);

    // Create a commutils object
    MPI_Comm wcomm = MPI_COMM_WORLD;
    Comm_Utils client_commObj(wcomm);

    // Define problem size
    const uint32_t N = 2000; // Global problem size (glob nrows)

    // Set local sizes
    uint32_t N_loc = 0;
    uint32_t *arrSize_perRank = (uint32_t *)calloc(client_commObj.getLibSize(), sizeof(uint32_t));
    setLocalSizes(N, client_commObj.getLibRank(), client_commObj.getLibSize(), N_loc, arrSize_perRank);
    if (client_commObj.getLibRank() == 0)
    {
        for (int r = 0; r < client_commObj.getLibSize(); r++)
        {
            printf("Rank %d: Local size = %u\n", r, arrSize_perRank[r]);
        }
    }

    // Generate tridiagonal matrix
    double *cl = (double *)calloc(N_loc, sizeof(double));
    double *dl = (double *)calloc(N_loc, sizeof(double));
    double *el = (double *)calloc(N_loc, sizeof(double));
    generate_matrix<uint32_t, double>(N_loc, cl, dl, el);

    // Generate initial condition
    double *x0 = (double *)calloc(N_loc, sizeof(double));
    generate_inicond<uint32_t, double>(N_loc, x0);

    // Generate RHS vector
    double *b = (double *)calloc(N_loc, sizeof(double));
    for (uint32_t i = 0; i < N_loc; i++)
    {
        uint32_t val = client_commObj.getLibRank() * N_loc + i + 1;
        // val = static_cast<uint32_t>(1);
        b[i] = static_cast<double>(val);
    }

#ifdef USE_GPU
    // Generate device vars
    double *d_cl, *d_dl, *d_el;
    CUDA_CHECK(cudaMalloc(&d_cl, N_loc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dl, N_loc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_el, N_loc * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cl, cl, N_loc * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dl, dl, N_loc * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_el, el, N_loc * sizeof(double), cudaMemcpyHostToDevice));
    double *d_x0, *d_b;
    CUDA_CHECK(cudaMalloc(&d_x0, N_loc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, N_loc * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x0, x0, N_loc * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, N_loc * sizeof(double), cudaMemcpyHostToDevice));
#endif

    // Create ConjugateGradient object
    uint32_t maxIters = 100;
    double tol = static_cast<double>(1e-7);
    MPI_Comm client_comm = client_commObj.getLibComm();
    ConjugateGradient<uint32_t, double> solver(client_comm, N_loc, maxIters, tol);
    int nranks = client_commObj.getLibSize();

    // Buffers for comms
    double *ldata;
    double *rdata;
#ifdef USE_GPU
    CUDA_CHECK(cudaMalloc((void **)&ldata, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&rdata, sizeof(double)));
    solver.setup(d_x0, d_b);
    double startSample = MPI_Wtime();
    for (int run = 0; run < 200; run++)
    {
        runSolver_64(client_commObj, N_loc, d_cl, d_dl, d_el, ldata, rdata, solver);
    }
    double endSample = MPI_Wtime();
    double avgSampleTime_p = (endSample - startSample) / 200.0;
    double avgSampleTime = 0.0;
    if (client_commObj.isParallel && client_commObj.getLibSize() > 1)
    {
        MPI_CHECK(MPI_Reduce(&avgSampleTime_p, &avgSampleTime, 1, MPI_DOUBLE, MPI_MAX, 0, client_comm));
    }
    else
    {
        avgSampleTime = avgSampleTime_p;
    }
    if (client_commObj.getLibRank() == 0)
    {
        printf("Average time per FPCG solve over 200 runs: %f (ms)\n", avgSampleTime * 1000.0);
    }
    cudaFree(ldata);
    cudaFree(rdata);
#else
    ldata = (double *)calloc(1, sizeof(double));
    rdata = (double *)calloc(1, sizeof(double));
    // Setup solver
    solver.setup(x0, b);

    // Solve using Flexible PCG
    auto matvec = [=](const double *x_in, double *x_out)
    {
        // call tridiagonal matvec
        host_tridiagMatVec<uint32_t, double>(cl, dl, el, x_in, x_out, N_loc);
        // If parallel, do halo exchange
        if (client_commObj.isParallel && nranks > 1)
        {
            halo_exchange<uint32_t, double>(client_commObj, ldata, rdata, N_loc, cl, el, x_in, x_out);
        }
    };

    auto precond = [=](const double *r_in, double *r_out)
    {
        // Diagonal preconditioner
        host_diagPrecond<uint32_t, double>(dl, r_in, r_out, N_loc);
    };

    double startSample = MPI_Wtime();
    for (int run = 0; run < 500; run++)
    {
        solver.fpcgSolver(matvec, precond);
    }
    double endSample = MPI_Wtime();
    double avgSampleTime_p = (endSample - startSample) / 500.0;
    double avgSampleTime = 0.0;
    if (client_commObj.isParallel && client_commObj.getLibSize() > 1)
    {
        MPI_CHECK(MPI_Reduce(&avgSampleTime_p, &avgSampleTime, 1, MPI_DOUBLE, MPI_MAX, 0, client_comm));
    }
    else
    {
        avgSampleTime = avgSampleTime_p;
    }
    if (client_commObj.getLibRank() == 0)
    {
        printf("Average time per FPCG solve over 5000 runs: %f (ms)\n", avgSampleTime * 1000.0);
    }
    free(ldata);
    free(rdata);
#endif

    // Retrieve solution
    double *x_sol = (double *)calloc(N_loc, sizeof(double));
#ifdef USE_GPU
    double *d_x_sol;
    CUDA_CHECK(cudaMalloc(&d_x_sol, N_loc * sizeof(double)));
    solver.getSolution(d_x_sol);
    CUDA_CHECK(cudaMemcpy(x_sol, d_x_sol, N_loc * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x_sol));
#else
    solver.getSolution(x_sol);
#endif

    // Rank 0 writes to a .dat file, using MPI_Get to retrieve data from other ranks
    // Create a MPI window
    MPI_Win win;
    MPI_CHECK(MPI_Win_create(x_sol, N_loc * sizeof(double), sizeof(double), MPI_INFO_NULL, client_comm, &win));

    if (client_commObj.getLibRank() == 0)
    {
        // Open file
        FILE *fout = fopen("unitt_64_fpcg_solution.dat", "w");
        // Write own data
        for (uint32_t i = 0; i < N_loc; i++)
        {
            fprintf(fout, "%d, %f\n", i, x_sol[i]);
        }
        // Get data from other ranks using MPI_Get
        for (int r = 1; r < client_commObj.getLibSize(); r++)
        {
            uint32_t r_size = arrSize_perRank[r];
            double *r_buf = (double *)calloc(r_size, sizeof(double));
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, r, 0, win));
            MPI_CHECK(MPI_Get(r_buf, r_size, MPI_DOUBLE, r, 0, r_size, MPI_DOUBLE, win));
            MPI_CHECK(MPI_Win_unlock(r, win));
            for (uint32_t i = 0; i < r_size; i++)
            {
                fprintf(fout, "%d, %f\n", i + r * N_loc, r_buf[i]);
            }
            free(r_buf);
        }
        fclose(fout);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
