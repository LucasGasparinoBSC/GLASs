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
    const uint32_t N = 20000000; // Global problem size (glob nrows)

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
    __nv_bfloat16 *cl = (__nv_bfloat16 *)calloc(N_loc, sizeof(__nv_bfloat16));
    __nv_bfloat16 *dl = (__nv_bfloat16 *)calloc(N_loc, sizeof(__nv_bfloat16));
    __nv_bfloat16 *el = (__nv_bfloat16 *)calloc(N_loc, sizeof(__nv_bfloat16));
    generate_matrix<uint32_t, __nv_bfloat16>(N_loc, cl, dl, el);

    // Generate initial condition
    __nv_bfloat16 *x0 = (__nv_bfloat16 *)calloc(N_loc, sizeof(__nv_bfloat16));
    generate_inicond<uint32_t, __nv_bfloat16>(N_loc, x0);

    // Generate RHS vector
    __nv_bfloat16 *b = (__nv_bfloat16 *)calloc(N_loc, sizeof(__nv_bfloat16));
    for (uint32_t i = 0; i < N_loc; i++)
    {
        uint32_t val = client_commObj.getLibRank() * N_loc + i + 1;
        b[i] = static_cast<__nv_bfloat16>(val);
    }

    // Generate device vars
    __nv_bfloat16 *d_cl, *d_dl, *d_el;
    CUDA_CHECK(cudaMalloc(&d_cl, N_loc * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dl, N_loc * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_el, N_loc * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_cl, cl, N_loc * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dl, dl, N_loc * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_el, el, N_loc * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    __nv_bfloat16 *d_x0, *d_b;
    CUDA_CHECK(cudaMalloc(&d_x0, N_loc * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b, N_loc * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_x0, x0, N_loc * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, N_loc * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Create ConjugateGradient object
    uint32_t maxIters = 1000;
    double tol = static_cast<double>(1e-3);
    MPI_Comm client_comm = client_commObj.getLibComm();
    ConjugateGradient<uint32_t, __nv_bfloat16> solver(client_comm, N_loc, maxIters, tol);
    int nranks = client_commObj.getLibSize();

    solver.setup(d_x0, d_b);
    double startSample = MPI_Wtime();
    for (int run = 0; run < 50; run++)
    {
        runSolver_16(N_loc, d_cl, d_dl, d_el, solver);
    }
    double endSample = MPI_Wtime();
    double avgSampleTime_p = (endSample - startSample) / 50.0;
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
        printf("Average time per FPCG solve over 50 runs: %f (ms)\n", avgSampleTime * 1000.0);
    }

    // Retrieve solution
    __nv_bfloat16 *x_sol = (__nv_bfloat16 *)calloc(N_loc, sizeof(__nv_bfloat16));
    float *x_sol_write = (float *)calloc(N_loc, sizeof(float));
    __nv_bfloat16 *d_x_sol;
    CUDA_CHECK(cudaMalloc(&d_x_sol, N_loc * sizeof(__nv_bfloat16)));
    solver.getSolution(d_x_sol);
    CUDA_CHECK(cudaMemcpy(x_sol, d_x_sol, N_loc * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < N_loc; i++)
    {
        x_sol_write[i] = static_cast<float>(x_sol[i]);
    }
    CUDA_CHECK(cudaFree(d_x_sol));

    // Rank 0 writes to a .dat file, using MPI_Get to retrieve data from other ranks
    // Create a MPI window
    MPI_Win win;
    MPI_CHECK(MPI_Win_create(x_sol_write, N_loc * sizeof(float), sizeof(float), MPI_INFO_NULL, client_comm, &win));

    if (client_commObj.getLibRank() == 0)
    {
        // Open file
        FILE *fout = fopen("unitt_16_fpcg_solution.dat", "w");
        // Write own data
        for (uint32_t i = 0; i < N_loc; i++)
        {
            fprintf(fout, "%f\n", x_sol_write[i]);
        }
        // Get data from other ranks using MPI_Get
        for (int r = 1; r < client_commObj.getLibSize(); r++)
        {
            uint32_t r_size = arrSize_perRank[r];
            float *r_buf = (float *)calloc(r_size, sizeof(float));
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, r, 0, win));
            MPI_CHECK(MPI_Get(r_buf, r_size, MPI_FLOAT, r, 0, r_size, MPI_FLOAT, win));
            MPI_CHECK(MPI_Win_unlock(r, win));
            for (uint32_t i = 0; i < r_size; i++)
            {
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