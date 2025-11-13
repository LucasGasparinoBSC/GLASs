#ifdef USE_GPU
#include "unittKernel.cuh"
#endif
#include "ConjugateGradient.hpp"

// Diagonal precond
void host_diagPrecond(const double *d, const double *r_in, double *r_out, const uint32_t N)
{
    for (uint32_t i = 0; i < N; i++)
    {
        r_out[i] = r_in[i] / d[i];
    }
}

// Test MatVec for CPU
void host_tridiagdiagMatVec(const double *c, const double *d, const double *e, const double *x_in, double *x_out, const uint32_t N)
{
    // entries 1 to N-2 are generic: c[i-1]*x[i-1] + d[i]*x[i] + e[i]*x[i+1]
    for (uint32_t i = 1; i < N - 1; i++)
    {
        x_out[i] = c[i - 1] * x_in[i - 1] + d[i] * x_in[i] + e[i] * x_in[i + 1];
    }

    // First row
    x_out[0] = d[0] * x_in[0] + e[0] * x_in[1];

    // Last row
    x_out[N - 1] = c[N - 1] * x_in[N - 2] + d[N - 1] * x_in[N - 1];
}

// Comms for CPU matvec
void halo_exchange(const int prank, const int nranks, const uint32_t nLocal, double *c, double *e, const double *x_in, double *x_out, const MPI_Comm comm)
{
    // ... nl-1 --|-- 0 ... nl-1 --|-- 0 ...
    //      p-1  exc      p       exc  p+1

    // Generic exchanges: both left and right interfaces
    double *left_data;
    double *right_data;
    left_data = (double *)malloc(sizeof(double));
    right_data = (double *)malloc(sizeof(double));

    // Ring algo for exchange, determine left and right ranks using modulo
    int left_rank = (prank - 1 + nranks) % nranks;
    int right_rank = (prank + 1) % nranks;

    // Ring left2right xin[0]
    MPI_Sendrecv(&x_in[0], 1, MPI_DOUBLE, left_rank, 0,
                 right_data, 1, MPI_DOUBLE, right_rank, 0,
                 comm, MPI_STATUS_IGNORE);

    // Ring right2left xin[nLocal-1]
    MPI_Sendrecv(&x_in[nLocal - 1], 1, MPI_DOUBLE, right_rank, 0,
                 left_data, 1, MPI_DOUBLE, left_rank, 0,
                 comm, MPI_STATUS_IGNORE);

    if (prank == 0)
    {
        x_out[nLocal - 1] += e[nLocal - 1] * right_data[0];
    }
    else if (prank == nranks - 1)
    {
        x_out[0] += e[0] * left_data[0];
    }
    else
    {
        x_out[0] += c[0] * left_data[0];
        x_out[nLocal - 1] += e[nLocal - 1] * right_data[0];
    }

    free(left_data);
    free(right_data);
}

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
    uint32_t arrSize = 2000;
#else
    uint32_t arrSize = 2000;
#endif
    uint32_t mIters = 50;
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

    // Create local Tridiagonal matrix (cl,dl,el)
    // NOTE: to avoid complications with allocation, i let the
    //       computation handle the extra entries in cl and el;
    double *cl = (double *)calloc(arrSize_loc, sizeof(double)); // under-diagonal
    double *dl = (double *)calloc(arrSize_loc, sizeof(double)); // main diagonal
    double *el = (double *)calloc(arrSize_loc, sizeof(double)); // upper-diagonal

    // Matrix structure is 1 - 4 - 1
    for (uint32_t i = 0; i < arrSize_loc; i++)
    {
        cl[i] = 1.0f;
        dl[i] = 4.0f;
        el[i] = 1.0f;
    }

#ifdef USE_GPU
    // Send matrix to GPU
    double *d_cl, *d_dl, *d_el;
    CUDA_CHECK(cudaMalloc((void **)&d_cl, arrSize_loc * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cl, cl, arrSize_loc * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_dl, arrSize_loc * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_dl, dl, arrSize_loc * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_el, arrSize_loc * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_el, el, arrSize_loc * sizeof(double), cudaMemcpyHostToDevice));
#endif

    // Create x0 and rhs
    double *x0 = (double *)calloc(arrSize_loc, sizeof(double));
    double *b = (double *)calloc(arrSize_loc, sizeof(double));
    for (uint32_t i = 0; i < arrSize_loc; i++)
    {
        x0[i] = static_cast<double>(1);
        b[i] = 10.0f;
    }
#ifdef USE_GPU
    // Send x0 and b to GPU
    double *d_x0, *d_b;
    CUDA_CHECK(cudaMalloc((void **)&d_x0, arrSize_loc * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_x0, x0, arrSize_loc * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_b, arrSize_loc * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_b, b, arrSize_loc * sizeof(double), cudaMemcpyHostToDevice));
#endif

    // Create solver
    ConjugateGradient<uint32_t, double> solver(client_comm, arrSize_loc, mIters, tol);

    // Setup solver
    solver.setup(x0, b);

    // Run fpcg solver
    auto MatVec = [=](const double *x_in, double *x_out)
    {
        host_tridiagdiagMatVec(cl, dl, el, x_in, x_out, arrSize_loc);
        if (client_size > 1)
        {
            halo_exchange(client_rank, client_size, arrSize_loc, cl, el, x_in, x_out, client_comm);
        }
    };

    auto Precond = [=](const double *r_in, double *r_out)
    {
        host_diagPrecond(dl, r_in, r_out, arrSize_loc);
    };

    solver.fpcgSolver(MatVec, Precond);

    // Get solution
    double *x_sol = (double *)calloc(arrSize_loc, sizeof(double));
    solver.getSolution(x_sol);

    // Rank 0 writes to a .dat file, using MPI_Get to retrieve data from other ranks
    // Create a MPI window
    MPI_Win win;
    MPI_CHECK(MPI_Win_create(x_sol, arrSize_loc * sizeof(double), sizeof(double), MPI_INFO_NULL, client_comm, &win));

    if (client_rank == 0)
    {
        // Open file
        FILE *fout = fopen("unitt_32_fpcg_solution.dat", "w");
        // Write own data
        for (uint32_t i = 0; i < arrSize_loc; i++)
        {
            fprintf(fout, "%f\n", x_sol[i]);
        }
        // Get data from other ranks using MPI_Get
        for (int r = 1; r < client_size; r++)
        {
            uint32_t r_size = arrSize_perRank[r];
            double *r_buf = (double *)calloc(r_size, sizeof(double));
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, r, 0, win));
            MPI_CHECK(MPI_Get(r_buf, r_size, MPI_DOUBLE, r, 0, r_size, MPI_DOUBLE, win));
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
