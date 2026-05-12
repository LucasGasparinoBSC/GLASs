#include "ConjugateGradient.hpp"
#include "HostSide.hpp"

int main() {

    // Initialize MPI environment
    MPI_Init(NULL, NULL);

    // Create a commutils object
    MPI_Comm wcomm = MPI_COMM_WORLD;
    Comm_Utils client_commObj(wcomm);

    // Define problem size
    const uint32_t N = 10; // Global problem size (glob nrows)
    const uint32_t Nwork = N-1; // Remove 1 node to account for periodicity between 0 and N-1

    uint32_t N_loc = 0;
    uint32_t* arrSize_perRank = (uint32_t *)calloc(client_commObj.getLibSize(), sizeof(uint32_t));
    HostSide<uint32_t, float>::setLocalSizes(Nwork, client_commObj.getLibRank(), client_commObj.getLibSize(), N_loc, arrSize_perRank);

    // Generate tridiagonal matrix
    float *cl = (float *)calloc(N_loc, sizeof(float));
    float *dl = (float *)calloc(N_loc, sizeof(float));
    float *el = (float *)calloc(N_loc, sizeof(float));
    HostSide<uint32_t, float>::generate_matrix(N_loc, cl, dl, el);

    // Generate initial condition
    float *x0 = (float *)calloc(N_loc, sizeof(float));
    HostSide<uint32_t, float>::generate_inicond(N_loc, x0);

    // Generate RHS vector
    float *b = (float *)calloc(N_loc, sizeof(float));
    HostSide<uint32_t, float>::generate_rhs(N_loc, b, client_commObj.getLibRank());

    #ifdef USE_GPU
        // Generate device vars
        float *d_cl, *d_dl, *d_el, *d_x0, *d_b;
        d_cl = DeviceMemory<uint32_t, float>::deviceCalloc(N_loc);
        d_dl = DeviceMemory<uint32_t, float>::deviceCalloc(N_loc);
        d_el = DeviceMemory<uint32_t, float>::deviceCalloc(N_loc);
        d_x0 = DeviceMemory<uint32_t, float>::deviceCalloc(N_loc);
        d_b = DeviceMemory<uint32_t, float>::deviceCalloc(N_loc);
        DeviceMemory<uint32_t, float>::copyHostToDevice(N_loc, cl, d_cl);
        DeviceMemory<uint32_t, float>::copyHostToDevice(N_loc, dl, d_dl);
        DeviceMemory<uint32_t, float>::copyHostToDevice(N_loc, el, d_el);
        DeviceMemory<uint32_t, float>::copyHostToDevice(N_loc, x0, d_x0);
        DeviceMemory<uint32_t, float>::copyHostToDevice(N_loc, b, d_b);
    #endif

    // Plan/setup the solver
    uint32_t maxIters = 100;
    double tol = 1e-7;
    MPI_Comm client_comm = client_commObj.getLibComm();
    ConjugateGradient<uint32_t, float> solver(client_comm, N_loc, maxIters, tol);

    // Test parallel cyclic matvec:
    // 1. Compute left/right ranks
    int leftRank, rightRank;
    HostSide<uint32_t, float>::computeLeftRightRanks(client_commObj.getLibRank(), client_commObj.getLibSize(), leftRank, rightRank);

    // 2. Create MPI window for ghost data exchange
    float buffer[4]; // buffer for halo data: [left_cl, left_el, right_cl, right_el]
    Matvec<uint32_t, float>::fillBuffer(cl, el, x0, buffer, N_loc);
    MPI_Win win;
    MPI_Win_create(buffer, 4*sizeof(float), sizeof(float), MPI_INFO_NULL, client_comm, &win);

    // 3. Launch matvec with overlapped comms-compute
    float *y = (float *)calloc(N_loc, sizeof(float));
    float ghostData[4];
    Matvec<uint32_t, float>::launch_matvec(win, client_commObj.getLibRank(), client_commObj.getLibSize(), leftRank, rightRank, cl, dl, el, x0, ghostData, y, N_loc);

    MPI_Barrier(client_comm); // Ensure all ranks have completed before proceeding
    for (int ir = 0; ir < client_commObj.getLibSize(); ir++)
    {
        if (ir == client_commObj.getLibRank())
        {
            for (uint32_t i = 0; i < N_loc; i++)
            {
                std::cout << "Rank " << ir << ", y[" << i << "] = " << y[i] << std::endl;
            }
        }
        MPI_Barrier(client_comm);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
