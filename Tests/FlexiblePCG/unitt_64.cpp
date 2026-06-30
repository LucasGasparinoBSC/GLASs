#include "ConjugateGradient.hpp"
#include "HostSide.hpp"

int main() {

    // Initialize MPI environment
    MPI_Init(NULL, NULL);

    // Create a commutils object
    MPI_Comm wcomm = MPI_COMM_WORLD;
    Comm_Utils client_commObj(wcomm);

    // Define problem size
    //const uint32_t N = 1200;    // Global problem size (glob nrows)
    const uint32_t N = (1*2000) + 1;
    const uint32_t Nwork = N-1; // Remove 1 node to account for periodicity between 0 and N-1

    uint32_t N_loc = 0;
    uint32_t* arrSize_perRank = (uint32_t *)calloc(client_commObj.getLibSize(), sizeof(uint32_t));
    HostSide<uint32_t, double>::setLocalSizes(Nwork, client_commObj.getLibRank(), client_commObj.getLibSize(), N_loc, arrSize_perRank);
    uint32_t globalStart = 0;
    for (int r = 0; r < client_commObj.getLibRank(); ++r) {
        globalStart += arrSize_perRank[r];
    }

    // Generate tridiagonal matrix
    double *cl = (double *)calloc(N_loc, sizeof(double));
    double *dl = (double *)calloc(N_loc, sizeof(double));
    double *el = (double *)calloc(N_loc, sizeof(double));
    HostSide<uint32_t, double>::generate_matrix(N_loc, cl, dl, el);

    // Generate initial condition
    double *x0 = (double *)calloc(N_loc, sizeof(double));
    HostSide<uint32_t, double>::generate_inicond(N_loc, x0, globalStart);

    // Generate RHS vector
    double *b = (double *)calloc(N_loc, sizeof(double));
    HostSide<uint32_t, double>::generate_rhs(N_loc, b, globalStart);

    #ifdef USE_GPU
        // Generate device vars
        double *d_cl, *d_dl, *d_el, *d_x0, *d_b;
        d_cl = DeviceMemory<uint32_t, double>::deviceCalloc(N_loc);
        d_dl = DeviceMemory<uint32_t, double>::deviceCalloc(N_loc);
        d_el = DeviceMemory<uint32_t, double>::deviceCalloc(N_loc);
        d_x0 = DeviceMemory<uint32_t, double>::deviceCalloc(N_loc);
        d_b = DeviceMemory<uint32_t, double>::deviceCalloc(N_loc);
        DeviceMemory<uint32_t, double>::copyHostToDevice(N_loc, cl, d_cl);
        DeviceMemory<uint32_t, double>::copyHostToDevice(N_loc, dl, d_dl);
        DeviceMemory<uint32_t, double>::copyHostToDevice(N_loc, el, d_el);
        DeviceMemory<uint32_t, double>::copyHostToDevice(N_loc, x0, d_x0);
        DeviceMemory<uint32_t, double>::copyHostToDevice(N_loc, b, d_b);
    #endif

    // Plan/setup the solver
    uint32_t maxIters = 200;
    double tol = 1e-7;
    MPI_Comm client_comm = client_commObj.getLibComm();
    ConjugateGradient<uint32_t, double> solver(client_comm, N_loc, maxIters, tol);
    #if defined (USE_GPU)
        solver.setup(d_x0, d_b);
    #else
        solver.setup(x0, b);
    #endif

    // If running on GPU, get the kernel stream for use in the matvec
    #ifdef USE_GPU
        DeviceUtils::Stream_t kernelStream = solver.getKernelStream();
    #endif

    // Compute left/right ranks
    int leftRank, rightRank;
    HostSide<uint32_t, double>::computeLeftRightRanks(client_commObj.getLibRank(), client_commObj.getLibSize(), leftRank, rightRank);

    // 2. Create MPI window for ghost data exchange
    MPI_Win win;
    double *buffer;
    #if defined (USE_GPU)
        buffer = DeviceMemory<uint32_t, double>::deviceCalloc(4);
        DeviceUtils::launchKernel(AuxKernels::fillBuffer<uint32_t, double>, dim3(1), dim3(1), solver.getKernelStream(), d_cl, d_el, d_x0, buffer, N_loc);
        DeviceUtils::StreamSynchronize(solver.getKernelStream());
    #else
        buffer = (double *)calloc(4, sizeof(double)); // Buffer for halo exchange: [left_cl, left_u, right_el, right_u]
        Matvec<uint32_t, double>::fillBuffer(cl, el, x0, buffer, N_loc); // Initialize buffer with initial halo data
    #endif
    MPI_Win_create(buffer, 4 * sizeof(double), sizeof(double), MPI_INFO_NULL, client_commObj.getLibComm(), &win);

    // 3. Launch matvec with overlapped comms-compute
    double *ghostData;
    #if defined (USE_GPU)
        ghostData = DeviceMemory<uint32_t, double>::deviceCalloc(4);
    #else
        ghostData = (double *)calloc(4, sizeof(double)); // Buffer to receive halo data
    #endif

    // Call the solver with launch_matvec as the matvec operator (with appropriate lambda to capture the halo exchange)
    auto matvec_op = [&](const double* x_in, double* x_out) {
        // Update the buffer with the current x_in values for the halo nodes
        #if defined (USE_GPU)
            DeviceUtils::launchKernel(AuxKernels::fillBuffer<uint32_t, double>, dim3(1), dim3(1), solver.getKernelStream(), d_cl, d_el, x_in, buffer, N_loc);
            DeviceUtils::StreamSynchronize(solver.getKernelStream());
            Matvec<uint32_t, double>::launch_matvec(solver, win, client_commObj.getLibRank(), client_commObj.getLibSize(), leftRank, rightRank, d_cl, d_dl, d_el, x_in, ghostData, x_out, N_loc);
        #else
            Matvec<uint32_t, double>::fillBuffer(cl, el, x_in, buffer, N_loc);
            Matvec<uint32_t, double>::launch_matvec(solver, win, client_commObj.getLibRank(), client_commObj.getLibSize(), leftRank, rightRank, cl, dl, el, x_in, ghostData, x_out, N_loc);
        #endif
    };

    // Lambda for preconditioner (using simple diagonal preconditioner)
    auto precond_op = [&](const double *r_in, double *z_out)
    {
        #if defined (USE_GPU)
            DeviceUtils::Stream_t precondStream = solver.getKernelStream(); // Reuse the matvec stream for the preconditioner
            dim3 precondBlock = solver.getKernelBlock(); // Reuse the matvec block size for the preconditioner
            dim3 precondGrid = solver.getKernelGrid(); // Reuse the matvec grid size for the preconditioner
            DeviceUtils::launchKernel(AuxKernels::diag_precond<uint32_t,double>, precondGrid, precondBlock, precondStream, d_dl, r_in, z_out, N_loc);
        #else
            HostSide<uint32_t, double>::diag_precond(dl, r_in, z_out, N_loc);
        #endif
    };

    // Call FPCG solver N times
    const uint32_t num_runs = 20;
    for (uint32_t run = 0; run < num_runs; ++run)
    {
        solver.fpcgSolver(matvec_op, precond_op);
    }

    // Get the solution back to host
    double* x_sol = (double *)calloc(N_loc, sizeof(double));
    #if defined (USE_GPU)
        double* d_x_sol = DeviceMemory<uint32_t, double>::deviceCalloc(N_loc);
        solver.getSolution(d_x_sol);
        DeviceMemory<uint32_t, double>::copyDeviceToHost(N_loc, d_x_sol, x_sol);
    #else
        solver.getSolution(x_sol);
    #endif

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
