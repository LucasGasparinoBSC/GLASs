#include "TestSolver.hpp"

int main() {

    // Init MPI env
    MPI_Init(NULL,NULL);
    MPI_Comm clientComm = MPI_COMM_WORLD;
    int csize, crank;
    MPI_CHECK( MPI_Comm_size(clientComm, &csize) );
    MPI_CHECK( MPI_Comm_rank(clientComm, &crank) );

    // Basic data
    uint32_t arrSize_l = static_cast<uint32_t>(1024000);
    uint32_t maxIters = 10;
    double tol = 1e-6;

    // Create an instance of TestSolver
    TestSolver<uint32_t,float> tSolv(clientComm, arrSize_l, maxIters, tol);

    // Setup test
    PUSH_RANGE("Setup Test",5);
    #if defined(USE_GPU)
        uint32_t numBlocks = (arrSize_l + (uint32_t)TILE_SIZE - 1) / (uint32_t)TILE_SIZE;
        numBlocks = std::min(numBlocks, (uint32_t)MAX_BLOCKS);
        dim3 kGrid(numBlocks,1,1);
        dim3 kBlock(TILE_SIZE,1,1);
        DeviceUtils::Stream_t kStream;
        DeviceUtils::StreamCreate(&kStream);
        float* d_inicond_l = DeviceMemory<uint32_t,float>::deviceCalloc(arrSize_l);
        float* d_rhs_l = DeviceMemory<uint32_t,float>::deviceCalloc(arrSize_l);
        DeviceUtils::launchKernel(set_array<uint32_t, float>, kGrid, kBlock, kStream, d_inicond_l, 1.0f, arrSize_l);
        DeviceUtils::launchKernel(set_array<uint32_t, float>, kGrid, kBlock, kStream, d_rhs_l, 3.0f, arrSize_l);
        DeviceUtils::StreamSynchronize(kStream);
        tSolv.setup(d_inicond_l, d_rhs_l);
    #else
        float* inicond_l = (float*)calloc(arrSize_l, sizeof(float));
        float* rhs_l = (float*)calloc(arrSize_l, sizeof(float));
        for (uint32_t i = 0; i < arrSize_l; ++i) {
            inicond_l[i] = 1.0f;
            rhs_l[i] = 3.0f;
        }
        tSolv.setup(inicond_l, rhs_l);
    #endif
    tSolv.CheckSetup();
    POP_RANGE();

    // Call the dummy solver
    tSolv.dummySolver();

    // Retrieve the solution
    PUSH_RANGE("Retrieve Solution",5);
    float* h_res = (float*)calloc(arrSize_l, sizeof(float));
    #if defined(USE_GPU)
        float* d_res = DeviceMemory<uint32_t,float>::deviceCalloc(arrSize_l);
        tSolv.getSolution(d_res);
        DeviceMemory<uint32_t,float>::copyDeviceToHost(arrSize_l, d_res, h_res);
    #else
        tSolv.getSolution(h_res);
    #endif
    POP_RANGE();

    // Check result
    PUSH_RANGE("Check Result",5);
    for (uint32_t i = 0; i < arrSize_l; ++i) {
        if (h_res[i] != 2.0f) {
            std::cerr << "Test failed at index " << i << ": expected 2.0, got " << h_res[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    POP_RANGE();

    // Finalize MPI
    MPI_Finalize();
    return 0;
}