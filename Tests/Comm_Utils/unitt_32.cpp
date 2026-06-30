#include "Comm_Utils.hpp"
#include "TensorUtils.hpp"

int main() {

    // Initialize MPI (not part of class)
    MPI_Init(NULL, NULL);

    // Get world comm data
    const MPI_Comm wcomm = MPI_COMM_WORLD;
    int world_rank, world_size;
    MPI_Comm_rank(wcomm, &world_rank);
    MPI_Comm_size(wcomm, &world_size);

    // Create test data
    // Test is a dot product: each rank generates part of the data
    PUSH_RANGE("Create data", 3);
    uint32_t arrSize_loc = 1024; // local size per rank
    float *x1_part = (float *)calloc(arrSize_loc, sizeof(float));
    float *x2_part = (float *)calloc(arrSize_loc, sizeof(float));
    for (uint32_t i = 0; i < arrSize_loc; ++i) {
        x1_part[i] = static_cast<float>(world_rank * arrSize_loc + i + 1);
        x2_part[i] = static_cast<float>(world_rank * arrSize_loc + 2*i);
    }
    #if defined(USE_GPU)
        float *d_x1_part = DeviceMemory<uint32_t, float>::deviceCalloc(arrSize_loc);
        float *d_x2_part = DeviceMemory<uint32_t, float>::deviceCalloc(arrSize_loc);
        DeviceMemory<uint32_t, float>::copyHostToDevice(arrSize_loc, x1_part, d_x1_part);
        DeviceMemory<uint32_t, float>::copyHostToDevice(arrSize_loc, x2_part, d_x2_part);
    #endif
    POP_RANGE();

    // Create a new communicator that is a duplicate of the world comm
    MPI_Comm new_comm;
    MPI_Comm_dup(wcomm, &new_comm);

    // Create a Comm_Utils object
    Comm_Utils commUtils(new_comm);

    // Perform a dot product, then Allreduce the result across ranks
    PUSH_RANGE("Dot product + Allreduce", 3);
    double* h_res_l = (double*)calloc(1, sizeof(double));
    double* h_res_g = (double*)calloc(1, sizeof(double));
    #if defined(USE_GPU)
        double* res_l = DeviceMemory<uint32_t, double>::deviceCalloc(1);
        double* res_g = DeviceMemory<uint32_t, double>::deviceCalloc(1);
        uint32_t numBlocks = (arrSize_loc + TILE_SIZE - 1) / TILE_SIZE;
        numBlocks = std::min(numBlocks, static_cast<uint32_t>(MAX_BLOCKS));
        dim3 grid(numBlocks,1,1);
        dim3 block(TILE_SIZE,1,1);
        DeviceUtils::Stream_t kStream;
        DeviceUtils::StreamCreate(&kStream);
        DeviceUtils::launchKernel(dot_product<uint32_t,float>, grid, block, kStream, d_x1_part, d_x2_part, res_l, arrSize_loc);
        commUtils.Allreduce_Sum(res_l, res_g, 1);
        DeviceMemory<uint32_t, double>::copyDeviceToHost(1, res_g, h_res_g);
    #else
        double* aux = (double*)calloc(1, sizeof(double));
        TensorUtils<uint32_t, float>::dot_product(arrSize_loc, x1_part, x2_part, aux);
        commUtils.Allreduce_Sum(aux, h_res_g, 1);
    #endif
    printf("Dot product result: %f\n", *h_res_g);
    POP_RANGE();

    // finalize MPI (not part of class)
    MPI_Finalize();
    return 0;
}