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
    uint32_t arrSize_loc = 10; // local size per rank
    uint32_t arrSize = arrSize_loc * world_size; // global size
    float *x1_part = (float *)calloc(arrSize_loc, sizeof(float));
    float *x2_part = (float *)calloc(arrSize_loc, sizeof(float));
    for (uint32_t i = 0; i < arrSize_loc; ++i) {
        x1_part[i] = static_cast<float>(world_rank * arrSize_loc + i + 1);
        x2_part[i] = static_cast<float>(world_rank * arrSize_loc + 2*i);
    }

    // Call the TensorUtils dot_product function
    float local_dot = 0.0f;
    TensorUtils<uint32_t, float>::dot_product(arrSize_loc, x1_part, x2_part, &local_dot);

    // MPI_Allreduce to get the global dot product
    float global_dot = 0.0f;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_FLOAT, MPI_SUM, wcomm);
    if (world_rank == 0) {
        std::cout << "Global dot product result: " << global_dot << std::endl;
    }

    // Create a new communicator that is a duplicate of the world comm
    MPI_Comm new_comm;
    MPI_Comm_dup(wcomm, &new_comm);
    int new_rank, new_size;
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);

    // Create a Comm_Utils object
    Comm_Utils<uint32_t, float> commUtils(new_comm, world_rank, world_size, new_rank, new_size);

    // finalize MPI (not part of class)
    MPI_Finalize();
    return 0;
}