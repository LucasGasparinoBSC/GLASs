#include "Comm_Utils.hpp"
#include "TensorUtils.hpp"

int main() {

    // Initialize MPI (not part of class)
    MPI_Init(NULL, NULL);

    // Get mpi_rank and mpi_size (not part of class)
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    printf("Hello from rank %d of %d\n", mpi_rank, mpi_size);

    // finalize MPI (not part of class)
    MPI_Finalize();
    return 0;
}