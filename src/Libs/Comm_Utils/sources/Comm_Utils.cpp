#include "Comm_Utils.hpp"

// Constructor
Comm_Utils::Comm_Utils(MPI_Comm& client_comm) {
    // Call setup method
    this->setup(client_comm);
    isParallel = true;
}

// Destructor
Comm_Utils::~Comm_Utils() {
    if (this->world_rank == 0) printf("--| Comm_Utils: Library comms destroyed!\n");
}

// Setup method
void Comm_Utils::setup(MPI_Comm& client_comm) {
    // lib_comm is client_comm
    this->lib_comm = client_comm;

    // Get world communicator rank and size
    MPI_Comm_rank(this->world_comm, &this->world_rank);
    MPI_Comm_size(this->world_comm, &this->world_size);

    // Get library communicator rank and size
    MPI_Comm_rank(this->lib_comm, &this->lib_rank);
    MPI_Comm_size(this->lib_comm, &this->lib_size);

    // NCCL setup
    #ifdef NCCL_COMMS
        // A single rank determines the unique ID
        if (this->lib_rank == 0) {
            NCCL_CHECK(ncclGetUniqueId(&this->nccl_uid));
        }

        // Bcast uid to all ranks
        MPI_CHECK(MPI_Bcast((void *)&this->nccl_uid, sizeof(this->nccl_uid), MPI_BYTE, 0, this->lib_comm));

        // Generate a CUDA stream for NCCL
        CUDA_CHECK(cudaStreamCreate(&this->nccl_stream));

        // Ranks initialize NCCL
        NCCL_CHECK(ncclCommInitRank(&this->nccl_comm, this->lib_size, this->nccl_uid, this->lib_rank));
    #endif
    isParallel = true;
}

// Allreduce wrappers
template <typename VTYPE>
void Comm_Utils::Allreduce_Sum(VTYPE* sendbuf, VTYPE* recvbuf, int count) {
    #ifdef NCCL_COMMS
        NCCL_CHECK(ncclAllReduce((const void*) sendbuf, (void*) recvbuf, count, nccl_utils::NCCLType<VTYPE>(), ncclSum, this->nccl_comm, this->nccl_stream));
        CUDA_CHECK(cudaStreamSynchronize(this->nccl_stream));
    #else
        MPI_Allreduce(sendbuf, recvbuf, count, mpi_utils::MPIType<VTYPE>(), MPI_SUM, this->lib_comm);
    #endif
}

// Explicit template instantiation for supported types
template void Comm_Utils::Allreduce_Sum<int>(int* sendbuf, int* recvbuf, int count);
template void Comm_Utils::Allreduce_Sum<uint32_t>(uint32_t* sendbuf, uint32_t* recvbuf, int count);
template void Comm_Utils::Allreduce_Sum<uint64_t>(uint64_t* sendbuf, uint64_t* recvbuf, int count);
template void Comm_Utils::Allreduce_Sum<float>(float* sendbuf, float* recvbuf, int count);
template void Comm_Utils::Allreduce_Sum<double>(double* sendbuf, double* recvbuf, int count);
#ifdef USE_GPU
    template void Comm_Utils::Allreduce_Sum<__nv_bfloat16>(__nv_bfloat16* sendbuf, __nv_bfloat16* recvbuf, int count);
#endif