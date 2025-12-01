#include "Comm_Utils.hpp"

// Empty constructor
Comm_Utils::Comm_Utils() {
    isParallel = false;
    world_rank = 0;
    world_size = 1;
    lib_rank = 0;
    lib_size = 1;
}

// Constructor
Comm_Utils::Comm_Utils(MPI_Comm& client_comm) {
    // Call setup method
    this->setup(client_comm);
    isParallel = true;
}

// Destructor
Comm_Utils::~Comm_Utils() {
}

// Setup method
void Comm_Utils::setup(MPI_Comm& client_comm) {
    PUSH_RANGE("Comm_Utils::setup", 0)
    // lib_comm is client_comm
    this->lib_comm = client_comm;

    // Get world communicator rank and size
    MPI_Comm_rank(this->world_comm, &this->world_rank);
    MPI_Comm_size(this->world_comm, &this->world_size);

    // Get library communicator rank and size
    MPI_Comm_rank(this->lib_comm, &this->lib_rank);
    MPI_Comm_size(this->lib_comm, &this->lib_size);

    // Bind GPUs to ranks
    #ifdef USE_GPU
        // Shared memory communicator for local ranks
        MPI_Comm shm_comm;
        MPI_CHECK(MPI_Comm_split_type(this->lib_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm));

        // Get shm ranks and sizes
        int shm_rank, shm_size;
        MPI_CHECK(MPI_Comm_rank(shm_comm, &shm_rank));
        MPI_CHECK(MPI_Comm_size(shm_comm, &shm_size));

        // Get the total number of GPUs available
        int total_gpus = 0;
        CUDA_CHECK(cudaGetDeviceCount(&total_gpus));

        // Get the device id for the shm communicator
        int id_device = 0;
        id_device = shm_rank % total_gpus;
        CUDA_CHECK(cudaSetDevice(id_device));

        printf("--| Comm_Utils: World Rank %d, Lib Rank %d, Shm Rank %d/%d assigned to GPU %d (Total GPUs: %d)\n",
               this->world_rank, this->lib_rank, shm_rank, shm_size, id_device, total_gpus);
    #endif

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
    POP_RANGE

    // Warmup initialization message
    double *warmup_buf = nullptr;
    double *sum_buf = nullptr;
    #ifdef USE_GPU
        CUDA_CHECK(cudaMalloc((void **)&warmup_buf, sizeof(double)));
        CUDA_CHECK(cudaMemset(warmup_buf, 0, sizeof(double)));
        CUDA_CHECK(cudaMalloc((void **)&sum_buf, sizeof(double)));
        CUDA_CHECK(cudaMemset(sum_buf, 0, sizeof(double)));
    #else
        warmup_buf = (double *)calloc(1, sizeof(double));
        memset(warmup_buf, 0, sizeof(double));
        sum_buf = (double *)calloc(1, sizeof(double));
        memset(sum_buf, 0, sizeof(double));
    #endif
    int count = 1;
    this->Allreduce_Sum(warmup_buf, sum_buf, count);
}

// Allreduce wrappers
template <typename VTYPE>
void Comm_Utils::Allreduce_Sum(VTYPE* sendbuf, VTYPE* recvbuf, int count) {
    PUSH_RANGE("Comm_Utils::Allreduce_Sum", 0)
    #ifdef USE_GPU
        // Synchronize before comms
        CUDA_CHECK(cudaStreamSynchronize(0));
    #endif
    #ifdef NCCL_COMMS
        NCCL_CHECK(ncclAllReduce((const void*) sendbuf, (void*) recvbuf, count, nccl_utils::NCCLType<VTYPE>(), ncclSum, this->nccl_comm, this->nccl_stream));
        CUDA_CHECK(cudaStreamSynchronize(this->nccl_stream));
    #else
        MPI_Allreduce(sendbuf, recvbuf, count, mpi_utils::MPIType<VTYPE>(), MPI_SUM, this->lib_comm);
    #endif
    POP_RANGE
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