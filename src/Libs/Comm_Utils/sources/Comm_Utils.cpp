#include "Comm_Utils.hpp"

// Constructor
template <typename ITYPE, typename RTYPE>
Comm_Utils<ITYPE, RTYPE>::Comm_Utils(MPI_Comm& client_comm) {
    // Call setup method
    this->setup(client_comm);
    isParallel = true;
}

// Destructor
template <typename ITYPE, typename RTYPE>
Comm_Utils<ITYPE, RTYPE>::~Comm_Utils() {
    if (this->world_rank == 0) printf("--| Comm_Utils: Library comms destroyed!\n");
}

// Setup method
template <typename ITYPE, typename RTYPE>
void Comm_Utils<ITYPE, RTYPE>::setup(MPI_Comm& client_comm) {
    // lib_comm is client_comm
    this->lib_comm = client_comm;

    // Get world communicator rank and size
    MPI_Comm_rank(this->world_comm, &this->world_rank);
    MPI_Comm_size(this->world_comm, &this->world_size);

    // Get library communicator rank and size
    MPI_Comm_rank(this->lib_comm, &this->lib_rank);
    MPI_Comm_size(this->lib_comm, &this->lib_size);
}

template class Comm_Utils<uint32_t, float>;
template class Comm_Utils<uint64_t, float>;
template class Comm_Utils<uint32_t, double>;
template class Comm_Utils<uint64_t, double>;
#ifdef USE_GPU
template class Comm_Utils<uint32_t, __nv_bfloat16>;
template class Comm_Utils<uint64_t, __nv_bfloat16>;
#endif