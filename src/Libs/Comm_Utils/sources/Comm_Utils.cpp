#include "Comm_Utils.hpp"

// Constructor
template <typename ITYPE, typename RTYPE>
Comm_Utils<ITYPE, RTYPE>::Comm_Utils(MPI_Comm comm, ITYPE wr, ITYPE ws, ITYPE cr, ITYPE cs) {
    this->client_comm = comm;
    this->world_rank = wr;
    this->world_size = ws;
    this->client_rank = cr;
    this->client_size = cs;
    MPI_CHECK(MPI_Comm_dup(this->client_comm, &this->lib_comm));
    this->lib_rank = this->client_rank;
    this->lib_size = this->client_size;
    if (this->world_rank == 0) printf("--| Comm_Utils: Library comms initialized!\n");
    MPI_CHECK(MPI_Barrier(this->world_comm));
}

// Destructor
template <typename ITYPE, typename RTYPE>
Comm_Utils<ITYPE, RTYPE>::~Comm_Utils() {
    if (this->world_rank == 0) printf("--| Comm_Utils: Library comms destroyed!\n");
}

template class Comm_Utils<uint32_t, float>;
template class Comm_Utils<uint64_t, float>;
template class Comm_Utils<uint32_t, double>;
template class Comm_Utils<uint64_t, double>;