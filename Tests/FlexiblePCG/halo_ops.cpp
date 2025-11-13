#include "halo_ops.hpp"

// Comms for CPU matvec
template <typename ITYPE, typename RTYPE>
void halo_exchange(Comm_Utils commObj, const uint32_t nLocal, RTYPE *c, RTYPE *e, const RTYPE *x_in, RTYPE *x_out) {
    // ... nl-1 --|-- 0 ... nl-1 --|-- 0 ...
    //      p-1  exc      p       exc  p+1

    int prank = commObj.getLibRank();
    int nranks = commObj.getLibSize();
    MPI_Comm comm = commObj.getLibComm();

    // Generic exchanges: both left and right interfaces
    RTYPE *left_data;
    RTYPE *right_data;
    left_data =  (RTYPE*)calloc(1, sizeof(RTYPE));
    right_data = (RTYPE*)calloc(1, sizeof(RTYPE));

    // Ring algo for exchange, determine left and right ranks using modulo
    int left_rank = (prank - 1 + nranks) % nranks;
    int right_rank = (prank + 1) % nranks;

    // Ring left2right xin[0]
    MPI_Sendrecv(&x_in[0], 1, mpi_utils::MPIType<RTYPE>(), left_rank, 0,
                 right_data, 1, mpi_utils::MPIType<RTYPE>(), right_rank, 0,
                 comm, MPI_STATUS_IGNORE);

    // Ring right2left xin[nLocal-1]
    MPI_Sendrecv(&x_in[nLocal - 1], 1, mpi_utils::MPIType<RTYPE>(), right_rank, 0,
                 left_data, 1, mpi_utils::MPIType<RTYPE>(), left_rank, 0,
                 comm, MPI_STATUS_IGNORE);

    if (prank == 0)
    {
        x_out[nLocal - 1] += e[nLocal - 1] * right_data[0];
    }
    else if (prank == nranks - 1)
    {
        x_out[0] += e[0] * left_data[0];
    }
    else
    {
        x_out[0] += c[0] * left_data[0];
        x_out[nLocal - 1] += e[nLocal - 1] * right_data[0];
    }

    free(left_data);
    free(right_data);
}

template void halo_exchange<uint32_t, float>(Comm_Utils commObj, const uint32_t nLocal, float *c, float *e, const float *x_in, float *x_out);
template void halo_exchange<uint64_t, float>(Comm_Utils commObj, const uint32_t nLocal, float *c, float *e, const float *x_in, float *x_out);
template void halo_exchange<uint32_t, double>(Comm_Utils commObj, const uint32_t nLocal, double *c, double *e, const double *x_in, double *x_out);
template void halo_exchange<uint64_t, double>(Comm_Utils commObj, const uint32_t nLocal, double *c, double *e, const double *x_in, double *x_out);