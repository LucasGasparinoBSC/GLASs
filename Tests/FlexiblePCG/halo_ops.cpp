#include "halo_ops.hpp"

// Comms for CPU matvec
template <typename ITYPE, typename RTYPE>
void halo_exchange(Comm_Utils commObj, RTYPE* left_data, RTYPE* right_data, const uint32_t nLocal, const RTYPE *c, const RTYPE *e, const RTYPE *x_in, RTYPE *x_out) {
    // ... nl-1 --|-- 0 ... nl-1 --|-- 0 ...
    //      p-1  exc      p       exc  p+1

    int prank = commObj.getLibRank();
    int nranks = commObj.getLibSize();
    MPI_Comm comm = commObj.getLibComm();

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

    #ifdef USE_GPU
    #pragma acc kernels deviceptr(left_data, right_data, c, e, x_out)
    #endif
    {
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
    }
}

template void halo_exchange<uint32_t, float>(Comm_Utils commObj, float *left_data, float *right_data, const uint32_t nLocal, const float *c, const float *e, const float *x_in, float *x_out);
template void halo_exchange<uint64_t, float>(Comm_Utils commObj, float *left_data, float *right_data, const uint32_t nLocal, const float *c, const float *e, const float *x_in, float *x_out);
template void halo_exchange<uint32_t, double>(Comm_Utils commObj, double *left_data, double *right_data, const uint32_t nLocal, const double *c, const double *e, const double *x_in, double *x_out);
template void halo_exchange<uint64_t, double>(Comm_Utils commObj, double *left_data, double *right_data, const uint32_t nLocal, const double *c, const double *e, const double *x_in, double *x_out);
#ifdef USE_GPU
    template void halo_exchange<uint32_t, __nv_bfloat16>(Comm_Utils commObj, __nv_bfloat16 *left_data, __nv_bfloat16 *right_data, const uint32_t nLocal, const __nv_bfloat16 *c, const __nv_bfloat16 *e, const __nv_bfloat16 *x_in, __nv_bfloat16 *x_out);
    template void halo_exchange<uint64_t, __nv_bfloat16>(Comm_Utils commObj, __nv_bfloat16 *left_data, __nv_bfloat16 *right_data, const uint32_t nLocal, const __nv_bfloat16 *c, const __nv_bfloat16 *e, const __nv_bfloat16 *x_in, __nv_bfloat16 *x_out);
#endif