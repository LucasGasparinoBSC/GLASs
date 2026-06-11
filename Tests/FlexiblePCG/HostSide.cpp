#include "HostSide.hpp"

template <typename ITYPE, typename RTYPE>
void HostSide<ITYPE, RTYPE>::setLocalSizes(const ITYPE globalSize, const int rank, const int nranks, ITYPE &localSize, ITYPE *sizesPerRank)
{
    const ITYPE base = globalSize / static_cast<ITYPE>(nranks);
    const ITYPE rem = globalSize % static_cast<ITYPE>(nranks);
    for (int i = 0; i < nranks; i++)
        sizesPerRank[i] = base + ((static_cast<ITYPE>(i) < rem) ? static_cast<ITYPE>(1) : static_cast<ITYPE>(0));
    localSize = sizesPerRank[rank];
}

template <typename ITYPE, typename RTYPE>
void HostSide<ITYPE, RTYPE>::generate_matrix(const ITYPE N, RTYPE *c, RTYPE *d, RTYPE *e)
{
    for (ITYPE i = 0; i < N; i++)
    {
        c[i] = static_cast<RTYPE>(1);
        d[i] = static_cast<RTYPE>(4);
        e[i] = static_cast<RTYPE>(1);
    }
}

template <typename ITYPE, typename RTYPE>
void HostSide<ITYPE, RTYPE>::generate_inicond(const ITYPE N, RTYPE *x0, const ITYPE globalStart)
{
    // Build a deterministic initial state from global indices so
    // 1-rank and N-rank runs start from the same global x0.
    const RTYPE scale = static_cast<RTYPE>(0.002);
    const RTYPE inv997 = static_cast<RTYPE>(1.0 / 997.0);
    const RTYPE offset = static_cast<RTYPE>(0.001);
    for (ITYPE i = 0; i < N; i++)
    {
        const ITYPE gid = globalStart + i;
        const RTYPE frac = static_cast<RTYPE>((gid * static_cast<ITYPE>(37) + static_cast<ITYPE>(17)) % static_cast<ITYPE>(997)) * inv997;
        x0[i] = frac * scale - offset;
    }
}

template <typename ITYPE, typename RTYPE>
void HostSide<ITYPE, RTYPE>::generate_rhs(const ITYPE N, RTYPE *b, const ITYPE globalStart)
{
    for (ITYPE i = 0; i < N; i++)
    {
        b[i] = static_cast<RTYPE>(globalStart + i + static_cast<ITYPE>(1));
    }
}

template <typename ITYPE, typename RTYPE>
void HostSide<ITYPE, RTYPE>::computeLeftRightRanks(const int irank, const int nranks, int& leftRank, int& rightRank)
{
    leftRank = (irank - 1 + nranks) % nranks;
    rightRank = (irank + 1) % nranks;
}

template <typename ITYPE, typename RTYPE>
void HostSide<ITYPE, RTYPE>::matvec_nohalo(const RTYPE *cl, const RTYPE *dl, const RTYPE *el, const RTYPE *x, RTYPE *y, const ITYPE n)
{
    // Loop internal nodes only
    for (ITYPE i = 1; i < n - 1; i++)
    {
        y[i] = cl[i - 1] * x[i - 1] + dl[i] * x[i] + el[i + 1] * x[i + 1];
    }
}

template <typename ITYPE, typename RTYPE>
void HostSide<ITYPE, RTYPE>::matvec_halo(const RTYPE *cl, const RTYPE *dl, const RTYPE *el, const RTYPE *x, const RTYPE *ghosts, RTYPE *y, const ITYPE n)
{
    // Complete interface nodes
    y[0] = dl[0] * x[0] + el[1] * x[1] + ghosts[0] * ghosts[1];
    y[n - 1] = cl[n - 2] * x[n - 2] + dl[n - 1] * x[n - 1] + ghosts[2] * ghosts[3];
}

template <typename ITYPE, typename RTYPE>
void HostSide<ITYPE, RTYPE>::diag_precond(const RTYPE* dl, const RTYPE* r, RTYPE* z, const ITYPE n)
{
    for (ITYPE i = 0; i < n; i++)
    {
        z[i] = r[i] / dl[i];
    }
}

template class HostSide<uint32_t, float>;
template class HostSide<uint64_t, float>;
template class HostSide<uint32_t, double>;
template class HostSide<uint64_t, double>;

template <typename ITYPE, typename RTYPE>
void Matvec<ITYPE, RTYPE>::fillBuffer(const RTYPE* cl, const RTYPE* el, const RTYPE* x, RTYPE* buf, const ITYPE n)
{
    #if defined (USE_GPU)
        DeviceUtils::launchKernel(AuxKernels::fillBuffer<ITYPE,RTYPE>, dim3(1), dim3(1), 0, cl, el, x, buf, n);
    #else
        // Fill buffer with left and right halo data: [left_cl, left_u, right_el, right_u]
        buf[0] = cl[n-1];
        buf[1] = x[n-1];
        buf[2] = el[0];
        buf[3] = x[0];
    #endif
}

// Overlapped comms-compute using 1-sided RMA comms
// Example cyclic grid w/ partition:
//
//           P0      |       P1      |       P2      |  P0
//                   |               |               |
//      0    1    2  |  0    1    2  |  0    1    2  |  0
//    ->|----|----|--|--|----|----|--|--|----|----|--|--|->
//      0    1    2     3    4    5     6    7    8     9==0 (slave to 0)
template <typename ITYPE, typename RTYPE>
void Matvec<ITYPE, RTYPE>::launch_matvec(MPI_Win &win, const int irank, const int nranks, const int leftRank, const int rightRank,
                                         const RTYPE *cl, const RTYPE *dl, const RTYPE *el, const RTYPE *x, RTYPE *ghosts, RTYPE *y, const ITYPE n)
{
    // Serial case: no comms needed
    if (nranks == 1) {
        // Internal nodes (1 -> n-2)
        #if defined (USE_GPU)
        #else
            HostSide<ITYPE, RTYPE>::matvec_nohalo(cl, dl, el, x, y, n);
            // Boundary nodes: note y[n-1] is the node BEFORE the actual last node (in this case n = Nwork = N-1)
            y[0] = cl[n - 1] * x[n - 1] + dl[0] * x[0] + el[1] * x[1];
            y[n - 1] = cl[n - 2] * x[n - 2] + dl[n - 1] * x[n - 1] + el[0] * x[0];
        #endif
    } else {
        // start epoch
        MPI_Win_fence(0, win);

        // Get from left neighbor into ghosts[0] and ghosts[1]
        MPI_Get(&ghosts[0], 2, mpi_utils::MPIType<RTYPE>(), leftRank, 0, 2, mpi_utils::MPIType<RTYPE>(), win);

        // Get from right neighbor into ghosts[2] and ghosts[3]
        MPI_Get(&ghosts[2], 2, mpi_utils::MPIType<RTYPE>(), rightRank, 2, 2, mpi_utils::MPIType<RTYPE>(), win);

        // Compute internal nodes while comms are in-flight
        #if defined (USE_GPU)
            //matvec_nohalo(cl, dl, el, x, y, n);
        #else
            HostSide<ITYPE, RTYPE>::matvec_nohalo(cl, dl, el, x, y, n);
        #endif

        // end epoch
        MPI_Win_fence(0, win);

        // Compute boundary nodes after ensuring data has arrived
        #if defined (USE_GPU)
            //matvec_halo(cl, dl, el, x, ghosts, y, n);
        #else
            HostSide<ITYPE,RTYPE>::matvec_halo(cl, dl, el, x, ghosts, y, n);
        #endif
    }
}

template class Matvec<uint32_t, float>;
template class Matvec<uint64_t, float>;
template class Matvec<uint32_t, double>;
template class Matvec<uint64_t, double>;
#if defined (USE_GPU)
    template class Matvec<uint32_t, DeviceUtils::bf16>;
    template class Matvec<uint64_t, DeviceUtils::bf16>;
#endif
