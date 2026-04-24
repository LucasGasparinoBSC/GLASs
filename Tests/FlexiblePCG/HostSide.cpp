#include "HostSide.hpp"

template <typename ITYPE, typename RTYPE>
void HostSide<ITYPE, RTYPE>::setLocalSizes(const ITYPE globalSize, const int rank, const int nranks, ITYPE &localSize, ITYPE *sizesPerRank)
{
    double ratio = static_cast<double>(globalSize) / static_cast<double>(nranks);
    uint32_t aux = static_cast<uint32_t>(ratio);
    if ((ratio - static_cast<double>(aux)) > 0.5)
        aux++;
    for (int i = 0; i < nranks; i++)
        sizesPerRank[i] = aux;
    sizesPerRank[nranks - 1] = globalSize - (nranks - 1) * aux;
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
void HostSide<ITYPE, RTYPE>::generate_inicond(const ITYPE N, RTYPE *x0)
{
    RTYPE scale = static_cast<RTYPE>(0.002);
    RTYPE offset = static_cast<RTYPE>(0.001);
    for (ITYPE i = 0; i < N; i++)
    {
        x0[i] = static_cast<RTYPE>(rand()) / static_cast<RTYPE>(RAND_MAX) * scale - offset;
    }
}

template <typename ITYPE, typename RTYPE>
void HostSide<ITYPE, RTYPE>::generate_rhs(const ITYPE N, RTYPE *b, const int rank)
{
    for (ITYPE i = 0; i < N; i++)
    {
        ITYPE val = (rank * N) + i + 1;
        // val = static_cast<uint32_t>(1);
        b[i] = static_cast<RTYPE>(val);
    }
}

template class HostSide<uint32_t, float>;
template class HostSide<uint64_t, float>;
template class HostSide<uint32_t, double>;
template class HostSide<uint64_t, double>;