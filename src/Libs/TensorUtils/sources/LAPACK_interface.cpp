#include "LAPACK_interface.hpp"

template <typename ITYPE, typename RTYPE>
void LAPACK_interface<ITYPE, RTYPE>::UTRSV(const ITYPE ncols, const ITYPE lda, const RTYPE* __restrict A, const RTYPE* __restrict b, RTYPE* __restrict x)
{
    // Automatically maps uint32_t -> int32_t, or uint64_t -> int64_t (no truncation!)
    using SignedITYPE = std::make_signed_t<ITYPE>;
    SignedITYPE kStore = static_cast<SignedITYPE>(ncols);
    SignedITYPE ldaStore = static_cast<SignedITYPE>(lda);
    SignedITYPE idx, jdx;

    for (idx = kStore; idx >= 0; --idx)
    {
        x[idx] = b[idx];
        for (jdx = idx + 1; jdx <= kStore; ++jdx)
        {
            x[idx] -= A[idx + jdx * ldaStore] * x[jdx];
        }
        x[idx] /= A[idx + idx * ldaStore];
    }
}

template class LAPACK_interface<uint32_t, float>;
template class LAPACK_interface<uint64_t, float>;
template class LAPACK_interface<uint32_t, double>;
template class LAPACK_interface<uint64_t, double>;