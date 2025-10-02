#include "TensorUtils.hpp"

template <typename ITYPE, typename RTYPE>
void TensorUtils<ITYPE, RTYPE>::axpy(const ITYPE size, const RTYPE alpha, const RTYPE* x, RTYPE* y) {
    for (ITYPE i = 0; i < size; ++i) {
        y[i] += alpha * x[i];
    }
}

template <typename ITYPE, typename RTYPE>
void TensorUtils<ITYPE, RTYPE>::scale(const ITYPE size, const RTYPE alpha, RTYPE* x) {
    for (ITYPE i = 0; i < size; ++i) {
        x[i] *= alpha;
    }
}

template <typename ITYPE, typename RTYPE>
void TensorUtils<ITYPE, RTYPE>::dot_product(const ITYPE size, const RTYPE* x, const RTYPE* y, RTYPE* result) {
    result[0] = static_cast<RTYPE>(0);
    double tmp = 0.0;
    for (ITYPE i = 0; i < size; ++i) {
        tmp += static_cast<double>(x[i] * y[i]);
    }
    result[0] = static_cast<RTYPE>(tmp);
}

template <typename ITYPE, typename RTYPE>
void TensorUtils<ITYPE, RTYPE>::multiply_entries(const ITYPE size, const RTYPE* x, RTYPE* y) {
    for (ITYPE i = 0; i < size; ++i) {
        y[i] *= x[i];
    }
}

template <typename ITYPE, typename RTYPE>
void TensorUtils<ITYPE, RTYPE>::invert_entries(const ITYPE size, RTYPE* x) {
    for (ITYPE i = 0; i < size; ++i) {
        x[i] = static_cast<RTYPE>(1) / x[i];
    }
}

template <typename ITYPE, typename RTYPE>
void TensorUtils<ITYPE, RTYPE>::copy_array(const ITYPE size, const RTYPE* x, RTYPE* y) {
    for (ITYPE i = 0; i < size; ++i) {
        y[i] = x[i];
    }
}

template class TensorUtils<uint32_t, float>;
template class TensorUtils<uint64_t, float>;
template class TensorUtils<uint32_t, double>;
template class TensorUtils<uint64_t, double>;