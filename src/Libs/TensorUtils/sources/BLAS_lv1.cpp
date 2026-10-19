/**
 * @file BLAS_lv1.cpp
 * @author Lucas Gasparino (lucas.gasparino3110@gmail.com)
 * @brief Implementation of BLAS_lv1 class methods.
 * @version 0.1
 * @date 2025-10-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "BLAS_lv1.hpp"

template <typename ITYPE, typename RTYPE>
void BLAS_lv1<ITYPE, RTYPE>::axpy(const ITYPE size, const RTYPE alpha, const RTYPE *x, RTYPE *y)
{
    for (ITYPE i = 0; i < size; ++i)
    {
        y[i] += alpha * x[i];
    }
}

template <typename ITYPE, typename RTYPE>
void BLAS_lv1<ITYPE, RTYPE>::scale(const ITYPE size, const RTYPE alpha, RTYPE *x)
{
    for (ITYPE i = 0; i < size; ++i)
    {
        x[i] *= alpha;
    }
}

template <typename ITYPE, typename RTYPE>
void BLAS_lv1<ITYPE, RTYPE>::dot_product(const ITYPE size, const RTYPE *x, const RTYPE *y, double *result)
{
    result[0] = static_cast<double>(0);
    double tmp = 0.0;
    for (ITYPE i = 0; i < size; ++i)
    {
        const RTYPE value = x[i] * y[i];
        tmp += static_cast<double>(value);
    }
    result[0] = tmp;
}

template <typename ITYPE, typename RTYPE>
void BLAS_lv1<ITYPE, RTYPE>::guided_dot_product(const ITYPE sizeList, const ITYPE *listEntries, const RTYPE *x, const RTYPE *y, double *result)
{
    result[0] = static_cast<double>(0);
    double tmp = 0.0;
    for (ITYPE i = 0; i < sizeList; ++i)
    {
        ITYPE idx = listEntries[i];
        const RTYPE value = x[idx] * y[idx];
        tmp += static_cast<double>(value);
    }
    result[0] = tmp;
}

template <typename ITYPE, typename RTYPE>
void BLAS_lv1<ITYPE, RTYPE>::multiply_entries(const ITYPE size, const RTYPE *x, RTYPE *y)
{
    for (ITYPE i = 0; i < size; ++i)
    {
        y[i] *= x[i];
    }
}

template <typename ITYPE, typename RTYPE>
void BLAS_lv1<ITYPE, RTYPE>::invert_entries(const ITYPE size, RTYPE *x)
{
    for (ITYPE i = 0; i < size; ++i)
    {
        x[i] = static_cast<RTYPE>(1) / x[i];
    }
}

template <typename ITYPE, typename RTYPE>
void BLAS_lv1<ITYPE, RTYPE>::copy_array(const ITYPE size, const RTYPE *x, RTYPE *y)
{
    for (ITYPE i = 0; i < size; ++i)
    {
        y[i] = x[i];
    }
}

template class BLAS_lv1<uint32_t, float>;
template class BLAS_lv1<uint64_t, float>;
template class BLAS_lv1<uint32_t, double>;
template class BLAS_lv1<uint64_t, double>;