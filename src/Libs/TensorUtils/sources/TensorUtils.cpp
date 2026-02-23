/**
 * @file TensorUtils.cpp
 * @author Lucas Gasparino (lucas.gasparino3110@gmail.com)
 * @brief Implementation of TensorUtils class methods.
 * @version 0.1
 * @date 2025-10-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

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
void TensorUtils<ITYPE, RTYPE>::dot_product(const ITYPE size, const RTYPE* x, const RTYPE* y, double* result) {
    result[0] = static_cast<double>(0);
    double tmp = 0.0;
    for (ITYPE i = 0; i < size; ++i) {
        tmp += static_cast<double>(x[i] * y[i]);
    }
    result[0] = tmp;
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

template <typename ITYPE, typename RTYPE>
void set_row(const ITYPE nrows, const ITYPE ncols, RTYPE *matrix, const ITYPE row_index, const RTYPE *row_vector) {
    if (row_index >= nrows) {
        std::cerr << "Error: Row index out of bounds in set_row." << std::endl;
        exit(EXIT_FAILURE);
    }
    ITYPE offset = row_index * ncols;
    for (ITYPE j = 0; j < ncols; ++j) {
        matrix[offset + j] = row_vector[j];
    }
}

template <typename ITYPE, typename RTYPE>
void extract_row(const ITYPE nrows, const ITYPE ncols, const RTYPE *matrix, const ITYPE row_index, RTYPE *row_vector) {
    if (row_index >= nrows) {
        std::cerr << "Error: Row index out of bounds in extract_row." << std::endl;
        exit(EXIT_FAILURE);
    }
    ITYPE offset = row_index * ncols;
    for (ITYPE j = 0; j < ncols; ++j) {
        row_vector[j] = matrix[offset + j];
    }
}

template <typename ITYPE, typename RTYPE>
void set_column(const ITYPE nrows, const ITYPE ncols, RTYPE *matrix, const ITYPE col_index, const RTYPE *col_vector)
{
    if (col_index >= ncols)
    {
        std::cerr << "Error: Column index out of bounds in set_column." << std::endl;
        exit(EXIT_FAILURE);
    }
    ITYPE offset = col_index;
    for (ITYPE i = 0; i < nrows; ++i)
    {
        matrix[i * ncols + offset] = col_vector[i];
    }
}

template <typename ITYPE, typename RTYPE>
void extract_column(const ITYPE nrows, const ITYPE ncols, const RTYPE *matrix, const ITYPE col_index, RTYPE *col_vector)
{
    if (col_index >= ncols)
    {
        std::cerr << "Error: Column index out of bounds in extract_column." << std::endl;
        exit(EXIT_FAILURE);
    }
    ITYPE offset = col_index;
    for (ITYPE i = 0; i < nrows; ++i)
    {
        col_vector[i] = matrix[i * ncols + offset];
    }
}

template class TensorUtils<uint32_t, float>;
template class TensorUtils<uint64_t, float>;
template class TensorUtils<uint32_t, double>;
template class TensorUtils<uint64_t, double>;