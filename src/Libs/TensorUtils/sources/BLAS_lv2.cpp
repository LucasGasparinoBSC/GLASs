/**
 * @file BLAS_lv2.cpp
 * @author Lucas Gasparino (lucas.gasparino3110@gmail.com)
 * @brief Implementation of BLAS_lv2 class methods.
 * @version 0.1
 * @date 2025-10-07
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "BLAS_lv2.hpp"

/**
 * @brief compute x[:] += alpha * A[:][j], assuming column-major storage of A (transposed view of A).
 * In essence, the "row" being operated on is the j-th column of A, hence the parameter names.
 * @tparam ITYPE : uint32_t or uint64_t
 * @tparam RTYPE : float or double
 * @param cols : number of columns in the matrix A^T (rows of A)
 * @param rowIdx : index of the row in A^T (column in A)
 * @param alpha : scalar multiplier
 * @param A : pointer to the matrix A stored in column-major order (transposed view)
 * @param x : pointer to the vector x
 */
template <typename ITYPE, typename RTYPE>
void BLAS_lv2<ITYPE, RTYPE>::axpy_vec_mat(const ITYPE cols, const ITYPE rowIdx, const RTYPE alpha, const RTYPE* A, RTYPE* x) {
    const RTYPE* A_row = A + rowIdx * cols;
    for (ITYPE j = 0; j < cols; ++j) {
        x[j] += alpha * A_row[j];
    }
}

/**
 * @brief compute result = dot(A[:][j], x), assuming column-major storage of A (transposed view of A).
 * See AXPY_VEC_MAT for further explanation.
 * @tparam ITYPE : uint32_t or uint64_t
 * @tparam RTYPE : float or double
 * @param cols : number of columns in the matrix A^T (rows of A)
 * @param rowIdx : index of the row in A^T (column in A)
 * @param A : pointer to the matrix A stored in column-major order (transposed view)
 * @param x : pointer to the vector x
 * @param result : pointer to the result of the dot product
 */
template <typename ITYPE, typename RTYPE>
void BLAS_lv2<ITYPE, RTYPE>::dot_vec_mat(const ITYPE cols, const ITYPE rowIdx, const RTYPE* A, const RTYPE* x, double* result) {
    double tmp = static_cast<double>(0);
    const RTYPE* A_row = A + rowIdx * cols;
    for (ITYPE j = 0; j < cols; ++j) {
        const RTYPE value = A_row[j] * x[j];
        tmp += static_cast<double>(value);
    }
    result[0] = tmp;
}

template class BLAS_lv2<uint32_t, float>;
template class BLAS_lv2<uint64_t, float>;
template class BLAS_lv2<uint32_t, double>;
template class BLAS_lv2<uint64_t, double>;