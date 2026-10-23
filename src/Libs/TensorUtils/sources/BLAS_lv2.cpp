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

// Auxiliary intrinsics functions for vectorized operations
#if defined(__x86_64__) || defined(__i386__)
template <typename ITYPE, typename RTYPE>
double BLAS_lv2<ITYPE, RTYPE>::hsum256_pd(__m256d v) {
    const __m128d lo = _mm256_castpd256_pd128(v);
    const __m128d hi = _mm256_extractf128_pd(v, 1);
    const __m128d pair = _mm_add_pd(lo, hi);
    const __m128d hi64 = _mm_unpackhi_pd(pair, pair);
    return _mm_cvtsd_f64(_mm_add_sd(pair, hi64));
}
#endif

// GEMV implementations

/**
 * @brief Dense matrix-vector multiplication (GEMV)
 * For now, this is without ANY optimizations, just a simple implementation.
 * Nevertheless, it takes RTYPE inputs and produces a double output, which is important for numerical stability.
 * @tparam ITYPE : uint32_t or uint64_t
 * @tparam RTYPE : float or double
 */
template <typename ITYPE, typename RTYPE>
void BLAS_lv2<ITYPE, RTYPE>::GEMV_dense(const ITYPE rows, const ITYPE cols, const RTYPE* __restrict A, const RTYPE* __restrict x, double* __restrict y) {
    #if defined(__x86_64__) || defined(__i386__)
        if constexpr (std::is_same_v<RTYPE, float>) {
            // 4-row AVX micro-kernel specialized for float inputs with double accumulation.
            ITYPE i = 0;
            for (; i + static_cast<ITYPE>(4) <= rows; i += static_cast<ITYPE>(4)) {
                const float* row0 = A + (i + static_cast<ITYPE>(0)) * cols;
                const float* row1 = A + (i + static_cast<ITYPE>(1)) * cols;
                const float* row2 = A + (i + static_cast<ITYPE>(2)) * cols;
                const float* row3 = A + (i + static_cast<ITYPE>(3)) * cols;

                __m256d s0_lo = _mm256_setzero_pd();
                __m256d s0_hi = _mm256_setzero_pd();
                __m256d s1_lo = _mm256_setzero_pd();
                __m256d s1_hi = _mm256_setzero_pd();
                __m256d s2_lo = _mm256_setzero_pd();
                __m256d s2_hi = _mm256_setzero_pd();
                __m256d s3_lo = _mm256_setzero_pd();
                __m256d s3_hi = _mm256_setzero_pd();

                ITYPE j = 0;
                for (; j + static_cast<ITYPE>(8) <= cols; j += static_cast<ITYPE>(8)) {
                    const __m256 xv = _mm256_loadu_ps(x + j);

                    const __m256 p0 = _mm256_mul_ps(_mm256_loadu_ps(row0 + j), xv);
                    const __m256 p1 = _mm256_mul_ps(_mm256_loadu_ps(row1 + j), xv);
                    const __m256 p2 = _mm256_mul_ps(_mm256_loadu_ps(row2 + j), xv);
                    const __m256 p3 = _mm256_mul_ps(_mm256_loadu_ps(row3 + j), xv);

                    s0_lo = _mm256_add_pd(s0_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p0)));
                    s0_hi = _mm256_add_pd(s0_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p0, 1)));
                    s1_lo = _mm256_add_pd(s1_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p1)));
                    s1_hi = _mm256_add_pd(s1_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p1, 1)));
                    s2_lo = _mm256_add_pd(s2_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p2)));
                    s2_hi = _mm256_add_pd(s2_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p2, 1)));
                    s3_lo = _mm256_add_pd(s3_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p3)));
                    s3_hi = _mm256_add_pd(s3_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p3, 1)));
                }

                double sum0 = hsum256_pd(_mm256_add_pd(s0_lo, s0_hi));
                double sum1 = hsum256_pd(_mm256_add_pd(s1_lo, s1_hi));
                double sum2 = hsum256_pd(_mm256_add_pd(s2_lo, s2_hi));
                double sum3 = hsum256_pd(_mm256_add_pd(s3_lo, s3_hi));

                for (; j < cols; ++j) {
                    const float xj = x[j];
                    sum0 += static_cast<double>(row0[j] * xj);
                    sum1 += static_cast<double>(row1[j] * xj);
                    sum2 += static_cast<double>(row2[j] * xj);
                    sum3 += static_cast<double>(row3[j] * xj);
                }

                y[i + static_cast<ITYPE>(0)] = sum0;
                y[i + static_cast<ITYPE>(1)] = sum1;
                y[i + static_cast<ITYPE>(2)] = sum2;
                y[i + static_cast<ITYPE>(3)] = sum3;
            }

            // Remainder rows use the single-row AVX structure.
            for (; i < rows; ++i) {
                const float* Arow = A + i * cols;
                __m256d sum_lo = _mm256_setzero_pd();
                __m256d sum_hi = _mm256_setzero_pd();

                ITYPE j = 0;
                for (; j + static_cast<ITYPE>(8) <= cols; j += static_cast<ITYPE>(8)) {
                    const __m256 a = _mm256_loadu_ps(Arow + j);
                    const __m256 b = _mm256_loadu_ps(x + j);
                    const __m256 prod = _mm256_mul_ps(a, b);

                    sum_lo = _mm256_add_pd(sum_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(prod)));
                    sum_hi = _mm256_add_pd(sum_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(prod, 1)));
                }

                double sum = hsum256_pd(_mm256_add_pd(sum_lo, sum_hi));
                for (; j < cols; ++j) {
                    sum += static_cast<double>(Arow[j] * x[j]);
                }
                y[i] = sum;
            }
            return;
        }
    #endif

    // Generic path for non-float RTYPE or non-x86 targets.
    for (ITYPE i = 0; i < rows; ++i) {
        const RTYPE* Arow = A + i * cols;
        double sum = 0.0;
        RTYPE multval;
        #pragma omp simd reduction(+:sum)
        for (ITYPE j = 0; j < cols; ++j) {
            multval = Arow[j] * x[j];
            sum += static_cast<double>(multval);
        }
        y[i] = sum;
    }
}

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