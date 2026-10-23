#ifndef BLAS_LV2_HPP
#define BLAS_LV2_HPP

#pragma once

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

template <typename ITYPE, typename RTYPE>
class BLAS_lv2
{
    private:
        // Auxiliary intrinsics functions for vectorized operations
        #if defined(__x86_64__) || defined(__i386__)
            static inline double hsum256_pd(__m256d v);
        #endif

        // GEMV type-specific implementations for different matrix formats
        static void GEMV_dense(const ITYPE rows, const ITYPE cols, const RTYPE* __restrict A, const RTYPE* __restrict x, double* __restrict y);
        static void GEMV_csr();
        static void GEMV_coo();
        static void GEMV_ell();
        static void GEMV_bsr();

    public:
        // BLAS-1 using matrix rows/columns as vectors
        static void axpy_vec_mat(const ITYPE cols, const ITYPE rowIdx, const RTYPE alpha, const RTYPE* A, RTYPE* x);
        static void dot_vec_mat(const ITYPE cols, const ITYPE rowIdx, const RTYPE* A, const RTYPE* x, double* result);

        /**
         * @brief GEMV API: selects between different matrix formats (dense, csr, coo, ell, bsr) at compile time.
         * @tparam Format : integer representing the matrix format (0: dense, 1: csr, 2: coo, 3: ell, 4: bsr)
         * @tparam Args : variadic template parameters for additional arguments specific to the chosen format
         * @param rows : number of rows in input array x
         * @param x : pointer to the input array RTYPE* x
         * @param y : pointer to the output array double* y
         */
        template <int Format, typename... Args>
        static void GEMV(const RTYPE* x, double* y, Args &&...args)
        {
            if constexpr (Format == 0) { // dense
                // Enforce dense signature at compile time: GEMV(x, y, rows, cols, A)
                static_assert(sizeof...(args) == 3,
                              "Error: GEMV with dense format requires exactly 3 additional arguments (rows, cols, A).");
                using DenseArgs = std::tuple<std::decay_t<Args>...>;
                static_assert(std::is_same_v<std::tuple_element_t<0, DenseArgs>, ITYPE>,
                              "Error: GEMV with dense format requires the first additional argument to be ITYPE (rows).");
                static_assert(std::is_same_v<std::tuple_element_t<1, DenseArgs>, ITYPE>,
                              "Error: GEMV with dense format requires the second additional argument to be ITYPE (cols).");
                static_assert(std::is_same_v<std::tuple_element_t<2, DenseArgs>, RTYPE*>,
                              "Error: GEMV with dense format requires the third additional argument to be RTYPE* (A).");
                // Call the dense GEMV implementation
                GEMV_dense(std::get<0>(std::forward_as_tuple(args...)),
                           std::get<1>(std::forward_as_tuple(args...)),
                           std::get<2>(std::forward_as_tuple(args...)),
                           x, y);
            } else if constexpr (Format == 1) { // csr
                // GEMV_csr(std::forward<Args>(args)..., x, y);
                std::cerr << "Error: GEMV with CSR format is not yet implemented." << std::endl;
                std::exit(EXIT_FAILURE);
            } else if constexpr (Format == 2) { // coo
                // GEMV_coo(std::forward<Args>(args)..., x, y);
                std::cerr << "Error: GEMV with COO format is not yet implemented." << std::endl;
            } else if constexpr (Format == 3) { // ell
                // GEMV_ell(std::forward<Args>(args)..., x, y);
                std::cerr << "Error: GEMV with ELL format is not yet implemented." << std::endl;
            } else if constexpr (Format == 4) { // bsr
                // GEMV_bsr(std::forward<Args>(args)..., x, y);
                std::cerr << "Error: GEMV with BSR format is not yet implemented." << std::endl;
            } else {
                std::cerr << "Error: Unsupported matrix format for GEMV." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
};

#endif //! BLAS_LV2_HPP