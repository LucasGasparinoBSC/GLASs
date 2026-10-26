/**
 * @file LAPACK_interface.hpp
 * @author your name (you@domain.com)
 * @brief Thin layer abstraction for LAPACK calls
 * @details For now this is meant mostly to provide access to
 * simple LAPACK calls like GESV and TRSV for both FP32 and
 * FP64 inputs/outpus. The API should be able to call MKL
 * LAPACK for Intel OneAPI/LLVM compilers.
 * @version 0.1
 * @date 2026-10-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#ifndef LAPACK_INTERFACE_HPP
#define LAPACK_INTERFACE_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// LAPACK includes:


template <typename ITYPE, typename RTYPE>
class LAPACK_interface
{
    private:
    public:
        /**
         * @brief Solves an upper triangular system of linear equations with a single right-hand side.
         * Assumes A is stored full and in column-major order. Notice ncols <= lda.
         * 
         * @param ncols Number of columns of the matrix A to process
         * @param lda Leading dimension of the matrix A (stride)
         * @param A Pointer to the upper triangular matrix A.
         * @param b Pointer to the right-hand side vector b.
         * @param x Pointer to the solution vector x.
         */
        static void UTRSV(const ITYPE ncols, const ITYPE lda, const RTYPE* __restrict A, const RTYPE* __restrict b, RTYPE* __restrict x);
};

#endif //! LAPACK_INTERFACE_HPP