#ifndef BLAS_LV2_HPP
#define BLAS_LV2_HPP

#pragma once

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <iostream>

template <typename ITYPE, typename RTYPE>
class BLAS_lv2
{
    public:
        // BLAS-1 using matrix rows/columns as vectors
        static void axpy_vec_mat(const ITYPE cols, const ITYPE rowIdx, const RTYPE alpha, const RTYPE* A, RTYPE* x);
        static void dot_vec_mat(const ITYPE cols, const ITYPE rowIdx, const RTYPE* A, const RTYPE* x, double* result);
};

#endif //! BLAS_LV2_HPP