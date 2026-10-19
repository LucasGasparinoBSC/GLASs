#ifndef BLAS_LV1_HPP
#define BLAS_LV1_HPP

#pragma once

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <iostream>

template <typename ITYPE, typename RTYPE>
class BLAS_lv1
{
    public:
        // axpy: Performs the operation y = alpha * x + y
        static void axpy(const ITYPE size, const RTYPE alpha, const RTYPE* x, RTYPE* y);

        // scale: Scales the vector x by alpha (x = alpha * x)
        static void scale(const ITYPE size, const RTYPE alpha, RTYPE* x);

        // dot_product: Computes the dot product of vectors x and y
        static void dot_product(const ITYPE size, const RTYPE* x, const RTYPE* y, double* result);

        // guided dot_product: Computes the dot product of vectors x and y, following a specified access pattern
        static void guided_dot_product(const ITYPE sizeList, const ITYPE *listEntries, const RTYPE *x, const RTYPE *y, double *result);

        // multiply_entries: Performs element-wise multiplication of vectors x and y (y = x * y)
        static void multiply_entries(const ITYPE size, const RTYPE* x, RTYPE* y);

        // invert_entries: Inverts each entry of vector x (x = 1 / x)
        static void invert_entries(const ITYPE size, RTYPE* x);

        // copy_array: Copies the contents of array x to array y
        static void copy_array(const ITYPE size, const RTYPE* x, RTYPE* y);

};

#endif //! BLAS_LV1_HPP