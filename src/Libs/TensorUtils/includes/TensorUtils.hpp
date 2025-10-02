#ifndef TENSORUTILS_HPP
#define TENSORUTILS_HPP

#pragma once

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cmath>

template <typename ITYPE, typename RTYPE>
class TensorUtils
{
    public:
        static void axpy(const ITYPE size, const RTYPE alpha, const RTYPE* x, RTYPE* y);
        static void scale(const ITYPE size, const RTYPE alpha, RTYPE* x);
        static void dot_product(const ITYPE size, const RTYPE* x, const RTYPE* y, RTYPE* result);
        static void multiply_entries(const ITYPE size, const RTYPE* x, RTYPE* y);
        static void invert_entries(const ITYPE size, RTYPE* x);
        static void copy_array(const ITYPE size, const RTYPE* x, RTYPE* y);
};

#endif // TENSORUTILS_HPP