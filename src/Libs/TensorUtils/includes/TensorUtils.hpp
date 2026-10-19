/**
 * @file TensorUtils.hpp
 * @author Lucas Gasparino (lucas.gasparino3110@gmail.com)
 * @brief Set of utility functions for performing typical tensor operations.
 * @details This class provides static methods for common tensor operations
 *          such as AXPY, scaling, dot product, element-wise multiplication,
 *          and more. The templating allows for flexibility in the choice of
 *          integer and real number types. Since this is intented for CPU
 *          execution, no half precision support is provided.
 *          Other operations can be added as required.
 * @version 0.1
 * @date 2025-10-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef TENSORUTILS_HPP
#define TENSORUTILS_HPP

#pragma once

#include "BLAS_lv1.hpp"
#include "BLAS_lv2.hpp"

// Declaration of TensorUtils class
// Template types:
// ITYPE - Integer type for sizes and indices (e.g., int32_t, int64_t)
// RTYPE - Real number type for tensor values (float, double)
template <typename ITYPE, typename RTYPE>
class TensorUtils : public BLAS_lv1<ITYPE, RTYPE>, public BLAS_lv2<ITYPE, RTYPE>
{
};

#endif // TENSORUTILS_HPP