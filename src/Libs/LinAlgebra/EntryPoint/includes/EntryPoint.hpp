/**
 * @file EntryPoint.hpp
 * @author your name (you@domain.com)
 * @brief Abstract class for ensuring internal libs share common attributes, especially the CommUtils object.
 * @version 0.1
 * @date 2025-11-04
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef ENTRYPOINT_HPP
#define ENTRYPOINT_HPP

#pragma once

#ifdef USE_GPU
    #include "CUDA_Utils.cuh"
#else
    #define PUSH_RANGE(name, cid)
    #define POP_RANGE
#endif

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <functional>
#include <algorithm>

#include "Comm_Utils.hpp"
#include "TensorUtils.hpp"

template <typename ITYPE, typename RTYPE>
class EntryPoint
{
protected:
    // Common function type definitions for linear algebra operations
    using MatVecOp = std::function<void(const RTYPE *x_in, RTYPE *x_out)>;
    using PrecondOp = std::function<void(const RTYPE *x_in, RTYPE *x_out)>;
    using ModVecOp = std::function<void(RTYPE *x_inout)>;
    
    // Common communication utilities object
    Comm_Utils entrypoint_comm;

public:
    // Virtual destructor for proper cleanup of derived classes
    virtual ~EntryPoint() = default;
};

#endif // ENTRYPOINT_HPP