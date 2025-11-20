#ifndef HALO_OPS_HPP
#define HALO_OPS_HPP

#pragma once

#include <cstdint>
#include <cstdlib>
#include "Comm_Utils.hpp"
#ifdef USE_GPU
    #include "CUDA_Utils.cuh"
#endif

template <typename ITYPE, typename RTYPE>
void halo_exchange(Comm_Utils commObj, RTYPE* left_data, RTYPE* right_data, const uint32_t nLocal, const RTYPE *c, const RTYPE *e, const RTYPE *x_in, RTYPE *x_out);

#endif // HALO_OPS_HPP