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
void halo_exchange(Comm_Utils commObj, const uint32_t nLocal, RTYPE *c, RTYPE *e, const RTYPE *x_in, RTYPE *x_out);

#endif // HALO_OPS_HPP