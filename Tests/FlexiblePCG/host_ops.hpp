#ifndef HOST_OPS_HPP
#define HOST_OPS_HPP

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <cstring>

void setLocalSizes(const uint32_t globalSize, const int rank, const int nranks, uint32_t& localSize, uint32_t* sizesPerRank);

template <typename ITYPE, typename RTYPE>
void generate_matrix(const ITYPE N, RTYPE* c, RTYPE* d, RTYPE* e);

template <typename ITYPE, typename RTYPE>
void host_diagPrecond(const RTYPE* d, const RTYPE* r_in, RTYPE* r_out, const ITYPE N);

template <typename ITYPE, typename RTYPE>
void host_tridiagMatVec(const RTYPE* c, const RTYPE *d, const RTYPE *e, const RTYPE* x_in, RTYPE* x_out, const ITYPE N);

template <typename ITYPE, typename RTYPE>
void generate_inicond(const ITYPE N, RTYPE* x0);

#endif // HOST_OPS_HPP