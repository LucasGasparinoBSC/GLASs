#ifndef HOSTSIDE_HPP
#define HOSTSIDE_HPP

#pragma once

#include "ConjugateGradient.hpp"

template <typename ITYPE, typename RTYPE>
class HostSide
{
    private:
    public:
        static void setLocalSizes(const ITYPE globalSize, const int rank, const int nranks, ITYPE& localSize, ITYPE* sizesPerRank);
        static void generate_matrix(const ITYPE N, RTYPE *c, RTYPE *d, RTYPE *e);
        static void generate_inicond(const ITYPE N, RTYPE *x0);
        static void generate_rhs(const ITYPE N, RTYPE *b, const int rank);
};

#endif //! HOSTSIDE_HPP