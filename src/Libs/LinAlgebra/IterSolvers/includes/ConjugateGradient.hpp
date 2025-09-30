#ifndef CONJUGATEGRADIENT_H
#define CONJUGATEGRADIENT_H

#pragma once

#include "IterSolvers.hpp"

// Conjugate Gradient solver class, child of IterSolvers
template <typename ITYPE, typename RTYPE>
class ConjugateGradient : public IterSolvers<ITYPE, RTYPE>
{
    private:
        RTYPE *p0,    *d_p0;    // Search dir.
        RTYPE *alpha, *d_alpha; // Alpha param
        RTYPE *beta,  *d_beta;  // Beta param
    public:
        // Empty constructor, calls parent empty constructor
        ConjugateGradient();

        // Param constructor, calls parent param constructor
        ConjugateGradient(ITYPE arrSize, ITYPE maxIters, double tol);

        // Destructor, calls parent destructor
        ~ConjugateGradient();

};

#endif