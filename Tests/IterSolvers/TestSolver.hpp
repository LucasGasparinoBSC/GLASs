#ifndef TESTSOLVER_HPP
#define TESTSOLVER_HPP

#pragma once

#include "IterSolvers.hpp"

template <typename ITYPE, typename RTYPE>
class TestSolver : public IterSolvers<ITYPE,RTYPE>
{
    private:
    public:
        // Constructor
        TestSolver(MPI_Comm &c_comm, ITYPE arrSize, ITYPE maxIters, double tol);

        // Destructor
        ~TestSolver();

        // CheckSetup
        void CheckSetup();

        // Dummy solver
        void dummySolver();
};

#endif //! TESTSOLVER_HPP