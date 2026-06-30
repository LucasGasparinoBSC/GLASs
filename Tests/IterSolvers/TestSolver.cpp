#include "TestSolver.hpp"

template <typename ITYPE, typename RTYPE>
TestSolver<ITYPE, RTYPE>::TestSolver(MPI_Comm &c_comm, ITYPE arrSize, ITYPE maxIters, double tol) : IterSolvers<ITYPE, RTYPE>(c_comm, arrSize, maxIters, tol)
{
    PUSH_RANGE("TestSolver::TestSolver",4);
    if (this->flag_planned == false) {
        std::cerr << "IterSolvers planner failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
    POP_RANGE();
}

template <typename ITYPE, typename RTYPE>
TestSolver<ITYPE, RTYPE>::~TestSolver() {
    PUSH_RANGE("TestSolver::~TestSolver",4);
    POP_RANGE();
}

template <typename ITYPE, typename RTYPE>
void TestSolver<ITYPE, RTYPE>::CheckSetup() {
    PUSH_RANGE("TestSolver::CheckSetup",4);
    // Check flag
    if (this->flag_setup == false) {
        std::cerr << "IterSolvers setup failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Check referenced arrays
    #if defined (USE_GPU)
        if (this->d_x0 == nullptr) {
            std::cerr << "IterSolvers setup failed: d_x0 is null!" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (this->d_b == nullptr) {
            std::cerr << "IterSolvers setup failed: d_b is null!" << std::endl;
            exit(EXIT_FAILURE);
        }
    #else
        if (this->x0 == nullptr) {
            std::cerr << "IterSolvers setup failed: x0 is null!" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (this->b == nullptr) {
            std::cerr << "IterSolvers setup failed: b is null!" << std::endl;
            exit(EXIT_FAILURE);
        }
    #endif
    POP_RANGE();
}

template <typename ITYPE, typename RTYPE>
void TestSolver<ITYPE, RTYPE>::dummySolver() {
    PUSH_RANGE("TestSolver::dummySolver",4);
    RTYPE a = static_cast<RTYPE>(-1);
    #if defined (USE_GPU)
        DeviceUtils::launchKernel(copy_array<ITYPE,RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, this->d_b, this->d_x_sol, this->arrSize);
        DeviceUtils::launchKernel(axpy<ITYPE,RTYPE>, this->kernelGrid, this->kernelBlock, this->kernelStream, a, this->d_x0, this->d_x_sol, this->arrSize);
    #else
        TensorUtils<ITYPE, RTYPE>::copy_array(this->arrSize, this->b, this->x_sol);
        TensorUtils<ITYPE, RTYPE>::axpy(this->arrSize, a, this->x0, this->x_sol);
    #endif
    POP_RANGE();
}

template class TestSolver<uint32_t, float>;
template class TestSolver<uint64_t, float>;
template class TestSolver<uint32_t, double>;
template class TestSolver<uint64_t, double>;
#if defined(USE_GPU)
    template class TestSolver<uint32_t, DeviceUtils::bf16>;
    template class TestSolver<uint64_t, DeviceUtils::bf16>;
#endif
