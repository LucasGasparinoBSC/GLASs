#include "IterSolvers.hpp"

// Test child class to test the abstract base class IterSolvers
class TestSolver : public IterSolvers<uint32_t, float>
{
    public:
        // Empty constructor, calls parent empty constructor
        TestSolver() : IterSolvers<uint32_t, float>() {}

        // Param constructor, calls parent param constructor
        TestSolver(uint32_t arrSize, uint32_t maxIters, double tol) : IterSolvers<uint32_t, float>(arrSize, maxIters, tol) {}

        // Destructor, calls parent destructor
        ~TestSolver() {}

};

int main() {
    // Prblem params.
    uint32_t narr = static_cast<uint32_t>(1e6);
    uint32_t mIters = 1000;
    double tol = 1e-6;

    // Instantiate solver
    TestSolver testSolver(narr, mIters, tol);

    // Setup initial condition and RHS
    float *x0 = new float[narr];
    float *b = new float[narr];
    for (uint32_t i = 0; i < narr; i++) {
        x0[i] = 0.0f;
        b[i] = 1.0f;
    }
    testSolver.setup(x0, b);
    return 0;
}