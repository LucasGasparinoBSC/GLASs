#include "IterSolvers.hpp"
#include "CUDA_Utils.cuh"

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

        // Test setup vars
        void test_setup() {
            int passed = 0;
            #pragma acc parallel loop deviceptr(this->x0, this->b)
            for (uint32_t i = 0; i < this->arrSize; i++) {
                this->x0[i] += static_cast<float>(1.0);
                this->b[i] += static_cast<float>(2.0);
            }

            #pragma acc parallel loop reduction(+:passed)
            for (uint32_t i = 0; i < this->arrSize; i++) {
                if (this->x0[i] != 1.0f) {
                    passed += 1;
                }
                if (this->b[i] != 3.0f) {
                    passed += 1;
                }
            }
            if (passed == 0) {
                std::cout << "--| IterSolvers: setup test passed!" << std::endl;
            } else {
                std::cerr << "--| IterSolvers: setup test failed!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }

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

    #ifdef USE_GPU
        float* dx0;
        cudaMalloc(&dx0, narr * sizeof(float));
        cudaMemcpy(dx0, x0, narr * sizeof(float), cudaMemcpyHostToDevice);
        float* db;
        cudaMalloc(&db, narr * sizeof(float));
        cudaMemcpy(db, b, narr * sizeof(float), cudaMemcpyHostToDevice);
        testSolver.setup(dx0, db);
    #else
        testSolver.setup(x0, b);
    #endif
    testSolver.test_setup();

    // OpenACC test
    #ifdef USE_GPU
        float *x1 = (float *)calloc(narr, sizeof(float));
        float *rhs = (float *)calloc(narr, sizeof(float));
        #pragma acc enter data create(x1[0:narr], rhs[0:narr])
        #pragma acc parallel loop
        for (uint32_t i = 0; i < narr; i++) {
            x1[i] = 0.0f;
            rhs[i] = 1.0f;
        }
        #pragma acc host_data use_device(x1, rhs)
        {
            testSolver.setup(x1, rhs);
        }
        testSolver.test_setup();
    #endif
    return 0;
}