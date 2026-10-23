#include "Comm_Utils.hpp"
#include "TensorUtils.hpp"

int main()
{
    MPI_Init(NULL, NULL);

    int nranks, prank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    // Basic test sizes for A matrix
    const uint32_t rows = 10;
    const uint32_t cols = 100;

    // Generate x vector of ones
    float* x = (float*)calloc(cols, sizeof(float));
    for (uint32_t i = 0; i < cols; ++i) {
        x[i] = 1.0f;
    }

    // Generate a 10x100 dense matrix.
    // Each row is composed of irow+1 entries so:
    // Row 0: 1.0, 1.0, ..., 1.0
    // Row 1: 2.0, 2.0, ..., 2.0
    // Row 2: 3.0, 3.0, ..., 3.0
    float* A_dense = (float*)calloc(rows * cols, sizeof(float));
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            A_dense[i * cols + j] = static_cast<float>(i + 1);
        }
    }

    // Call GEMV with dense format
    double* y_dense = (double*)calloc(rows, sizeof(double));
    BLAS_lv2<uint32_t, float>::GEMV<0>(x, y_dense, rows, cols, A_dense);

    // Verification: Each entry of y_dense should be equal to (irow+1)*cols
    double expected_value;
    for (uint32_t i = 0; i < rows; ++i) {
        expected_value = static_cast<double>(i + 1) * static_cast<double>(cols);
        if (std::abs(y_dense[i] - expected_value) > 1e-6) {
            std::cerr << "Test failed at row " << i << ": expected " << expected_value
                      << ", got " << y_dense[i] << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    std::cout << "GEMV dense test passed successfully." << std::endl;

    free(x);
    free(A_dense);
    free(y_dense);

    MPI_Finalize();
    return 0;
}