#include "Comm_Utils.hpp"
#include "TensorUtils.hpp"

int main()
{
    MPI_Init(NULL, NULL);

    int nranks, prank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    // Basic test sizes for UT^T matrix
    // NOTE: this mimicks the modified upper-hessenberg matrix arising in the GMRES sovler
    const uint32_t lda = 11;         // Leading dimension (physical rows per column, e.g., MITERS + 1)
    const uint32_t num_cols = lda-1; // Number of active columns you want to test (e.g., kIter + 1)

    // Allocate column-major storage (calloc automatically gives you zeros everywhere else)
    double *UT = (double *)calloc(lda * num_cols, sizeof(double));

    // Fill the upper-triangular part in column-major order
    for (uint32_t icol = 0; icol < num_cols; ++icol)
    {
        // In an upper-triangular system, rows go from 0 up to the current column index (icol)
        for (uint32_t irow = 0; irow <= icol; ++irow)
        {
            // Diagonal = 4, others = 1
            if (irow == icol)
            {
                UT[irow + icol * lda] = static_cast<double>(4);
            }
            else
            {
                UT[irow + icol * lda] = static_cast<double>(1);
            }
        }
    }

    // Print the UT matrix
    for (uint32_t irow = 0; irow < lda; ++irow)
    {
        for (uint32_t icol = 0; icol < num_cols; ++icol)
        {
            printf("%f ", UT[irow + icol * lda]);
        }
        printf("\n");
    }

    // Right-hand side vector b
    // NOTE: again, the test should mimick the GMRES input,
    // so b is allocated with the same leading dimension as UT, but only the first num_cols entries are used.
    double *b = (double *)calloc(lda, sizeof(double));
    double one_four = static_cast<double>(1) / static_cast<double>(4);
    for (uint32_t i = 0; i < num_cols; ++i)
    {
        b[i] = static_cast<double>(1) + one_four * static_cast<double>(num_cols-i-1);
        printf("b[%u] = %f\n", i, b[i]);
    }

    // Solution vector x
    double *x = (double *)calloc(lda, sizeof(double));

    // Call the solver
    TensorUtils<uint32_t, double>::UTRSV((num_cols - 1), lda, UT, b, x);

    for (uint32_t i = 0; i < lda; ++i)
    {
        printf("x[%u] = %f\n", i, x[i]);
    }

    // Verify
    for (uint32_t i = 0; i < num_cols; ++i) {
        if (std::abs(x[i] - one_four) > 1e-10) {
            printf("Test failed: x[%u] = %f, expected %f\n", i, x[i], one_four);
            MPI_Finalize();
            return 1;
        }
    }
    if (x[lda-1] != 0.0) {
        printf("Test failed: x[%u] = %f, expected 0.0\n", lda-1, x[lda-1]);
        MPI_Finalize();
        return 1;
    }

    printf("Test passed: all values in x are as expected.\n");

    MPI_Finalize();
    return 0;
}