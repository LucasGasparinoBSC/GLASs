#include "Comm_Utils.hpp"
#include "TensorUtils.hpp"

int main()
{
    MPI_Init(NULL, NULL);

    int nranks, prank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm new_comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &new_comm);

    // Basic test sizes for UT^T matrix
    // NOTE: this mimicks the modified upper-hessenberg matrix arising in the GMRES sovler
    const uint32_t lda = 10001;           // Leading dimension (physical rows per column, e.g., MITERS + 1)
    const uint32_t num_cols = lda - 1; // Number of active columns you want to test (e.g., kIter + 1)

    // Allocate column-major storage (calloc automatically gives you zeros everywhere else)
    float *UT = (float *)calloc(lda * num_cols, sizeof(float));

    // Fill the upper-triangular part in column-major order
    for (uint32_t icol = 0; icol < num_cols; ++icol)
    {
        // In an upper-triangular system, rows go from 0 up to the current column index (icol)
        for (uint32_t irow = 0; irow <= icol; ++irow)
        {
            // Diagonal = 4, others = 1
            if (irow == icol)
            {
                UT[irow + icol * lda] = static_cast<float>(4);
            }
            else
            {
                UT[irow + icol * lda] = static_cast<float>(1);
            }
        }
    }

    // Right-hand side vector b
    // NOTE: again, the test should mimick the GMRES input,
    // so b is allocated with the same leading dimension as UT, but only the first num_cols entries are used.
    float *b = (float *)calloc(lda, sizeof(float));
    float one_four = static_cast<float>(1) / static_cast<float>(4);
    for (uint32_t i = 0; i < num_cols; ++i)
    {
        b[i] = static_cast<float>(1) + one_four * static_cast<float>(num_cols - i - 1);
    }

    // Solution vector x
    float *x = (float *)calloc(lda, sizeof(float));

    // Call and time the solver
    // NOTE: use the utility from Comm_Utils to time the UTRSV call
    const uint32_t nruns = 1000;
    Comm_Utils commUtils(new_comm);
    double avg_time = 0.0;
    for (uint32_t run = 0; run < nruns; ++run)
    {
        // Zero the solution vector x before each run
        std::fill(x, x + lda, 0.0f);
        double curr_time = commUtils.timeFunction([&] {
            TensorUtils<uint32_t, float>::UTRSV((num_cols - 1), lda, UT, b, x);
        });
        avg_time += curr_time;
        printf("Run %u: UTRSV time = %f seconds\n", run + 1, curr_time);
    }
    avg_time /= static_cast<double>(nruns);
    printf("Average time for UTRSV over %u runs: %f seconds\n", nruns, avg_time);

    MPI_Finalize();
    return 0;
}