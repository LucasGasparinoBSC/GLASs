#include "unittKernel.cuh"

__global__ void tridiagdiagMatVec_32(const float* c, const float *d, const float *e, const float* x_in, float* x_out, const uint32_t N) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        x_out[gid] = d[gid] * x_in[gid];
        if (gid > 0) {
            x_out[gid] += c[gid - 1] * x_in[gid - 1];
        }
        if (gid < N - 1) {
            x_out[gid] += e[gid] * x_in[gid + 1];
        }
        gid += blockDim.x * gridDim.x;
    }
}

/*
void runSolver_32(uint32_t nrows, float* A, ConjugateGradient<uint32_t, float>& solver) {
    uint32_t blockSize = 256;
    uint32_t nBlocks = (nrows + blockSize - 1) / blockSize;
    nBlocks = std::min(nBlocks, 10240u);

    // Lambda function for matrix-vector product
    auto MatVec = [=] __host__ (const float* x_in, float* x_out) {
        diagMatVec_32<<<nBlocks, blockSize>>>(A, x_in, x_out, nrows);
        cudaStreamSynchronize(0);
    };

    // Solve the system
    solver.cgSolver(MatVec);
}

void runSolver_64(uint32_t nrows, double* A, ConjugateGradient<uint32_t, double>& solver) {
    uint32_t blockSize = 256;
    uint32_t nBlocks = (nrows + blockSize - 1) / blockSize;
    nBlocks = std::min(nBlocks, 10240u);

    // Lambda function for matrix-vector product
    auto MatVec = [=] __host__ (const double* x_in, double* x_out) {
        diagMatVec_64<<<nBlocks, blockSize>>>(A, x_in, x_out, nrows);
        cudaStreamSynchronize(0);
    };

    // Solve the system
    solver.cgSolver(MatVec);
}
*/