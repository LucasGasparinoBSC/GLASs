#include "unittKernel.cuh"

__global__ void tridiagMatVec_32(const float* c, const float *d, const float *e, const float* x_in, float* x_out, const uint32_t N) {
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

__global__ void diagPrecond_32(const float* d_d, const float* r_in, float* r_out, const uint32_t N) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        r_out[gid] = r_in[gid] / d_d[gid];
        gid += blockDim.x * gridDim.x;
    }
}

__global__ void tridiagMatVec_64(const double* c, const double *d, const double *e, const double* x_in, double* x_out, const uint32_t N) {
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

__global__ void diagPrecond_64(const double* d_d, const double* r_in, double* r_out, const uint32_t N) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        r_out[gid] = r_in[gid] / d_d[gid];
        gid += blockDim.x * gridDim.x;
    }
}

void runSolver_32(uint32_t nrows, const float* c_d, const float* d_d, const float* e_d, ConjugateGradient<uint32_t, float>& solver) {
    uint32_t blockSize = 256;
    uint32_t nBlocks = (nrows + blockSize - 1) / blockSize;
    nBlocks = std::min(nBlocks, 10240u);

    // Lambda function for matrix-vector product
    auto MatVec = [=] __host__ (const float* x_in, float* x_out) {
        tridiagMatVec_32<<<nBlocks, blockSize>>>(c_d, d_d, e_d, x_in, x_out, nrows);
        cudaStreamSynchronize(0);
    };

    // Solve the system
    solver.cgSolver(MatVec);
}

void runSolver_64(uint32_t nrows, const double* c_d, const double* d_d, const double* e_d, ConjugateGradient<uint32_t, double>& solver) {
    uint32_t blockSize = 256;
    uint32_t nBlocks = (nrows + blockSize - 1) / blockSize;
    nBlocks = std::min(nBlocks, 10240u);

    // Lambda function for matrix-vector product
    auto MatVec = [=] __host__ (const double* x_in, double* x_out) {
        tridiagMatVec_64<<<nBlocks, blockSize>>>(c_d, d_d, e_d, x_in, x_out, nrows);
        cudaStreamSynchronize(0);
    };

    // Solve the system
    solver.cgSolver(MatVec);
}