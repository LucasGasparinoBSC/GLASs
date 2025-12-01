#include "unittKernel.cuh"

template<typename ITYPE, typename RTYPE>
__global__ void modify_xout(const ITYPE prank, const ITYPE nranks, const RTYPE* ldata, const RTYPE* rdata, const RTYPE* d_c, const RTYPE* d_e, RTYPE* x_out, const ITYPE N) {
    if (prank == 0) {
        x_out[N-1] += d_e[N-1] * rdata[0];
    } else if (prank == nranks - 1) {
        x_out[0] += d_c[0] * ldata[0];
    } else {
        x_out[0] += d_c[0] * ldata[0];
        x_out[N-1] += d_e[N-1] * rdata[0];
    }
}

template __global__ void modify_xout<uint32_t,float>(const uint32_t prank, const uint32_t nranks, const float* ldata, const float* rdata, const float* d_c, const float* d_e, float* x_out, const uint32_t N);
template __global__ void modify_xout<uint64_t,float>(const uint64_t prank, const uint64_t nranks, const float* ldata, const float* rdata, const float* d_c, const float* d_e, float* x_out, const uint64_t N);
template __global__ void modify_xout<uint32_t,double>(const uint32_t prank, const uint32_t nranks, const double* ldata, const double* rdata, const double* d_c, const double* d_e, double* x_out, const uint32_t N);
template __global__ void modify_xout<uint64_t,double>(const uint64_t prank, const uint64_t nranks, const double* ldata, const double* rdata, const double* d_c, const double* d_e, double* x_out, const uint64_t N);
template __global__ void modify_xout<uint32_t,__nv_bfloat16>(const uint32_t prank, const uint32_t nranks, const __nv_bfloat16* ldata, const __nv_bfloat16* rdata, const __nv_bfloat16* d_c, const __nv_bfloat16* d_e, __nv_bfloat16* x_out, const uint32_t N);
template __global__ void modify_xout<uint64_t,__nv_bfloat16>(const uint64_t prank, const uint64_t nranks, const __nv_bfloat16* ldata, const __nv_bfloat16* rdata, const __nv_bfloat16* d_c, const __nv_bfloat16* d_e, __nv_bfloat16* x_out, const uint64_t N);


__global__ void tridiagMatVec_16(const __nv_bfloat16* c, const __nv_bfloat16 *d, const __nv_bfloat16 *e, const __nv_bfloat16* x_in, __nv_bfloat16* x_out, const uint32_t N) {
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

__global__ void diagPrecond_16(const __nv_bfloat16* d_d, const __nv_bfloat16* r_in, __nv_bfloat16* r_out, const uint32_t N) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < N) {
        r_out[gid] = r_in[gid] / d_d[gid];
        gid += blockDim.x * gridDim.x;
    }
}

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

void runSolver_16(Comm_Utils commObj,
                  uint32_t nrows,
                  const __nv_bfloat16* c_d,
                  const __nv_bfloat16* d_d,
                  const __nv_bfloat16* e_d,
                  __nv_bfloat16* ldata,
                  __nv_bfloat16* rdata,
                  ConjugateGradient<uint32_t, __nv_bfloat16>& solver
) {
    uint32_t blockSize = 256;
    uint32_t nBlocks = (nrows + blockSize - 1) / blockSize;
    nBlocks = std::min(nBlocks, 10240u);

    // Lambda function for matrix-vector product
    int commSize = commObj.getLibSize();
    auto MatVec = [=] __host__ (const __nv_bfloat16* x_in, __nv_bfloat16* x_out) {
        tridiagMatVec_16<<<nBlocks, blockSize>>>(c_d, d_d, e_d, x_in, x_out, nrows);
        cudaStreamSynchronize(0);
        if (commObj.isParallel && commSize > 1) {
            // Halo exchange
            halo_exchange<uint32_t, __nv_bfloat16>(commObj, ldata, rdata, nrows, c_d, e_d, x_in, x_out);
        }
    };

    auto Precond = [=] __host__ (const __nv_bfloat16* r_in, __nv_bfloat16* r_out) {
        diagPrecond_16<<<nBlocks, blockSize>>>(d_d, r_in, r_out, nrows);
        //cudaStreamSynchronize(0);
    };

    // Solve the system
    solver.fpcgSolver(MatVec, Precond);
}

void runSolver_32(Comm_Utils commObj,
                  uint32_t nrows,
                  const float* c_d,
                  const float* d_d,
                  const float* e_d,
                  float* ldata,
                  float* rdata,
                  ConjugateGradient<uint32_t, float>& solver
) {
    uint32_t blockSize = 256;
    uint32_t nBlocks = (nrows + blockSize - 1) / blockSize;
    nBlocks = std::min(nBlocks, 10240u);

    // Lambda function for matrix-vector product
    int commSize = commObj.getLibSize();
    auto MatVec = [=] __host__ (const float* x_in, float* x_out) {
        tridiagMatVec_32<<<nBlocks, blockSize>>>(c_d, d_d, e_d, x_in, x_out, nrows);
        cudaStreamSynchronize(0);
        if (commObj.isParallel && commSize > 1) {
            // Halo exchange
            PUSH_RANGE("halo_exchange", 0)
            halo_exchange<uint32_t, float>(commObj, ldata, rdata, nrows, c_d, e_d, x_in, x_out);
            POP_RANGE
        }
    };

    auto Precond = [=] __host__ (const float* r_in, float* r_out) {
        diagPrecond_32<<<nBlocks, blockSize>>>(d_d, r_in, r_out, nrows);
        //cudaStreamSynchronize(0);
    };

    // Solve the system
    solver.fpcgSolver(MatVec, Precond);
}

void runSolver_64(Comm_Utils commObj,
                  uint32_t nrows,
                  const double* c_d,
                  const double* d_d,
                  const double* e_d,
                  double* ldata,
                  double* rdata,
                  ConjugateGradient<uint32_t, double>& solver
) {
    uint32_t blockSize = 256;
    uint32_t nBlocks = (nrows + blockSize - 1) / blockSize;
    nBlocks = std::min(nBlocks, 10240u);

    // Lambda function for matrix-vector product
    int commSize = commObj.getLibSize();
    auto MatVec = [=] __host__ (const double* x_in, double* x_out) {
        tridiagMatVec_64<<<nBlocks, blockSize>>>(c_d, d_d, e_d, x_in, x_out, nrows);
        cudaStreamSynchronize(0);
        if (commObj.isParallel && commSize > 1) {
            // Halo exchange
            halo_exchange<uint32_t, double>(commObj, ldata, rdata, nrows, c_d, e_d, x_in, x_out);
        }
    };

    auto Precond = [=] __host__ (const double* r_in, double* r_out) {
        diagPrecond_64<<<nBlocks, blockSize>>>(d_d, r_in, r_out, nrows);
        //cudaStreamSynchronize(0);
    };

    // Solve the system
    solver.fpcgSolver(MatVec, Precond);
}