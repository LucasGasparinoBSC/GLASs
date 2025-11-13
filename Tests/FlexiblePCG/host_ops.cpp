#include "host_ops.hpp"

void setLocalSizes(const uint32_t globalSize, const int rank, const int nranks, uint32_t& localSize, uint32_t *sizesPerRank) {
    double ratio = static_cast<double>(globalSize) / static_cast<double>(nranks);
    uint32_t aux = static_cast<uint32_t>(ratio);
    if ((ratio - static_cast<double>(aux)) > 0.5)
        aux++;
    for (int i = 0; i < nranks; i++) sizesPerRank[i] = aux;
    sizesPerRank[nranks - 1] = globalSize - (nranks - 1) * aux;
    localSize = sizesPerRank[rank];
}

template <typename ITYPE, typename RTYPE>
void generate_matrix(const ITYPE N, RTYPE *c, RTYPE *d, RTYPE *e) {
    for (ITYPE i = 0; i < N; i++) {
        c[i] = static_cast<RTYPE>(1);
        d[i] = static_cast<RTYPE>(4);
        e[i] = static_cast<RTYPE>(1);
    }
}

template void generate_matrix<uint32_t, float>(const uint32_t N, float *c, float *d, float *e);
template void generate_matrix<uint64_t, float>(const uint64_t N, float *c, float *d, float *e);
template void generate_matrix<uint32_t, double>(const uint32_t N, double *c, double *d, double *e);
template void generate_matrix<uint64_t, double>(const uint64_t N, double *c, double *d, double *e);

template <typename ITYPE, typename RTYPE>
void host_diagPrecond(const RTYPE *d, const RTYPE *r_in, RTYPE *r_out, const ITYPE N) {
    for (ITYPE i = 0; i < N; i++) {
        r_out[i] = r_in[i] / d[i];
    }
}

template void host_diagPrecond<uint32_t, float>(const float *d, const float *r_in, float *r_out, const uint32_t N);
template void host_diagPrecond<uint64_t, float>(const float *d, const float *r_in, float *r_out, const uint64_t N);
template void host_diagPrecond<uint32_t, double>(const double *d, const double *r_in, double *r_out, const uint32_t N);
template void host_diagPrecond<uint64_t, double>(const double *d, const double *r_in, double *r_out, const uint64_t N);

template <typename ITYPE, typename RTYPE>
void host_tridiagMatVec(const RTYPE *c, const RTYPE *d, const RTYPE *e, const RTYPE *x_in, RTYPE *x_out, const ITYPE N) {
    // entries 1 to N-2 are generic: c[i-1]*x[i-1] + d[i]*x[i] + e[i]*x[i+1]
    for (uint32_t i = 1; i < N - 1; i++)
    {
        x_out[i] = c[i - 1] * x_in[i - 1] + d[i] * x_in[i] + e[i] * x_in[i + 1];
    }

    // First row
    x_out[0] = d[0] * x_in[0] + e[0] * x_in[1];

    // Last row
    x_out[N - 1] = c[N - 1] * x_in[N - 2] + d[N - 1] * x_in[N - 1];
}

template void host_tridiagMatVec<uint32_t, float>(const float *c, const float *d, const float *e, const float *x_in, float *x_out, const uint32_t N);
template void host_tridiagMatVec<uint64_t, float>(const float *c, const float *d, const float *e, const float *x_in, float *x_out, const uint64_t N);
template void host_tridiagMatVec<uint32_t, double>(const double *c, const double *d, const double *e, const double *x_in, double *x_out, const uint32_t N);
template void host_tridiagMatVec<uint64_t, double>(const double *c, const double *d, const double *e, const double *x_in, double *x_out, const uint64_t N);

// Generate initial condition array with small random values [-0.001, 0.001]
template <typename ITYPE, typename RTYPE>
void generate_inicond(const ITYPE N, RTYPE *x0) {
    RTYPE scale = static_cast<RTYPE>(0.002);
    RTYPE offset = static_cast<RTYPE>(0.001);
    for (ITYPE i = 0; i < N; i++) {
        x0[i] = static_cast<RTYPE>(rand()) / static_cast<RTYPE>(RAND_MAX) * scale - offset;
    }
}

template void generate_inicond<uint32_t, float>(const uint32_t N, float *x0);
template void generate_inicond<uint64_t, float>(const uint64_t N, float *x0);
template void generate_inicond<uint32_t, double>(const uint32_t N, double *x0);
template void generate_inicond<uint64_t, double>(const uint64_t N, double *x0);