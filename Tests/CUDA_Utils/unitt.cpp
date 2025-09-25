#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include "CUDA_Utils.cuh"

int main() {
    uint32_t narr = static_cast<uint32_t>(1e7);
    float *x1 = (float *)calloc(narr, sizeof(float));
    float *x2 = (float *)calloc(narr, sizeof(float));
    for (uint32_t i = 0; i < narr; i++) {
        x1[i] = static_cast<float>(i+1);
        x2[i] = static_cast<float>(2*i);
    }

    float *dx1;
    float *dx2;
    PUSH_RANGE("Memory allocation", 0);
    CUDA_CHECK(cudaMalloc((void **)&dx1, narr * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&dx2, narr * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx1, x1, narr * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx2, x2, narr * sizeof(float), cudaMemcpyHostToDevice));
    POP_RANGE

    // Launch params
    uint32_t nblocks = (narr + static_cast<uint32_t>(TILE_SIZE) - 1) / static_cast<uint32_t>(TILE_SIZE);
    nblocks = std::min(nblocks, static_cast<uint32_t>(MAX_BLOCKS));
    dim3 grid(nblocks,1,1);
    dim3 block(static_cast<uint32_t>(256),1,1);
    printf("Launching with %d blocks of %d threads\n", nblocks, static_cast<uint32_t>(TILE_SIZE));

    // Puto dot product
    float *dres;
    float hres;
    CUDA_CHECK(cudaMalloc((void **)&dres, sizeof(float)));

    CUDA_CHECK(cudaMemset(dres, 0, sizeof(float)));
    launchKernel(dot_product<uint32_t, float>, grid, block, dx1, dx2, dres, narr);

    CUDA_CHECK(cudaMemcpy(&hres, dres, sizeof(float), cudaMemcpyDeviceToHost));
    printf("dot product result: %e\n", hres);

    double verif = 0.0;
    for (uint32_t i = 0; i < narr; i++) {
        verif += static_cast<double>(x1[i]) * static_cast<double>(x2[i]);
    }
    printf("verification: %e\n", static_cast<float>(verif));
    double diffRel = ((double)hres - verif) / verif;
    printf("relative difference: %e %\n", 100.0 * diffRel);

    // AXPY
    const float a = static_cast<float>(1);
    PUSH_RANGE("AXPY", 0);
    launchKernel(axpy<uint32_t, float>, grid, block, a, dx1, dx2, narr);
    POP_RANGE
    PUSH_RANGE("Copy x2 D2H", 0);
    CUDA_CHECK(cudaMemcpy(x2, dx2, narr * sizeof(float), cudaMemcpyDeviceToHost));
    POP_RANGE
    PUSH_RANGE("Verify AXPY", 0);
    for (uint32_t i = 0; i < narr; i++) {
        if (x2[i] != x1[i] + 2*i) {
            printf("Error in AXPY at index %d: %f != %f\n", i, x2[i], x1[i] + 2*i);
            exit(EXIT_FAILURE);
        }
    }
    printf("AXPY OK!\n");
    printf("x2[narr-1] = %f\n", x2[narr-1]);
    POP_RANGE

    return 0;
}