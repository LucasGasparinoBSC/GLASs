#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include "CUDA_Utils.cuh"

int main()
{
    PUSH_RANGE("Generate data", 0);
    uint32_t narr = static_cast<uint32_t>(1e7);
    printf("Array size: %e\n", static_cast<double>(narr));
    float *x1 = (float *)calloc(narr, sizeof(float));
    float *x2 = (float *)calloc(narr, sizeof(float));
    for (uint32_t i = 0; i < narr; i++)
    {
        x1[i] = static_cast<float>(i + 1);
        x2[i] = static_cast<float>(2 * i);
    }
    POP_RANGE

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
    dim3 grid(nblocks, 1, 1);
    dim3 block(static_cast<uint32_t>(256), 1, 1);
    printf("Launching with %d blocks of %d threads\n", nblocks, static_cast<uint32_t>(TILE_SIZE));

    // Puto dot product
    float *dres;
    float hres;
    CUDA_CHECK(cudaMalloc((void **)&dres, sizeof(float)));

    PUSH_RANGE("Dot product", 0);
    CUDA_CHECK(cudaMemset(dres, 0, sizeof(float)));
    launchKernel(dot_product<uint32_t, float>, grid, block, dx1, dx2, dres, narr);
    POP_RANGE

    PUSH_RANGE("Verify dot product", 0);
    CUDA_CHECK(cudaMemcpy(&hres, dres, sizeof(float), cudaMemcpyDeviceToHost));
    printf("dot product result: %e\n", hres);
    double verif = 0.0;
    for (uint32_t i = 0; i < narr; i++)
    {
        verif += static_cast<double>(x1[i]) * static_cast<double>(x2[i]);
    }
    printf("verification: %e\n", static_cast<float>(verif));
    double diffRel = ((double)hres - verif) / verif;
    printf("relative difference: %e %\n", 100.0 * diffRel);
    if (std::abs(diffRel) > 1e-2)
    {
        printf("Error: relative difference too high!\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("Dot product OK!\n");
    }
    POP_RANGE

    // AXPY
    const float a = static_cast<float>(1);
    PUSH_RANGE("AXPY", 0);
    launchKernel(axpy<uint32_t, float>, grid, block, a, dx1, dx2, narr);
    POP_RANGE
    PUSH_RANGE("Copy x2 D2H", 0);
    CUDA_CHECK(cudaMemcpy(x2, dx2, narr * sizeof(float), cudaMemcpyDeviceToHost));
    POP_RANGE
    PUSH_RANGE("Verify AXPY", 0);
    float diff = 0.0;
    float verif_axpy = 0.0;
    for (uint32_t i = 0; i < narr; i++)
    {
        verif_axpy = static_cast<float>(x1[i]) + static_cast<float>(2 * i);
        diff = (static_cast<float>(x2[i]) - verif_axpy) / verif_axpy;
        diff = 100.0f * std::abs(diff);
        if (diff > static_cast<float>(1))
        {
            printf("Error at index %d: x2 = %f, expected %f, rel diff %e %%\n", i, static_cast<float>(x2[i]), verif_axpy, diff);
            exit(EXIT_FAILURE);
        }
    }
    printf("AXPY OK!\n");
    printf("x2[narr-1] = %f\n", static_cast<float>(x2[narr - 1]));
    POP_RANGE

    return 0;
}