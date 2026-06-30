#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include "Kernels_Lv1.cuh"
#include "DeviceMemory.hpp"

int main()
{
    // Base data
    uint32_t N = 1024;
    double a = 2.0;
    double *x_h = (double *)calloc(N, sizeof(double));
    double *y_h = (double *)calloc(N, sizeof(double));
    double *x_d = DeviceMemory<uint32_t, double>::deviceCalloc(N);
    double *y_d = DeviceMemory<uint32_t, double>::deviceCalloc(N);
    DeviceUtils::Stream_t kStream;
    DeviceUtils::StreamCreate(&kStream);

    // AXPY test
    {
        PUSH_RANGE("AXPY Test", 2);
        for (uint32_t i = 0; i < N; ++i)
        {
            x_h[i] = 1.0;
            y_h[i] = 1.0;
        }
        DeviceMemory<uint32_t, double>::copyHostToDevice(N, x_h, x_d);
        DeviceMemory<uint32_t, double>::copyHostToDevice(N, y_h, y_d);
        uint32_t blocks = (N + TILE_SIZE - 1) / TILE_SIZE;
        blocks = std::min(blocks, (uint32_t)MAX_BLOCKS);
        dim3 grid(blocks,1,1);
        dim3 block(TILE_SIZE,1,1);
        DeviceUtils::launchKernel(axpy<uint32_t,double>, grid, block, kStream, a, x_d, y_d, N);
        DeviceUtils::StreamSynchronize(kStream);
        DeviceMemory<uint32_t, double>::copyDeviceToHost(N, y_d, y_h);
        for (uint32_t i = 0; i < N; ++i) {
            if (y_h[i] != 3.0) {
                std::cerr << "Error at index " << i << ": expected 3.0, got " << y_h[i] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        std::cout << "AXPY test passed!" << std::endl;
        POP_RANGE();
    }

    // dot product test
    {
        PUSH_RANGE("Dot Product Test", 2);
        uint32_t N = 1024;
        double *res_h = (double *)calloc(1, sizeof(double));
        double *res_d = DeviceMemory<uint32_t, double>::deviceCalloc(1);
        for (uint32_t i = 0; i < N; ++i)
        {
            x_h[i] = static_cast<double>(i + 1); // 1.0f, 2.0f, ..., N*1.0f
            y_h[i] = 1.0;
        }
        DeviceMemory<uint32_t, double>::copyHostToDevice(1, res_h, res_d);
        DeviceMemory<uint32_t, double>::copyHostToDevice(N, x_h, x_d);
        DeviceMemory<uint32_t, double>::copyHostToDevice(N, y_h, y_d);
        uint32_t blocks = (N + TILE_SIZE - 1) / TILE_SIZE;
        blocks = std::min(blocks, (uint32_t)MAX_BLOCKS);
        dim3 grid(blocks, 1, 1);
        dim3 block(TILE_SIZE, 1, 1);
        DeviceUtils::launchKernel(dot_product<uint32_t, double>, grid, block, kStream, x_d, y_d, res_d, N);
        DeviceUtils::StreamSynchronize(kStream);
        DeviceMemory<uint32_t, double>::copyDeviceToHost(1, res_d, res_h);
        double expected = 0.0;
        for (uint32_t i = 0; i < N; ++i) {
            expected += static_cast<double>(x_h[i] * y_h[i]);
        }
        if (std::abs(*res_h - expected) > 1e-8) {
            std::cerr << "Error in dot product: expected " << expected << ", got " << *res_h << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "Dot product test passed!" << std::endl;
        POP_RANGE();
    }

    // Scale test
    {
        PUSH_RANGE("Scale Test", 2);
        for (uint32_t i = 0; i < N; ++i)
        {
            x_h[i] = 1.0;
        }
        DeviceMemory<uint32_t, double>::copyHostToDevice(N, x_h, x_d);
        uint32_t blocks = (N + TILE_SIZE - 1) / TILE_SIZE;
        blocks = std::min(blocks, (uint32_t)MAX_BLOCKS);
        dim3 grid(blocks, 1, 1);
        dim3 block(TILE_SIZE, 1, 1);
        DeviceUtils::launchKernel(scale<uint32_t, double>, grid, block, kStream, a, x_d, N);
        DeviceUtils::StreamSynchronize(kStream);
        DeviceMemory<uint32_t, double>::copyDeviceToHost(N, x_d, x_h);
        for (uint32_t i = 0; i < N; ++i)
        {
            if (x_h[i] != 2.0) {
                std::cerr << "Error at index " << i << ": expected 2.0, got " << x_h[i] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        std::cout << "Scale test passed!" << std::endl;
        POP_RANGE();
    }

    // Copy array test
    {
        PUSH_RANGE("Copy Array Test", 2);
        for (uint32_t i = 0; i < N; ++i)
        {
            x_h[i] = static_cast<double>(i+1);
            y_h[i] = 0.0f;
        }
        DeviceMemory<uint32_t, double>::copyHostToDevice(N, x_h, x_d);
        DeviceMemory<uint32_t, double>::copyHostToDevice(N, y_h, y_d);
        uint32_t blocks = (N + TILE_SIZE - 1) / TILE_SIZE;
        blocks = std::min(blocks, (uint32_t)MAX_BLOCKS);
        dim3 grid(blocks, 1, 1);
        dim3 block(TILE_SIZE, 1, 1);
        DeviceUtils::launchKernel(copy_array<uint32_t, double>, grid, block, kStream, x_d, y_d, N);
        DeviceUtils::StreamSynchronize(kStream);
        DeviceMemory<uint32_t, double>::copyDeviceToHost(N, y_d, y_h);
        for (uint32_t i = 0; i < N; ++i)
        {
            if (y_h[i] != x_h[i]) {
                std::cerr << "Error at index " << i << ": expected " << x_h[i] << ", got " << y_h[i] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        std::cout << "Copy array test passed!" << std::endl;
        POP_RANGE();
    }

    // TODO: Add more tests for other kernels...
    return 0;
}