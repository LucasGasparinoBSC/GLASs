#include <cstdlib>
#include "CUDA_Utils.cuh"
#include "DeviceMemory.hpp"

int main()
{

    const uint32_t arrSize = 1024;
    double *h_array;
    double *d_array;

    h_array = (double *)calloc(arrSize, sizeof(double));
    for (uint32_t i = 0; i < arrSize; ++i)
    {
        h_array[i] = static_cast<double>(i);
    }

    // Allocate device memory
    d_array = DeviceMemory<uint32_t, double>::deviceCalloc(arrSize);

    // Copy host data to device
    DeviceMemory<uint32_t, double>::copyHostToDevice(arrSize, h_array, d_array);

// Simple ACC kernel to increment each element by 1
#pragma acc parallel loop deviceptr(d_array)
    for (uint32_t i = 0; i < arrSize; ++i)
    {
        d_array[i] += 1.0f;
    }

    // Copy back to host and check results
    DeviceMemory<uint32_t, double>::copyDeviceToHost(arrSize, d_array, h_array);
    for (uint32_t i = 0; i < arrSize; ++i)
    {
        if (h_array[i] != static_cast<double>(i + 1))
        {
            std::cerr << "Mismatch at index " << i << ": " << h_array[i] << " != " << static_cast<double>(i + 1) << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Free both memory spaces
    DeviceMemory<uint32_t, double>::deviceFree(d_array);
    free(h_array);
    return 0;
}