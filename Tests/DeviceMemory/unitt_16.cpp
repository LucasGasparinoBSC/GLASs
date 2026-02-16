#include "DeviceMemory.hpp"

int main()
{

    const uint32_t arrSize = 1024;
    DeviceUtils::bf16 *h_array;
    DeviceUtils::bf16 *d_array;

    h_array = (DeviceUtils::bf16 *)calloc(arrSize, sizeof(DeviceUtils::bf16));
    for (uint32_t i = 0; i < arrSize; ++i)
    {
        h_array[i] = static_cast<DeviceUtils::bf16>(i);
    }

    // Allocate device memory
    d_array = DeviceMemory<uint32_t, DeviceUtils::bf16>::deviceCalloc(arrSize);

    // Copy host data to device
    DeviceMemory<uint32_t, DeviceUtils::bf16>::copyHostToDevice(arrSize, h_array, d_array);

    // Copy back to host and check results
    DeviceMemory<uint32_t, DeviceUtils::bf16>::copyDeviceToHost(arrSize, d_array, h_array);

    // Verify the results
    for (uint32_t i = 0; i < arrSize; ++i)
    {
        if (h_array[i] != static_cast<DeviceUtils::bf16>(i))
        {
            float real = static_cast<float>(i);
            float value = static_cast<float>(h_array[i]);
            std::cerr << "Mismatch at index " << i << ": " << value << " != " << real << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Free both memory spaces
    DeviceMemory<uint32_t, DeviceUtils::bf16>::deviceFree(d_array);
    free(h_array);
    return 0;
}