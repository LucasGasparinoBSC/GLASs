/**
 * @file unitt_32.cpp
 * @author Lucas Gasparino
 * @brief Unit test DeviceUtils members using FP32 data type
 * @version 0.1
 * @date 2024-06-01
 * @details Creates 2 FP32 arrays on the host, copies them to device, then calls a kernel using DeviceUtils member functions.
 */
#include "DeviceUtils.hpp"
#include "DeviceMemory.hpp"
#include "Kernels_Lv1.cuh"
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstdint>

int main() {
    // Basic info
    const uint32_t narr = static_cast<uint32_t>(1024);

    // Host arrays
    PUSH_RANGE("Host Array Creation", 0);
    float *h_input = (float *)calloc(narr, sizeof(float));
    float *h_output = (float *)calloc(narr, sizeof(float));
    for (uint32_t i = 0; i < narr; ++i) {
        h_input[i] = static_cast<float>(i+1);
    }
    POP_RANGE();

	// Device arrays
	PUSH_RANGE("Device Array Creation", 0);
	float *d_input = DeviceMemory<uint32_t, float>::deviceCalloc(narr);
	float *d_output = DeviceMemory<uint32_t, float>::deviceCalloc(narr);
	DeviceMemory<uint32_t, float>::copyHostToDevice(narr, h_input, d_input);
	POP_RANGE();

	// Prepare for kernel launch
	uint32_t threadsPerBlock = static_cast<uint32_t>(TILE_SIZE);
	uint32_t blocksPerGrid = (narr + threadsPerBlock - 1) / threadsPerBlock;
	blocksPerGrid = std::min(blocksPerGrid, static_cast<uint32_t>(MAX_BLOCKS));
	dim3 block(threadsPerBlock,1,1);
	dim3 grid(blocksPerGrid,1,1);
	DeviceUtils::Stream_t kStream;
	DeviceUtils::StreamCreate(&kStream);

	// Launch the copy kernel
	PUSH_RANGE("copy_array", 0);
	DeviceUtils::launchKernel(copy_array<uint32_t,float>, grid, block, kStream, d_input, d_output, narr);
	DeviceUtils::StreamSynchronize(0);
	POP_RANGE();

	// Copy back and verify
	DeviceMemory<uint32_t, float>::copyDeviceToHost(narr, d_output, h_output);
	for (uint32_t i = 0; i < narr; ++i) {
		if (h_input[i] != h_output[i]) {
			std::cerr << "Mismatch at index " << i << ": " << h_input[i] << " != " << h_output[i] << std::endl;
			return -1;
		}
	}
	printf("Test passed successfully!\n");

	// Clean-up
	DeviceUtils::StreamDestroy(kStream);
	DeviceMemory<uint32_t, float>::deviceFree(d_input);
	DeviceMemory<uint32_t, float>::deviceFree(d_output);
	free(h_input);
	free(h_output);
    return 0;
}