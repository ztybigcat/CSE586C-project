#include "gpu_custom_sorter.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <climits> // For INT_MAX

// Error checking macro
#define CUDA_CHECK(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)               \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

// Define block size for kernels
#define BLOCK_SIZE 256

// Bitonic sort kernel
__global__ void bitonicSortKernel(int* d_keys, int* d_values, int N, int stage, int passOfStage);

void sortDataGPU_custom(const std::vector<int>& A, const std::vector<int>& B,
                        std::vector<int>& A_sorted, std::vector<int>& B_sorted) {
    std::uint64_t N = A.size();

    // Find the next power of two
    int log2N = std::ceil(std::log2(N));
    std::uint64_t N_padded = 1 << log2N;

    // Create padded arrays
    std::vector<int> A_padded = A;
    std::vector<int> B_padded = B;

    // Pad the arrays with maximum integer values
    if (N_padded > N) {
        A_padded.resize(N_padded, INT_MAX);
        B_padded.resize(N_padded, INT_MAX);
    }

    // Allocate device memory
    int* d_keys;
    int* d_values;

    CUDA_CHECK(cudaMalloc(&d_keys, N_padded * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, N_padded * sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_keys, B_padded.data(), N_padded * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, A_padded.data(), N_padded * sizeof(int), cudaMemcpyHostToDevice));

    // Perform bitonic sort
    int num_blocks = (N_padded + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(num_blocks);
    dim3 threads(BLOCK_SIZE);

    // Main bitonic sort loop
    int totalStages = log2N;
    for (int stage = 1; stage <= totalStages; ++stage) {
        for (int passOfStage = stage; passOfStage > 0; --passOfStage) {
            bitonicSortKernel<<<blocks, threads>>>(d_keys, d_values, N_padded, stage, passOfStage);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Allocate host memory for sorted data
    B_sorted.resize(N_padded);
    A_sorted.resize(N_padded);

    // Copy sorted data back to host
    CUDA_CHECK(cudaMemcpy(B_sorted.data(), d_keys, N_padded * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(A_sorted.data(), d_values, N_padded * sizeof(int), cudaMemcpyDeviceToHost));

    // Remove padding
    B_sorted.resize(N);
    A_sorted.resize(N);

    // Free device memory
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_values));
}

// Bitonic sort kernel
__global__ void bitonicSortKernel(int* d_keys, int* d_values, int N, int stage, int passOfStage) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    unsigned int pairDistance = 1 << (passOfStage - 1);
    unsigned int blockWidth = 1 << stage;

    unsigned int left = idx;
    unsigned int right = idx ^ pairDistance;

    if (right > left) {
        // Determine the sorting direction
        bool ascending = ((idx / blockWidth) % 2) == 0;

        // Compare and swap
        if ((d_keys[left] > d_keys[right]) == ascending) {
            // Swap keys
            int temp_key = d_keys[left];
            d_keys[left] = d_keys[right];
            d_keys[right] = temp_key;

            // Swap values
            int temp_value = d_values[left];
            d_values[left] = d_values[right];
            d_values[right] = temp_value;
        }
    }
}
