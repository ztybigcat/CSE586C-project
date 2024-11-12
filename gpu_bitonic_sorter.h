// gpu_bitonic_sorter.h
#ifndef GPU_BITONIC_SORTER_H
#define GPU_BITONIC_SORTER_H

#include <vector>
#include <cstdint>
#include <climits> // For INT_MAX
#include <iostream>
#include <cmath>

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
#define BLOCK_SIZE 1024

// Function declarations
void sortDataGPU_bitonic(const std::vector<int>& A, const std::vector<int>& B,
                         std::vector<int>& A_sorted, std::vector<int>& B_sorted);

void sortDataGPU_bitonic_shared_memory(const std::vector<int>& A, const std::vector<int>& B,
                                       std::vector<int>& A_sorted, std::vector<int>& B_sorted);
void sortDataGPU_bitonic_Hybrid(const std::vector<int>& A, const std::vector<int>& B,
                                       std::vector<int>& A_sorted, std::vector<int>& B_sorted);

#endif // GPU_BITONIC_SORTER_H
