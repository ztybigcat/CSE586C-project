// gpu_radix_sorter.h
#ifndef GPU_RADIX_SORTER_H
#define GPU_RADIX_SORTER_H

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
#define BLOCK_SIZE 256

// Number of bits processed per pass (radix)
#define RADIX_BITS 8
#define RADIX (1 << RADIX_BITS)

// Function declaration
void sortDataGPU_radix(const std::vector<int>& A, const std::vector<int>& B,
                       std::vector<int>& A_sorted, std::vector<int>& B_sorted);

#endif // GPU_RADIX_SORTER_H
