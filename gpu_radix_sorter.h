#ifndef GPU_RADIX_SORTER_H
#define GPU_RADIX_SORTER_H

#include <vector>
#include <cstdint>
#include <climits>
#include <iostream>
#include <cmath>
#include <algorithm>

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

#define BLOCK_SIZE 256
#define RADIX_BITS 8
#define RADIX (1 << RADIX_BITS)

void sortDataGPU_radix(const std::vector<int>& A, const std::vector<int>& B,
                       std::vector<int>& A_sorted, std::vector<int>& B_sorted);

#endif // GPU_RADIX_SORTER_H
