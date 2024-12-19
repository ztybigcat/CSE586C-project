#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "prefix_sum.h"
#include <cuda_runtime.h>

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

int main() {
    // Test parameters
    const int N = 10000000; // 10 million elements for testing
    std::vector<int> h_data(N);

    // Initialize random data
    std::srand((unsigned)std::time(nullptr));
    for (int i = 0; i < N; ++i) {
        h_data[i] = std::rand() % 100; // small random numbers
    }

    // Compute prefix sum on the CPU for verification
    std::vector<int> h_ref(N);
    h_ref[0] = h_data[0];
    for (int i = 1; i < N; i++) {
        h_ref[i] = h_ref[i-1] + h_data[i];
    }

    // Allocate and copy to GPU
    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Run prefix sum on GPU
    prefixSumLargeArray(d_data, N);

    // Copy back result
    std::vector<int> h_result(N);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify results
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_result[i] != h_ref[i]) {
            std::cerr << "Mismatch at index " << i << ": got " << h_result[i]
                      << ", expected " << h_ref[i] << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Prefix sum test PASSED." << std::endl;
    } else {
        std::cout << "Prefix sum test FAILED." << std::endl;
    }

    CUDA_CHECK(cudaFree(d_data));

    return correct ? 0 : 1;
}
