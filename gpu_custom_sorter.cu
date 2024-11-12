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

// Number of bits processed per pass (radix)
#define RADIX_BITS 8
#define RADIX (1 << RADIX_BITS)

// Function prototypes
__global__ void bitonicSortKernel(int* d_keys, int* d_values, int N, int stage, int passOfStage);
__global__ void bitonicSortKernelShared(int* d_keys, int* d_values, int N, int stage, int passOfStage);

// Radix Sort Kernels
__global__ void histogramKernel(const int* keys, int* histograms, int n, int bitOffset);
__global__ void reorderKernel(const int* keys_in, const int* values_in, int* keys_out, int* values_out,
                              const int* indices, const int* histograms, int n, int bitOffset);

// Scan Kernels
// __global__ void scanKernel(int* data, int n);
// __global__ void scanHillisSteeleKernel(int* data, int n);

// Device function to get digit
__device__ __forceinline__ int getDigit(int key, int bitOffset) {
    return (key >> bitOffset) & (RADIX - 1);
}

// Bitonic Sort using global memory
void sortDataGPU_bitonic(const std::vector<int>& A, const std::vector<int>& B,
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

    // Set up kernel dimensions
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

// Bitonic Sort with shared memory optimization
void sortDataGPU_bitonic_shared_memory(const std::vector<int>& A, const std::vector<int>& B,
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

    // Set up kernel dimensions
    int num_blocks = (N_padded + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(num_blocks);
    dim3 threads(BLOCK_SIZE);

    // Main bitonic sort loop
    int totalStages = log2N;
    for (int stage = 1; stage <= totalStages; ++stage) {
        for (int passOfStage = stage; passOfStage > 0; --passOfStage) {
            bitonicSortKernelShared<<<blocks, threads>>>(d_keys, d_values, N_padded, stage, passOfStage);
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

// Radix Sort
void sortDataGPU_radix(const std::vector<int>& A, const std::vector<int>& B,
                       std::vector<int>& A_sorted, std::vector<int>& B_sorted) {
    std::uint64_t N = A.size();

    // Allocate device memory
    int* d_keys_in;
    int* d_values_in;
    int* d_keys_out;
    int* d_values_out;

    CUDA_CHECK(cudaMalloc(&d_keys_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keys_out, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values_out, N * sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_keys_in, B.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values_in, A.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate memory for histograms and indices
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int* d_histograms;
    CUDA_CHECK(cudaMalloc(&d_histograms, numBlocks * RADIX * sizeof(int)));
    int* d_scanSums;
    CUDA_CHECK(cudaMalloc(&d_scanSums, RADIX * sizeof(int)));

    // Allocate memory for indices
    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, N * sizeof(int)));

    // Radix sort parameters
    int numBits = sizeof(int) * 8; // 32 bits for int
    int numPasses = (numBits + RADIX_BITS - 1) / RADIX_BITS;

    for (int pass = 0; pass < numPasses; ++pass) {
        int bitOffset = pass * RADIX_BITS;

        // Initialize histograms
        CUDA_CHECK(cudaMemset(d_histograms, 0, numBlocks * RADIX * sizeof(int)));

        // Step 1: Compute local histograms
        histogramKernel<<<numBlocks, BLOCK_SIZE>>>(d_keys_in, d_histograms, N, bitOffset);
        CUDA_CHECK(cudaGetLastError());

        // Step 2: Compute global histogram (sum of local histograms)
        int* h_histograms = new int[numBlocks * RADIX];
        CUDA_CHECK(cudaMemcpy(h_histograms, d_histograms, numBlocks * RADIX * sizeof(int), cudaMemcpyDeviceToHost));

        int* h_globalHistogram = new int[RADIX];
        std::fill(h_globalHistogram, h_globalHistogram + RADIX, 0);

        for (int i = 0; i < numBlocks; ++i) {
            for (int j = 0; j < RADIX; ++j) {
                h_globalHistogram[j] += h_histograms[i * RADIX + j];
            }
        }

        // Step 3: Compute exclusive prefix sum (scan) on global histogram
        int sum = 0;
        for (int i = 0; i < RADIX; ++i) {
            int temp = h_globalHistogram[i];
            h_globalHistogram[i] = sum;
            sum += temp;
        }

        // Copy global histogram back to device
        CUDA_CHECK(cudaMemcpy(d_scanSums, h_globalHistogram, RADIX * sizeof(int), cudaMemcpyHostToDevice));

        // Step 4: Compute indices for each element
        reorderKernel<<<numBlocks, BLOCK_SIZE>>>(d_keys_in, d_values_in, d_keys_out, d_values_out,
                                                 d_indices, d_histograms, N, bitOffset);
        CUDA_CHECK(cudaGetLastError());

        // Swap input and output arrays for next pass
        std::swap(d_keys_in, d_keys_out);
        std::swap(d_values_in, d_values_out);

        // Clean up
        delete[] h_histograms;
        delete[] h_globalHistogram;
    }

    // Copy sorted data back to host
    B_sorted.resize(N);
    A_sorted.resize(N);
    CUDA_CHECK(cudaMemcpy(B_sorted.data(), d_keys_in, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(A_sorted.data(), d_values_in, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_values_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_values_out));
    CUDA_CHECK(cudaFree(d_histograms));
    CUDA_CHECK(cudaFree(d_scanSums));
    CUDA_CHECK(cudaFree(d_indices));
}

// Kernel to compute local histograms
__global__ void histogramKernel(const int* keys, int* histograms, int n, int bitOffset) {
    __shared__ int localHist[RADIX];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize local histogram
    for (int i = tid; i < RADIX; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Accumulate local histogram
    if (idx < n) {
        int key = keys[idx];
        int digit = getDigit(key, bitOffset);
        atomicAdd(&localHist[digit], 1);
    }
    __syncthreads();

    // Write local histogram to global memory
    for (int i = tid; i < RADIX; i += blockDim.x) {
        histograms[blockIdx.x * RADIX + i] = localHist[i];
    }
}

// Kernel to reorder elements based on computed indices
__global__ void reorderKernel(const int* keys_in, const int* values_in, int* keys_out, int* values_out,
                              const int* indices, const int* histograms, int n, int bitOffset) {
    __shared__ int localHist[RADIX];
    __shared__ int localScan[RADIX];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load global histogram into shared memory
    if (tid < RADIX) {
        localHist[tid] = histograms[blockIdx.x * RADIX + tid];
    }
    __syncthreads();

    // Compute exclusive scan on local histogram
    if (tid < RADIX) {
        int sum = 0;
        for (int i = 0; i < tid; ++i) {
            sum += localHist[i];
        }
        localScan[tid] = sum;
    }
    __syncthreads();

    // Compute global offsets
    int digit;
    if (idx < n) {
        int key = keys_in[idx];
        digit = getDigit(key, bitOffset);
    }

    __shared__ int base[RADIX];
    if (tid < RADIX) {
        base[tid] = 0;
    }
    __syncthreads();

    // Compute base address for each radix
    if (tid == 0) {
        for (int i = 0; i < RADIX; ++i) {
            int sum = 0;
            for (int j = 0; j < blockIdx.x; ++j) {
                sum += histograms[j * RADIX + i];
            }
            base[i] = localScan[i] + sum;
        }
    }
    __syncthreads();

    // Write elements to output array
    if (idx < n) {
        int pos = base[digit] + atomicAdd(&localHist[digit], -1) - 1;
        keys_out[pos] = keys_in[idx];
        values_out[pos] = values_in[idx];
    }
}

// Bitonic Sort Kernel
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

// Bitonic Sort Kernel with Shared Memory Optimization
__global__ void bitonicSortKernelShared(int* d_keys, int* d_values, int N, int stage, int passOfStage) {
    // Shared memory for the keys and values
    __shared__ int shared_keys[BLOCK_SIZE];
    __shared__ int shared_values[BLOCK_SIZE];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;  // Ensure the global index is within bounds

    // Load elements into shared memory (using thread index for shared memory access)
    shared_keys[threadIdx.x] = d_keys[idx];
    shared_values[threadIdx.x] = d_values[idx];
    __syncthreads();  // Ensure all threads have loaded their data

    unsigned int pairDistance = 1 << (passOfStage - 1);
    unsigned int blockWidth = 1 << stage;

    // Calculate local thread indices for the current block
    unsigned int local_idx = threadIdx.x;
    unsigned int local_pair_idx = local_idx ^ pairDistance;  // Calculate pair index in shared memory

    // Ensure the pair index is within block size (to avoid out-of-bounds access in shared memory)
    if (local_pair_idx < BLOCK_SIZE) {
        // Determine the sorting direction (ascending or descending)
        bool ascending = ((idx / blockWidth) % 2) == 0;

        // Synchronize before comparison
        __syncthreads();

        // Compare and swap within shared memory
        if ((shared_keys[local_idx] > shared_keys[local_pair_idx]) == ascending) {
            // Swap keys
            int temp_key = shared_keys[local_idx];
            shared_keys[local_idx] = shared_keys[local_pair_idx];
            shared_keys[local_pair_idx] = temp_key;

            // Swap values
            int temp_value = shared_values[local_idx];
            shared_values[local_idx] = shared_values[local_pair_idx];
            shared_values[local_pair_idx] = temp_value;
        }

        // Synchronize after comparison and swap
        __syncthreads();
    }

    // Synchronize before writing back to global memory
    __syncthreads();

    // Write back the sorted keys and values to global memory
    d_keys[idx] = shared_keys[local_idx];
    d_values[idx] = shared_values[local_idx];
}
