// gpu_bitonic_sorter.cu
#include "gpu_bitonic_sorter.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits.h>

#define BLOCK_SIZE 1024 //

// Function prototypes for kernels
__global__ void bitonicSortKernel(int* d_keys, int* d_values, int N, int stage, int passOfStage);
__global__ void bitonicSortKernelShared(int* d_keys, int* d_values, int N, int stage, int passOfStage);
__global__ void bitonicSortKernelHybrid(int* d_keys, int* d_values, int N, int stage, int passOfStage);
// Bitonic Sort using global memory
void sortDataGPU_bitonic(const std::vector<int>& A, const std::vector<int>& B,
                         std::vector<int>& A_sorted, std::vector<int>& B_sorted) {
    std::uint64_t N = A.size();

    // Find the next power of two
    int log2N = std::ceil(std::log2(N));
    std::uint64_t N_padded = 1ULL << log2N;

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
    cudaMalloc(&d_keys, N_padded * sizeof(int));
    cudaMalloc(&d_values, N_padded * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_keys, B_padded.data(), N_padded * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, A_padded.data(), N_padded * sizeof(int), cudaMemcpyHostToDevice);

    // Set up kernel dimensions
    int num_blocks = (N_padded + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(num_blocks);
    dim3 threads(BLOCK_SIZE);

    // Main bitonic sort loop
    int totalStages = log2N;
    for (int stage = 1; stage <= totalStages; ++stage) {
        for (int passOfStage = stage; passOfStage > 0; --passOfStage) {
            bitonicSortKernel<<<blocks, threads>>>(d_keys, d_values, N_padded, stage, passOfStage);
            cudaDeviceSynchronize();
        }
    }

    // Allocate host memory for sorted data
    B_sorted.resize(N_padded);
    A_sorted.resize(N_padded);

    // Copy sorted data back to host
    cudaMemcpy(B_sorted.data(), d_keys, N_padded * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(A_sorted.data(), d_values, N_padded * sizeof(int), cudaMemcpyDeviceToHost);

    // Remove padding
    B_sorted.resize(N);
    A_sorted.resize(N);

    // Free device memory
    cudaFree(d_keys);
    cudaFree(d_values);
}

// Bitonic Sort with shared memory optimization
void sortDataGPU_bitonic_shared_memory(const std::vector<int>& A, const std::vector<int>& B,
                                       std::vector<int>& A_sorted, std::vector<int>& B_sorted) {
    std::uint64_t N = A.size();

    // Find the next power of two
    int log2N = std::ceil(std::log2(N));
    std::uint64_t N_padded = 1ULL << log2N;

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
    cudaMalloc(&d_keys, N_padded * sizeof(int));
    cudaMalloc(&d_values, N_padded * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_keys, B_padded.data(), N_padded * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, A_padded.data(), N_padded * sizeof(int), cudaMemcpyHostToDevice);

    // Set up kernel dimensions
    int num_blocks = (N_padded + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(num_blocks);
    dim3 threads(BLOCK_SIZE);

    // Main bitonic sort loop
    int totalStages = log2N;
    for (int stage = 1; stage <= totalStages; ++stage) {
        int blockWidth = 1 << stage;

        for (int passOfStage = stage; passOfStage > 0; --passOfStage) {
            if (blockWidth <= BLOCK_SIZE) {
                bitonicSortKernelShared<<<blocks, threads>>>(d_keys, d_values, N_padded, stage, passOfStage);
            } 
            if (blockWidth > BLOCK_SIZE)
            
            {
              bitonicSortKernel<<<blocks, threads>>>(d_keys, d_values, N_padded, stage, passOfStage);
            }
            else
            {
              bitonicSortKernel<<<blocks, threads>>>(d_keys, d_values, N_padded, stage, passOfStage);
            }
            cudaDeviceSynchronize();
        }
    }

    // Allocate host memory for sorted data
    B_sorted.resize(N_padded);
    A_sorted.resize(N_padded);

    // Copy sorted data back to host
    cudaMemcpy(B_sorted.data(), d_keys, N_padded * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(A_sorted.data(), d_values, N_padded * sizeof(int), cudaMemcpyDeviceToHost);

    // Remove padding
    B_sorted.resize(N);
    A_sorted.resize(N);

    // Free device memory
    cudaFree(d_keys);
    cudaFree(d_values);
}
void sortDataGPU_bitonic_Hybrid(const std::vector<int>& A, const std::vector<int>& B,
                                       std::vector<int>& A_sorted, std::vector<int>& B_sorted) {
    std::uint64_t N = A.size();

    // Find the next power of two
    int log2N = std::ceil(std::log2(N));
    std::uint64_t N_padded = 1ULL << log2N;

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
    cudaMalloc(&d_keys, N_padded * sizeof(int));
    cudaMalloc(&d_values, N_padded * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_keys, B_padded.data(), N_padded * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, A_padded.data(), N_padded * sizeof(int), cudaMemcpyHostToDevice);

    // Set up kernel dimensions
    int num_blocks = (N_padded + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(num_blocks);
    dim3 threads(BLOCK_SIZE);

    // Main bitonic sort loop
    int totalStages = log2N;
    for (int stage = 1; stage <= totalStages; ++stage) {
        int blockWidth = 1 << stage;

        for (int passOfStage = stage; passOfStage > 0; --passOfStage) {
            if (blockWidth <= BLOCK_SIZE) {
                bitonicSortKernelHybrid<<<blocks, threads>>>(d_keys, d_values, N_padded, stage, passOfStage);
            } 
            if (blockWidth > BLOCK_SIZE)
            
            {
              bitonicSortKernel<<<blocks, threads>>>(d_keys, d_values, N_padded, stage, passOfStage);
            }
            else
            {
              bitonicSortKernel<<<blocks, threads>>>(d_keys, d_values, N_padded, stage, passOfStage);
            }
            cudaDeviceSynchronize();
        }
    }

    // Allocate host memory for sorted data
    B_sorted.resize(N_padded);
    A_sorted.resize(N_padded);

    // Copy sorted data back to host
    cudaMemcpy(B_sorted.data(), d_keys, N_padded * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(A_sorted.data(), d_values, N_padded * sizeof(int), cudaMemcpyDeviceToHost);

    // Remove padding
    B_sorted.resize(N);
    A_sorted.resize(N);

    // Free device memory
    cudaFree(d_keys);
    cudaFree(d_values);
}
// Bitonic Sort Kernel using global memory
__global__ void bitonicSortKernel(int* d_keys, int* d_values, int N, int stage, int passOfStage) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    unsigned int pairDistance = 1 << (passOfStage - 1);
    unsigned int blockWidth = 1 << stage;

    unsigned int left = idx;
    unsigned int right = idx ^ pairDistance;

    if (right > left) {
        bool ascending = ((idx / blockWidth) % 2) == 0;
        if ((d_keys[left] > d_keys[right]) == ascending) {
            int temp_key = d_keys[left];
            d_keys[left] = d_keys[right];
            d_keys[right] = temp_key;

            int temp_value = d_values[left];
            d_values[left] = d_values[right];
            d_values[right] = temp_value;
        }
    }
}

// Bitonic Sort Kernel with shared memory optimization
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

    // Debug: Print loaded values into shared memory for block 0 and thread 0
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        //printf("Initial load - Stage %d, Pass %d: shared_keys[0] = %d, shared_values[0] = %d\n", stage, passOfStage, shared_keys[0], shared_values[0]);
    }

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

        // Debug: Before comparison and swap for thread 0 in block 0
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            //printf("Before swap - Stage %d, Pass %d: shared_keys[0] = %d, shared_keys[1] = %d\n", stage, passOfStage, shared_keys[0], shared_keys[1]);
        }

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
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            //printf("After swap - Stage %d, Pass %d: shared_keys[0] = %d, shared_keys[1] = %d\n", stage, passOfStage, shared_keys[0], shared_keys[1]);
        }
}
}

__global__ void bitonicSortKernelHybrid(int* d_keys, int* d_values, int N, int stage, int passOfStage) {
    __shared__ int shared_keys[BLOCK_SIZE];
    __shared__ int shared_values[BLOCK_SIZE];

    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_idx = threadIdx.x;

    if (global_idx >= N) return;

    unsigned int pairDistance = 1 << (passOfStage - 1);
    unsigned int blockWidth = 1 << stage;

    // If the stage is within block boundaries, use shared memory
    if (blockWidth <= BLOCK_SIZE) {
        // Load elements into shared memory
        shared_keys[local_idx] = d_keys[global_idx];
        shared_values[local_idx] = d_values[global_idx];
        __syncthreads();

        unsigned int local_pair_idx = local_idx ^ pairDistance;

        // Ensure the pair index is within shared memory bounds
        if (local_pair_idx < BLOCK_SIZE) {
            // Determine sorting direction (ascending or descending)
            bool ascending = ((global_idx / blockWidth) % 2) == 0;

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
        }

        // Synchronize before writing back to global memory
        __syncthreads();

        // Write sorted values back to global memory
        d_keys[global_idx] = shared_keys[local_idx];
        d_values[global_idx] = shared_values[local_idx];
    } 
    // If the stage spans across blocks, work directly in global memory
    else {
        unsigned int global_pair_idx = global_idx ^ pairDistance;
        
        // Ensure the pair index is within bounds of the entire array
        if (global_pair_idx < N) {
            // Determine sorting direction (ascending or descending)
            bool ascending = ((global_idx / blockWidth) % 2) == 0;

            // Perform comparison and swap in global memory
            if ((d_keys[global_idx] > d_keys[global_pair_idx]) == ascending) {
                // Swap keys
                int temp_key = d_keys[global_idx];
                d_keys[global_idx] = d_keys[global_pair_idx];
                d_keys[global_pair_idx] = temp_key;

                // Swap values
                int temp_value = d_values[global_idx];
                d_values[global_idx] = d_values[global_pair_idx];
                d_values[global_pair_idx] = temp_value;
            }
        }
    }
}

