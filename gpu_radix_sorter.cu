// gpu_radix_sorter.cu
#include "gpu_radix_sorter.h"

__device__ __forceinline__ int getDigit(int key, int bitOffset) {
    return (key >> bitOffset) & (RADIX - 1);
}

// Modified histogram kernel with proper synchronization
__global__ void histogramKernel(const int* keys, int* histograms, int n, int bitOffset) {
    __shared__ int localHist[RADIX];

    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    
    // Initialize local histogram
    for (int i = tid; i < RADIX; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Process multiple elements per thread if n > blockDim.x
    for (int idx = block_start + tid; idx < n; idx += blockDim.x * gridDim.x) {
        int key = keys[idx];
        int digit = getDigit(key, bitOffset);
        atomicAdd(&localHist[digit], 1);
    }
    __syncthreads();

    // Write back to global memory
    for (int i = tid; i < RADIX; i += blockDim.x) {
        histograms[blockIdx.x * RADIX + i] = localHist[i];
    }
}

__global__ void reorderKernel(const int* keys_in, const int* values_in, int* keys_out, int* values_out,
                              const int* digitOffsets, int n, int bitOffset) {
    // Use global offsets directly instead of shared memory
    // since we need atomic operations to work across blocks
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        int key = keys_in[idx];
        int value = values_in[idx];
        int digit = getDigit(key, bitOffset);
        
        // Use global memory atomic operation
        int globalPos = atomicAdd((int*)&digitOffsets[digit], 1);
        
        if (globalPos < n) {
            keys_out[globalPos] = key;
            values_out[globalPos] = value;
        }
    }
}

void sortDataGPU_radix(const std::vector<int>& A, const std::vector<int>& B,
                       std::vector<int>& A_sorted, std::vector<int>& B_sorted) {
    std::uint64_t N = A.size();
    if (N == 0) return;

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

    // Calculate grid dimensions
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = min(numBlocks, 1024);  // Limit number of blocks
    
    // Allocate memory for histograms
    int* d_histograms;
    CUDA_CHECK(cudaMalloc(&d_histograms, numBlocks * RADIX * sizeof(int)));

    int numBits = sizeof(int) * 8;
    int numPasses = (numBits + RADIX_BITS - 1) / RADIX_BITS;

    for (int pass = 0; pass < numPasses; ++pass) {
        int bitOffset = pass * RADIX_BITS;

        // Initialize histograms
        CUDA_CHECK(cudaMemset(d_histograms, 0, numBlocks * RADIX * sizeof(int)));

        // Step 1: Compute local histograms
        histogramKernel<<<numBlocks, BLOCK_SIZE>>>(d_keys_in, d_histograms, N, bitOffset);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 2: Compute global histogram on CPU
        std::vector<int> h_histograms(numBlocks * RADIX);
        CUDA_CHECK(cudaMemcpy(h_histograms.data(), d_histograms, numBlocks * RADIX * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<int> h_globalHistogram(RADIX, 0);
        for (int i = 0; i < numBlocks; ++i) {
            for (int j = 0; j < RADIX; ++j) {
                h_globalHistogram[j] += h_histograms[i * RADIX + j];
            }
        }

        // Compute digit offsets
        std::vector<int> h_digitOffsets(RADIX);
        int total = 0;
        for (int i = 0; i < RADIX; ++i) {
            h_digitOffsets[i] = total;
            total += h_globalHistogram[i];
        }

        // Copy digit offsets to device and make them mutable
        int* d_digitOffsets;
        CUDA_CHECK(cudaMalloc(&d_digitOffsets, RADIX * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_digitOffsets, h_digitOffsets.data(), RADIX * sizeof(int), cudaMemcpyHostToDevice));

        // Step 3: Reorder elements
        reorderKernel<<<numBlocks, BLOCK_SIZE>>>(d_keys_in, d_values_in, d_keys_out, d_values_out,
                                                 d_digitOffsets, N, bitOffset);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap input and output arrays for next pass
        std::swap(d_keys_in, d_keys_out);
        std::swap(d_values_in, d_values_out);

        CUDA_CHECK(cudaFree(d_digitOffsets));
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
}