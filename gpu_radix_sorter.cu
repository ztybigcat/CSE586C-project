// gpu_radix_sorter.cu
#include "gpu_radix_sorter.h"

// Function prototypes for kernels
__global__ void histogramKernel(const int* keys, int* histograms, int n, int bitOffset);
__global__ void reorderKernel(const int* keys_in, const int* values_in, int* keys_out, int* values_out,
                              const int* globalHistogram, int n, int bitOffset);

// Device function to get digit
__device__ __forceinline__ int getDigit(int key, int bitOffset) {
    return (key >> bitOffset) & (RADIX - 1);
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

    // Allocate memory for histograms
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int* d_histograms;
    CUDA_CHECK(cudaMalloc(&d_histograms, numBlocks * RADIX * sizeof(int)));

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
        CUDA_CHECK(cudaDeviceSynchronize());

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
        int* d_globalHistogram;
        CUDA_CHECK(cudaMalloc(&d_globalHistogram, RADIX * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_globalHistogram, h_globalHistogram, RADIX * sizeof(int), cudaMemcpyHostToDevice));

        // Step 4: Reorder elements based on computed indices
        reorderKernel<<<numBlocks, BLOCK_SIZE>>>(d_keys_in, d_values_in, d_keys_out, d_values_out,
                                                 d_globalHistogram, N, bitOffset);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap input and output arrays for next pass
        std::swap(d_keys_in, d_keys_out);
        std::swap(d_values_in, d_values_out);

        // Clean up
        delete[] h_histograms;
        delete[] h_globalHistogram;
        CUDA_CHECK(cudaFree(d_globalHistogram));
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
                              const int* globalHistogram, int n, int bitOffset) {
    __shared__ int localHist[RADIX];
    __shared__ int localScan[RADIX];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize local histogram
    for (int i = tid; i < RADIX; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Build local histogram
    if (idx < n) {
        int key = keys_in[idx];
        int digit = getDigit(key, bitOffset);
        atomicAdd(&localHist[digit], 1);
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

    // Compute base address for each digit
    __shared__ int base[RADIX];
    if (tid < RADIX) {
        base[tid] = globalHistogram[tid];
    }
    __syncthreads();

    // Write elements to output array
    if (idx < n) {
        int key = keys_in[idx];
        int value = values_in[idx];
        int digit = getDigit(key, bitOffset);

        int pos = base[digit] + localScan[digit] + tid - localScan[digit];

        keys_out[pos] = key;
        values_out[pos] = value;

        atomicAdd(&base[digit], 1);
    }
}
