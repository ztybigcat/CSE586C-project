// gpu_radix_sorter.cu
#include "gpu_radix_sorter.h"

__device__ __forceinline__ int getDigit(int key, int bitOffset)
{
    return (key >> bitOffset) & (RADIX - 1);
}

__global__ void histogramKernel(const int *keys, int *histograms, int n, int bitOffset)
{
    __shared__ int localHist[RADIX];

    // Initialize local histogram
    for (int i = threadIdx.x; i < RADIX; i += blockDim.x)
    {
        localHist[i] = 0;
    }
    __syncthreads();

    // Each block handles its portion
    int itemsPerBlock = (n + gridDim.x - 1) / gridDim.x;
    int blockStart = blockIdx.x * itemsPerBlock;
    int blockEnd = min(blockStart + itemsPerBlock, n);

    // Count elements in this block's range
    for (int i = blockStart + threadIdx.x; i < blockEnd; i += blockDim.x)
    {
        int digit = getDigit(keys[i], bitOffset);
        atomicAdd(&localHist[digit], 1);
    }
    __syncthreads();

    // Write local histogram to global memory
    for (int i = threadIdx.x; i < RADIX; i += blockDim.x)
    {
        atomicAdd(&histograms[i], localHist[i]);
    }
}

__global__ void reorderKernel(const int *keys_in, const int *values_in, int *keys_out, int *values_out,
                              const int *baseOffsets, const int *digitCounts, int *currentOffsets,
                              int n, int bitOffset, int *errorFlag)
{

    // Each block handles its portion
    int itemsPerBlock = (n + gridDim.x - 1) / gridDim.x;
    int blockStart = blockIdx.x * itemsPerBlock;
    int blockEnd = min(blockStart + itemsPerBlock, n);

    // Each thread in the block handles one element at a time
    for (int i = blockStart; i < blockEnd; i++)
    {
        if (threadIdx.x == (i - blockStart) % blockDim.x)
        {
            int key = keys_in[i];
            int value = values_in[i];
            int digit = getDigit(key, bitOffset);

            // Get exclusive position in output array
            int writePos = baseOffsets[digit] + atomicAdd(&currentOffsets[digit], 1);

            if (writePos < baseOffsets[digit] + digitCounts[digit])
            {
                atomicExch(&keys_out[writePos], key);
                atomicExch(&values_out[writePos], value);
            }
        }
        __syncthreads(); // Ensure one element is fully processed before moving to next
    }
}

void sortDataGPU_radix(const std::vector<int> &A, const std::vector<int> &B,
                       std::vector<int> &A_sorted, std::vector<int> &B_sorted)
{
    std::uint64_t N = A.size();
    if (N == 0)
        return;

    // Allocate device memory
    int *d_keys_in;
    int *d_values_in;
    int *d_keys_out;
    int *d_values_out;
    int *d_histograms;
    int *d_errorFlag;

    CUDA_CHECK(cudaMalloc(&d_keys_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keys_out, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values_out, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_histograms, RADIX * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_errorFlag, sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_keys_in, B.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values_in, A.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = min(numBlocks, 1024);

    int numBits = sizeof(int) * 8;
    int numPasses = (numBits + RADIX_BITS - 1) / RADIX_BITS;

    for (int pass = 0; pass < numPasses; ++pass)
    {
        int bitOffset = pass * RADIX_BITS;

        // Reset histograms and error flag
        CUDA_CHECK(cudaMemset(d_histograms, 0, RADIX * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_errorFlag, 0, sizeof(int)));

        // Step 1: Compute histograms
        histogramKernel<<<numBlocks, BLOCK_SIZE>>>(d_keys_in, d_histograms, N, bitOffset);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 2: Get histogram data from device
        std::vector<int> h_globalHistogram(RADIX);
        CUDA_CHECK(cudaMemcpy(h_globalHistogram.data(), d_histograms,
                              RADIX * sizeof(int), cudaMemcpyDeviceToHost));

        // Compute offsets
        std::vector<int> h_digitOffsets(RADIX);
        int total = 0;
        for (int i = 0; i < RADIX; ++i)
        {
            h_digitOffsets[i] = total;
            total += h_globalHistogram[i];
        }

        // Allocate and copy data for reorder kernel
        int *d_baseOffsets, *d_digitCounts, *d_currentOffsets;
        CUDA_CHECK(cudaMalloc(&d_baseOffsets, RADIX * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_digitCounts, RADIX * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_currentOffsets, RADIX * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_baseOffsets, h_digitOffsets.data(),
                              RADIX * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_digitCounts, h_globalHistogram.data(),
                              RADIX * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_currentOffsets, 0, RADIX * sizeof(int)));

        // Step 3: Reorder elements
        reorderKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_keys_in, d_values_in, d_keys_out, d_values_out,
            d_baseOffsets, d_digitCounts, d_currentOffsets,
            N, bitOffset, d_errorFlag);
        // After kernel launch:
        if (pass == 0)
        { // Only for first pass
            std::vector<int> check_keys(N);
            std::vector<int> check_values(N);
            CUDA_CHECK(cudaMemcpy(check_keys.data(), d_keys_out, N * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(check_values.data(), d_values_out, N * sizeof(int), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check for errors
        int h_errorFlag;
        CUDA_CHECK(cudaMemcpy(&h_errorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_errorFlag)
        {
            printf("Error detected in pass %d\n", pass);

            // Verify final offsets
            std::vector<int> final_offsets(RADIX);
            CUDA_CHECK(cudaMemcpy(final_offsets.data(), d_currentOffsets,
                                  RADIX * sizeof(int), cudaMemcpyDeviceToHost));

            printf("Final offset verification:\n");
            for (int i = 0; i < RADIX; i++)
            {
                if (final_offsets[i] != h_globalHistogram[i])
                {
                    printf("Digit %d: expected count %d, got %d\n",
                           i, h_globalHistogram[i], final_offsets[i]);
                }
            }
        }

        // Cleanup pass-specific allocations
        CUDA_CHECK(cudaFree(d_baseOffsets));
        CUDA_CHECK(cudaFree(d_digitCounts));
        CUDA_CHECK(cudaFree(d_currentOffsets));

        // Swap buffers for next pass
        std::swap(d_keys_in, d_keys_out);
        std::swap(d_values_in, d_values_out);
    }

    // Copy final results back to host
    B_sorted.resize(N);
    A_sorted.resize(N);
    CUDA_CHECK(cudaMemcpy(B_sorted.data(), d_keys_in, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(A_sorted.data(), d_values_in, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_values_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_values_out));
    CUDA_CHECK(cudaFree(d_histograms));
    CUDA_CHECK(cudaFree(d_errorFlag));
}