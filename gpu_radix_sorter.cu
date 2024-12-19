#include "gpu_radix_sorter.h"
#include "prefix_sum.h"

// Device function to get digit from key
__device__ __forceinline__ int getDigit(int key, int bitOffset) {
    return (key >> bitOffset) & (RADIX - 1);
}

__global__ void makeExclusiveKernel(const int *in, int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 && n > 0) {
        out[0] = 0;
    } else if (idx < n && idx > 0) {
        out[idx] = in[idx - 1];
    }
}

__global__ void histogramKernel(const int *keys, int *globalHist, int *blockHist, int n, int bitOffset) {
    __shared__ int localHist[RADIX];

    for (int i = threadIdx.x; i < RADIX; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        int digit = getDigit(keys[i], bitOffset);
        atomicAdd(&localHist[digit], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < RADIX; i += blockDim.x) {
        int val = localHist[i];
        atomicAdd(&globalHist[i], val);
        blockHist[blockIdx.x * RADIX + i] = val;
    }
}

__global__ void gatherDigitKernel(const int *blockHist, int *digitArray, int digit, int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBlocks) {
        digitArray[idx] = blockHist[idx * RADIX + digit];
    }
}

__global__ void shiftScanResultKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // We want to convert inclusive prefix sums into offsets. 
    // data[i] after prefixSumLargeArray is inclusive. We want data[i] = (i==0)?0:data[i-1].
    if (idx == 0 && n > 0) {
        data[0] = 0;
    } else if (idx < n && idx > 0) {
        __syncthreads(); // ensure previous is read correctly
        data[idx] = data[idx - 1];
    }
}


__global__ void addConstantKernel(int *data, int n, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += c;
    }
}

__global__ void storeDigitOffsetsKernel(int *blockDigitOffsets, const int *digitArray, int digit, int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBlocks) {
        blockDigitOffsets[idx * RADIX + digit] = digitArray[idx];
    }
}

__global__ void reorderKernel_stable(
    const int *keys_in, const int *values_in,
    int *keys_out, int *values_out,
    const int *blockDigitOffsets,
    int n, int bitOffset)
{
    __shared__ int threadDigits[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d = -1;
    int key = 0, value = 0;
    if (idx < n) {
        key = keys_in[idx];
        value = values_in[idx];
        d = getDigit(key, bitOffset);
    } else {
        d = -1;
    }
    threadDigits[threadIdx.x] = d;
    __syncthreads();

    // Compute local rank
    int localRank = 0;
    if (d != -1) {
        for (int i = 0; i <= threadIdx.x; i++) {
            if (threadDigits[i] == d) localRank++;
        }
    }

    __syncthreads();

    if (idx < n) {
        int pos = blockDigitOffsets[blockIdx.x * RADIX + d] + (localRank - 1);
        keys_out[pos] = key;
        values_out[pos] = value;
    }
}

void sortDataGPU_radix(const std::vector<int> &A, const std::vector<int> &B,
                       std::vector<int> &A_sorted, std::vector<int> &B_sorted)
{
    std::uint64_t N = A.size();
    if (N == 0) return;

    int *d_keys_in, *d_values_in;
    int *d_keys_out, *d_values_out;
    int *d_histograms;       // Global hist
    int *d_blockHist;        // Per-block hist

    CUDA_CHECK(cudaMalloc(&d_keys_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keys_out, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values_out, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_keys_in, B.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values_in, A.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    int numBlocks = (int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    CUDA_CHECK(cudaMalloc(&d_histograms, RADIX * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blockHist, numBlocks * RADIX * sizeof(int)));

    int numBits = (int)(sizeof(int)*8);
    int numPasses = (numBits + RADIX_BITS - 1) / RADIX_BITS;

    // Temporary array for digit prefix sums
    int *d_digitArray;
    CUDA_CHECK(cudaMalloc(&d_digitArray, numBlocks * sizeof(int)));

    for (int pass = 0; pass < numPasses; ++pass) {
        int bitOffset = pass * RADIX_BITS;

        CUDA_CHECK(cudaMemset(d_histograms, 0, RADIX * sizeof(int)));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute histogram
        histogramKernel<<<numBlocks, BLOCK_SIZE>>>(d_keys_in, d_histograms, d_blockHist, (int)N, bitOffset);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Prefix sum global histogram
        prefixSumLargeArray(d_histograms, RADIX);
        CUDA_CHECK(cudaDeviceSynchronize());

        int *d_blockDigitOffsets;
        CUDA_CHECK(cudaMalloc(&d_blockDigitOffsets, numBlocks * RADIX * sizeof(int)));

        // For each digit, build block offsets
        for (int d = 0; d < RADIX; d++) {
            // Gather counts for this digit
            {
                int blocksGather = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
                gatherDigitKernel<<<blocksGather, BLOCK_SIZE>>>(d_blockHist, d_digitArray, d, numBlocks);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            // prefix sum the digit array (inclusive)
            prefixSumLargeArray(d_digitArray, numBlocks);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Convert inclusive prefix sums to exclusive
            int *d_tempForShift;
            CUDA_CHECK(cudaMalloc(&d_tempForShift, numBlocks * sizeof(int)));
            {
                int blocksShift = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
                makeExclusiveKernel<<<blocksShift, BLOCK_SIZE>>>(d_digitArray, d_tempForShift, numBlocks);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            CUDA_CHECK(cudaMemcpy(d_digitArray, d_tempForShift, numBlocks * sizeof(int), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaFree(d_tempForShift));
            CUDA_CHECK(cudaDeviceSynchronize());

            // Add the global offset for this digit
            int baseOffset = 0;
            if (d > 0) {
                int h_offset;
                CUDA_CHECK(cudaMemcpy(&h_offset, &d_histograms[d-1], sizeof(int), cudaMemcpyDeviceToHost));
                baseOffset = h_offset;
            }

            {
                int blocksAdd = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
                addConstantKernel<<<blocksAdd, BLOCK_SIZE>>>(d_digitArray, numBlocks, baseOffset);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            // Store offsets back
            {
                int blocksStore = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
                storeDigitOffsetsKernel<<<blocksStore, BLOCK_SIZE>>>(d_blockDigitOffsets, d_digitArray, d, numBlocks);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        // Reorder
        reorderKernel_stable<<<numBlocks, BLOCK_SIZE>>>(d_keys_in, d_values_in, d_keys_out, d_values_out,
                                                        d_blockDigitOffsets, (int)N, bitOffset);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_blockDigitOffsets));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap in/out buffers for next iteration
        std::swap(d_keys_in, d_keys_out);
        std::swap(d_values_in, d_values_out);
    }

    // Copy results back
    B_sorted.resize(N);
    A_sorted.resize(N);
    CUDA_CHECK(cudaMemcpy(B_sorted.data(), d_keys_in, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(A_sorted.data(), d_values_in, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_digitArray));
    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_values_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_values_out));
    CUDA_CHECK(cudaFree(d_histograms));
    CUDA_CHECK(cudaFree(d_blockHist));
    CUDA_CHECK(cudaDeviceSynchronize());
}
