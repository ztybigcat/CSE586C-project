#include "gpu_radix_sorter.h"
#include <stdio.h>

__device__ __forceinline__ int getDigit(int key, int bitOffset)
{
    return (key >> bitOffset) & (RADIX - 1);
}

__global__ void histogramKernel(const int *keys, int *histograms, int n, int bitOffset)
{
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
        atomicAdd(&histograms[i], localHist[i]);
    }
}

__global__ void reorderKernel_serial(
    const int *keys_in, const int *values_in,
    int *keys_out, int *values_out,
    const int *baseOffsets, const int *digitCounts,
    int *currentOffsets,
    int n, int bitOffset)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            int key = keys_in[i];
            int value = values_in[i];
            int digit = getDigit(key, bitOffset);

            int pos = baseOffsets[digit] + currentOffsets[digit];
            currentOffsets[digit]++;

            keys_out[pos] = key;
            values_out[pos] = value;
        }
    }
}

void sortDataGPU_radix(const std::vector<int> &A, const std::vector<int> &B,
                       std::vector<int> &A_sorted, std::vector<int> &B_sorted)
{
    std::uint64_t N = A.size();
    if (N == 0) return;

    int *d_keys_in, *d_values_in;
    int *d_keys_out, *d_values_out;
    int *d_histograms;

    CUDA_CHECK(cudaMalloc(&d_keys_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keys_out, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values_out, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_histograms, RADIX * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_keys_in, B.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_values_in, A.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    int numBlocks = (int)std::min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (size_t)1024);
    int numBits = sizeof(int)*8;
    int numPasses = (numBits + RADIX_BITS - 1) / RADIX_BITS;

    for (int pass = 0; pass < numPasses; ++pass) {
        int bitOffset = pass * RADIX_BITS;

        CUDA_CHECK(cudaMemset(d_histograms, 0, RADIX * sizeof(int)));
        CUDA_CHECK(cudaDeviceSynchronize());

        histogramKernel<<<numBlocks, BLOCK_SIZE>>>(d_keys_in, d_histograms, (int)N, bitOffset);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> h_globalHistogram(RADIX);
        CUDA_CHECK(cudaMemcpy(h_globalHistogram.data(), d_histograms,
                              RADIX * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> h_digitOffsets(RADIX);
        int total = 0;
        for (int i = 0; i < RADIX; ++i) {
            h_digitOffsets[i] = total;
            total += h_globalHistogram[i];
        }

        int *d_baseOffsets, *d_digitCounts, *d_currentOffsets;
        CUDA_CHECK(cudaMalloc(&d_baseOffsets, RADIX * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_digitCounts, RADIX * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_currentOffsets, RADIX * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_baseOffsets, h_digitOffsets.data(),
                              RADIX * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(d_digitCounts, h_globalHistogram.data(),
                              RADIX * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_currentOffsets, 0, RADIX * sizeof(int)));
        CUDA_CHECK(cudaDeviceSynchronize());

        reorderKernel_serial<<<1, 1>>>(
            d_keys_in, d_values_in, d_keys_out, d_values_out,
            d_baseOffsets, d_digitCounts, d_currentOffsets,
            (int)N, bitOffset
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_baseOffsets));
        CUDA_CHECK(cudaFree(d_digitCounts));
        CUDA_CHECK(cudaFree(d_currentOffsets));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(d_keys_in, d_keys_out);
        std::swap(d_values_in, d_values_out);
    }

    B_sorted.resize(N);
    A_sorted.resize(N);
    CUDA_CHECK(cudaMemcpy(B_sorted.data(), d_keys_in, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(A_sorted.data(), d_values_in, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_values_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_values_out));
    CUDA_CHECK(cudaFree(d_histograms));
    CUDA_CHECK(cudaDeviceSynchronize());
}
