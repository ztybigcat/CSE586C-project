#include "gpu_radix_sorter.h"

__device__ __forceinline__ int getDigit(int key, int bitOffset) {
    return (key >> bitOffset) & (RADIX - 1);
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

    // Write localHist to globalHist
    for (int i = threadIdx.x; i < RADIX; i += blockDim.x) {
        int val = localHist[i];
        atomicAdd(&globalHist[i], val);
        // Also store in block-specific histogram
        blockHist[blockIdx.x * RADIX + i] = val;
    }
}


// A utility device function for parallel prefix sum (scan) in a warp
__device__ int warpInclusiveScan(int x) {
    for (int offset = 1; offset < 32; offset <<= 1) {
        int y = __shfl_up_sync(0xffffffff, x, offset);
        if (threadIdx.x >= offset) x += y;
    }
    return x;
}

// Parallel prefix sum (scan) for BLOCK_SIZE threads
// This computes an inclusive prefix sum of an array in shared memory.
template <int BLOCK>
__device__ void blockPrefixSum(int *data) {
    __shared__ int temp[BLOCK];
    int tid = threadIdx.x;
    int val = data[tid];

    // Warp-level scans
    int lane = tid & 31;
    int warpId = tid >> 5;

    // Do warp scan
    for (int offset = 1; offset < 32; offset <<= 1) {
        int y = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += y;
    }

    // Write warp results to temp
    if (lane == 31) temp[warpId] = val;
    __syncthreads();

    // Scan warp results
    if (warpId == 0) {
        int warpVal = (tid < (BLOCK/32)) ? temp[tid] : 0;
        for (int offset = 1; offset < (BLOCK/32); offset <<= 1) {
            int y = __shfl_up_sync(0xffffffff, warpVal, offset);
            if ((tid) >= offset) warpVal += y;
        }
        if (tid < (BLOCK/32)) temp[tid] = warpVal;
    }
    __syncthreads();

    // Add warp offsets
    if (warpId > 0)
        val += temp[warpId - 1];

    data[tid] = val;
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
        // For threads outside range, set digit to -1
        // so they don't contribute
        d = -1;
    }
    threadDigits[threadIdx.x] = d;
    __syncthreads();
    int localRank = 0;
    if (d != -1) {
        for (int i = 0; i <= threadIdx.x; i++) {
            if (threadDigits[i] == d) localRank++;
        }
    }

    __syncthreads();

    // Now, blockDigitOffsets[blockIdx.x * RADIX + d] gives the start offset for digit d in this block.
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

    int numBits = (int)sizeof(int)*8;
    int numPasses = (numBits + RADIX_BITS - 1) / RADIX_BITS;

    for (int pass = 0; pass < numPasses; ++pass) {
        int bitOffset = pass * RADIX_BITS;

        CUDA_CHECK(cudaMemset(d_histograms, 0, RADIX * sizeof(int)));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute histogram and per-block hist
        histogramKernel<<<numBlocks, BLOCK_SIZE>>>(d_keys_in, d_histograms, d_blockHist, (int)N, bitOffset);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy histograms back to host
        std::vector<int> h_globalHistogram(RADIX);
        CUDA_CHECK(cudaMemcpy(h_globalHistogram.data(), d_histograms, RADIX * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> h_blockHist(numBlocks * RADIX);
        CUDA_CHECK(cudaMemcpy(h_blockHist.data(), d_blockHist, numBlocks * RADIX * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute global prefix sums (digitOffsets)
        std::vector<int> h_digitOffsets(RADIX);
        int total = 0;
        for (int i = 0; i < RADIX; ++i) {
            h_digitOffsets[i] = total;
            total += h_globalHistogram[i];
        }

        // Now compute per-block prefix sums for each digit.
        // This gives blockDigitOffsets: For block b and digit d,
        // blockDigitOffsets[b * RADIX + d] = h_digitOffsets[d] + sum_of_this_digit_in_all_previous_blocks < b.
        std::vector<int> h_blockDigitOffsets(numBlocks * RADIX);
        for (int d = 0; d < RADIX; d++) {
            int running = h_digitOffsets[d];
            for (int b = 0; b < numBlocks; b++) {
                h_blockDigitOffsets[b * RADIX + d] = running;
                running += h_blockHist[b * RADIX + d];
            }
        }

        int *d_blockDigitOffsets;
        CUDA_CHECK(cudaMalloc(&d_blockDigitOffsets, numBlocks * RADIX * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_blockDigitOffsets, h_blockDigitOffsets.data(), numBlocks * RADIX * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        reorderKernel_stable<<<numBlocks, BLOCK_SIZE>>>(
            d_keys_in, d_values_in, d_keys_out, d_values_out,
            d_blockDigitOffsets, (int)N, bitOffset
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_blockDigitOffsets));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap in/out buffers for next iteration
        std::swap(d_keys_in, d_keys_out);
        std::swap(d_values_in, d_values_out);
    }

    B_sorted.resize(N);
    A_sorted.resize(N);
    CUDA_CHECK(cudaMemcpy(B_sorted.data(), d_keys_in, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(A_sorted.data(), d_values_in, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_keys_in));
    CUDA_CHECK(cudaFree(d_values_in));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_values_out));
    CUDA_CHECK(cudaFree(d_histograms));
    CUDA_CHECK(cudaFree(d_blockHist));
    CUDA_CHECK(cudaDeviceSynchronize());
}