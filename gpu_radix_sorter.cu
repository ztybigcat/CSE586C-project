#include "gpu_radix_sorter.h"

// Device function to get digit from key
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

    for (int i = threadIdx.x; i < RADIX; i += blockDim.x) {
        int val = localHist[i];
        atomicAdd(&globalHist[i], val);
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

// Parallel prefix sum (scan) for BLOCK threads (assume BLOCK is a multiple of 32)
// This computes an inclusive prefix sum of an array in shared memory data[].
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

// Kernel to prefix sum the global histogram (one block, RADIX <= BLOCK_SIZE)
__global__ void prefixSumGlobalHist(int *globalHist) {
    __shared__ int temp[RADIX];
    int tid = threadIdx.x;

    if (tid < RADIX) {
        temp[tid] = globalHist[tid];
    }
    __syncthreads();

    blockPrefixSum<BLOCK_SIZE>(temp);
    __syncthreads();

    if (tid < RADIX) {
        globalHist[tid] = temp[tid];
    }
}

// Kernel to prefix sum each digit's per-block counts.
// grid.x = RADIX, block.x = numBlocks
// Each block handles prefix sums for one digit across all histogram blocks.
__global__ void prefixSumBlockHistPerDigit(const int *blockHist, int *blockDigitOffsets, const int *globalHist, int numBlocks) {
    int d = blockIdx.x; // digit index
    int b = threadIdx.x; // block index for that digit

    __shared__ int s_data[BLOCK_SIZE];  
    int val = 0;
    if (b < numBlocks) {
        val = blockHist[b * RADIX + d];
    }
    s_data[b] = val;
    __syncthreads();

    blockPrefixSum<BLOCK_SIZE>(s_data);
    __syncthreads();

    // globalHist is prefix sums of counts. globalHist[d-1] is the start offset for digit d,
    // if d > 0, else 0 if d=0.
    int baseOffset = (d == 0) ? 0 : globalHist[d-1];

    // s_data[b] is inclusive prefix. For the offset for block b, we want the position before block b's data:
    // That means we use s_data[b-1] if b > 0 else 0.
    int prevSum = (b == 0) ? 0 : s_data[b-1];

    if (b < numBlocks) {
        blockDigitOffsets[b * RADIX + d] = baseOffset + prevSum;
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

// The main sorting function
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

    for (int pass = 0; pass < numPasses; ++pass) {
        int bitOffset = pass * RADIX_BITS;

        CUDA_CHECK(cudaMemset(d_histograms, 0, RADIX * sizeof(int)));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute histogram
        histogramKernel<<<numBlocks, BLOCK_SIZE>>>(d_keys_in, d_histograms, d_blockHist, (int)N, bitOffset);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Prefix sum global histogram
        prefixSumGlobalHist<<<1, BLOCK_SIZE>>>(d_histograms);
        CUDA_CHECK(cudaDeviceSynchronize());

        int *d_blockDigitOffsets;
        CUDA_CHECK(cudaMalloc(&d_blockDigitOffsets, numBlocks * RADIX * sizeof(int)));

        // Prefix sum per-digit block histograms
        // We launch RADIX blocks, each block has BLOCK_SIZE threads
        // Assuming numBlocks <= BLOCK_SIZE for simplicity.
        prefixSumBlockHistPerDigit<<<RADIX, BLOCK_SIZE>>>(d_blockHist, d_blockDigitOffsets, d_histograms, numBlocks);
        CUDA_CHECK(cudaDeviceSynchronize());

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