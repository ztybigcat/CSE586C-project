#include "prefix_sum.h"
#include <cuda_runtime.h>
#include <cstdio>

__device__ __forceinline__ int warpInclusiveScan(int x) {
    for (int offset = 1; offset < 32; offset <<= 1) {
        int y = __shfl_up_sync(0xffffffff, x, offset);
        if ((threadIdx.x & 31) >= offset) x += y;
    }
    return x;
}

/**
 * @brief Kernel to perform an in-block inclusive prefix sum.
 * 
 * Steps:
 * 1. Each block loads its portion of data into shared memory.
 * 2. Do a warp-level scan for each warp.
 * 3. Each warp's last element (the warp sum) is stored into a separate `warpSums` array.
 * 4. The first warp scans the `warpSums` array to get cumulative offsets for each warp.
 * 5. Each thread adds the warp offset to its own partial sum.
 */
__global__ void scanBlockKernel(const int *in, int *out, int *blockSums, int n) {
    __shared__ int sdata[BLOCK_SIZE];
    __shared__ int warpSums[BLOCK_SIZE/32]; // One element per warp

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (idx < n) ? in[idx] : 0;
    sdata[threadIdx.x] = val;
    __syncthreads();

    // Step 1: Intra-warp inclusive scan
    int lane = threadIdx.x & 31;   
    int warpId = threadIdx.x >> 5; 
    int x = sdata[threadIdx.x];
    x = warpInclusiveScan(x);
    sdata[threadIdx.x] = x;  // Now sdata has warp-local prefix sums
    __syncthreads();

    // Step 2: Last thread in each warp writes its warp sum
    if (lane == 31) {
        warpSums[warpId] = x; 
    }
    __syncthreads();

    // Step 3: The first warp scans the warpSums array
    if (warpId == 0) {
        int valw = (threadIdx.x < (BLOCK_SIZE/32)) ? warpSums[threadIdx.x] : 0;
        valw = warpInclusiveScan(valw);
        if (threadIdx.x < (BLOCK_SIZE/32)) warpSums[threadIdx.x] = valw;
    }
    __syncthreads();

    // Step 4: Add the warp offset (for warpId>0) to each thread's prefix sum
    int blockAdd = 0;
    if (warpId > 0) {
        blockAdd = warpSums[warpId - 1];  
    }
    x = sdata[threadIdx.x] + blockAdd;

    // Write results
    if (idx < n) out[idx] = x;
    // Write block sum
    if (threadIdx.x == BLOCK_SIZE - 1 || idx == n - 1) {
        if (blockSums) blockSums[blockIdx.x] = x;
    }
}

/**
 * @brief Add block offsets to all elements of the scanned array (except the first block).
 */
__global__ void addBlockOffsetsKernel(int *data, const int *blockOffsets, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int offset = blockOffsets[blockIdx.x];
        data[idx] += offset;
    }
}

void prefixSumLargeArray(int* d_data, int n, cudaStream_t stream) {
    if (n <= 0) return;

    int threads = BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;

    // If only one block, just do a single-block scan
    if (blocks == 1) {
        scanBlockKernel<<<1, threads, 0, stream>>>(d_data, d_data, nullptr, n);
        CUDA_CHECK_ERROR(cudaGetLastError());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        return;
    }

    int *d_out, *d_blockSums;
    CUDA_CHECK_ERROR(cudaMalloc(&d_out, n * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_blockSums, blocks * sizeof(int)));

    // 1. Scan each block
    scanBlockKernel<<<blocks, threads, 0, stream>>>(d_data, d_out, d_blockSums, n);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // 2. Recursively scan the block sums array to get offsets
    prefixSumLargeArray(d_blockSums, blocks, stream);

    // 3. Add the offsets to all but the first block
    addBlockOffsetsKernel<<<blocks-1, threads, 0, stream>>>(d_out + threads, d_blockSums, n - threads);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Copy result back to d_data
    CUDA_CHECK_ERROR(cudaMemcpyAsync(d_data, d_out, n * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    CUDA_CHECK_ERROR(cudaFree(d_out));
    CUDA_CHECK_ERROR(cudaFree(d_blockSums));
}
