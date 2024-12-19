#pragma once

#include <cstdio>
#include <cuda_runtime.h>
/**
 * @brief Perform an in-place inclusive prefix sum (scan) on a device array of integers.
 * 
 * @param d_data    Pointer to device memory containing the array to be scanned.
 * @param n         Number of elements in the array.
 * @param stream    CUDA stream to use (default: 0).
 * 
 * After completion, d_data[i] will contain the sum of all elements from d_data[0] to d_data[i].
 */
void prefixSumLargeArray(int* d_data, int n, cudaStream_t stream = 0);

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// A macro for checking CUDA errors easily.
#define CUDA_CHECK_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            abort(); \
        } \
    } while(0)
