#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "data_generator.h"
#include "cpu_sorter.h"
#include "gpu_reference_sorter.h"
#include "gpu_custom_sorter.h"
#include "performance_timer.h"

// Function to time GPU execution
float timeGPUExecution(void (*gpu_sort_function)(const std::vector<int>&, const std::vector<int>&, std::vector<int>&, std::vector<int>&),
                       const std::vector<int>& A, const std::vector<int>& B,
                       std::vector<int>& A_sorted, std::vector<int>& B_sorted) {
    cudaEvent_t start, stop;  
    cudaEventCreate(&start);  
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Call the GPU sorting function
    gpu_sort_function(A, B, A_sorted, B_sorted);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    // Size of the arrays
    const std::uint64_t N = 10000000; // 10 million elements

    // Generate test data
    std::vector<int> A;
    std::vector<int> B;
    generateTestData(A, B, N);

    // Prepare variables for sorted data
    std::vector<int> A_sorted_cpu;
    std::vector<int> B_sorted_cpu;

    std::vector<int> A_sorted_gpu_ref;
    std::vector<int> B_sorted_gpu_ref;

    std::vector<int> A_sorted_bitonic;
    std::vector<int> B_sorted_bitonic;

    std::vector<int> A_sorted_bitonic_shared;
    std::vector<int> B_sorted_bitonic_shared;

    std::vector<int> A_sorted_radix;
    std::vector<int> B_sorted_radix;

    // Measure performance for CPU sorting
    PerformanceTimer timer_cpu;
    timer_cpu.start();

    // CPU sorting
    sortDataCPU(A, B, A_sorted_cpu, B_sorted_cpu);

    timer_cpu.stop();
    std::cout << "CPU sorting completed in " << timer_cpu.getElapsedSeconds() * 1000 << " ms.\n";

    // GPU reference sorting
    PerformanceTimer timer_gpu_ref;
    timer_gpu_ref.start();

    sortDataGPU_reference(A, B, A_sorted_gpu_ref, B_sorted_gpu_ref);

    timer_gpu_ref.stop();
    std::cout << "GPU reference sorting completed in " << timer_gpu_ref.getElapsedSeconds() * 1000 << " ms.\n";

    // GPU bitonic sorting
    float bitonic_gpu_time = timeGPUExecution(sortDataGPU_bitonic, A, B, A_sorted_bitonic, B_sorted_bitonic);
    std::cout << "GPU bitonic sorting completed in " << bitonic_gpu_time << " ms.\n";

    // GPU bitonic sorting with shared memory optimization
    float bitonic_shared_gpu_time = timeGPUExecution(sortDataGPU_bitonic_shared_memory, A, B, A_sorted_bitonic_shared, B_sorted_bitonic_shared);
    std::cout << "GPU bitonic sorting with shared memory completed in " << bitonic_shared_gpu_time << " ms.\n";

    // GPU radix sorting
    float radix_gpu_time = timeGPUExecution(sortDataGPU_radix, A, B, A_sorted_radix, B_sorted_radix);
    std::cout << "GPU radix sorting completed in " << radix_gpu_time << " ms.\n";

    // Output first 10 sorted elements for verification
    std::cout << "First 10 sorted elements from CPU:\n";
    for (std::uint64_t i = 0; i < 10; ++i) {
        std::cout << "B[" << i << "] = " << B_sorted_cpu[i]
                  << ", A[" << i << "] = " << A_sorted_cpu[i] << "\n";
    }

    // Output first 10 sorted elements from GPU radix sorter
    std::cout << "First 10 sorted elements from GPU radix sorter:\n";
    for (std::uint64_t i = 0; i < 10; ++i) {
        std::cout << "B[" << i << "] = " << B_sorted_radix[i]
                  << ", A[" << i << "] = " << A_sorted_radix[i] << "\n";
    }

    // Verify that the CPU and GPU radix sort results are the same
    bool is_correct_radix = true;
    for (std::uint64_t i = 0; i < N; ++i) {
        if (B_sorted_cpu[i] != B_sorted_radix[i] || A_sorted_cpu[i] != A_sorted_radix[i]) {
            is_correct_radix = false;
            std::cout << "Radix sort mismatch at index " << i << ": "
                      << "CPU (B, A) = (" << B_sorted_cpu[i] << ", " << A_sorted_cpu[i] << "), "
                      << "GPU Radix (B, A) = (" << B_sorted_radix[i] << ", " << A_sorted_radix[i] << ")\n";
            break;
        }
    }

    if (is_correct_radix) {
        std::cout << "Verification passed: CPU and GPU radix sort results match.\n";
    } else {
        std::cout << "Verification failed: CPU and GPU radix sort results do not match.\n";
    }

    // Similarly, verify for bitonic sort
    bool is_correct_bitonic = true;
    for (std::uint64_t i = 0; i < N; ++i) {
        if (B_sorted_cpu[i] != B_sorted_bitonic[i] || A_sorted_cpu[i] != A_sorted_bitonic[i]) {
            is_correct_bitonic = false;
            std::cout << "Bitonic sort mismatch at index " << i << ": "
                      << "CPU (B, A) = (" << B_sorted_cpu[i] << ", " << A_sorted_cpu[i] << "), "
                      << "GPU Bitonic (B, A) = (" << B_sorted_bitonic[i] << ", " << A_sorted_bitonic[i] << ")\n";
            break;
        }
    }

    if (is_correct_bitonic) {
        std::cout << "Verification passed: CPU and GPU bitonic sort results match.\n";
    } else {
        std::cout << "Verification failed: CPU and GPU bitonic sort results do not match.\n";
    }

    // Verify for bitonic sort with shared memory optimization
    bool is_correct_bitonic_shared = true;
    for (std::uint64_t i = 0; i < N; ++i) {
        if (B_sorted_cpu[i] != B_sorted_bitonic_shared[i] || A_sorted_cpu[i] != A_sorted_bitonic_shared[i]) {
            is_correct_bitonic_shared = false;
            std::cout << "Bitonic shared memory sort mismatch at index " << i << ": "
                      << "CPU (B, A) = (" << B_sorted_cpu[i] << ", " << A_sorted_cpu[i] << "), "
                      << "GPU Bitonic Shared (B, A) = (" << B_sorted_bitonic_shared[i] << ", " << A_sorted_bitonic_shared[i] << ")\n";
            break;
        }
    }

    if (is_correct_bitonic_shared) {
        std::cout << "Verification passed: CPU and GPU bitonic shared memory sort results match.\n";
    } else {
        std::cout << "Verification failed: CPU and GPU bitonic shared memory sort results do not match.\n";
    }

    return 0;
}
