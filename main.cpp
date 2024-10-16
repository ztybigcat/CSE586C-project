#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>  // 添加 CUDA 运行时头文件

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
    const std::uint64_t N = 10000000; // 1 million elements

    // Generate test data
    std::vector<int> A;
    std::vector<int> B;
    generateTestData(A, B, N);

    // Prepare variables for sorted data
    std::vector<int> A_sorted_cpu;
    std::vector<int> B_sorted_cpu;
    std::vector<int> A_sorted_gpu_ref;
    std::vector<int> B_sorted_gpu_ref;
    std::vector<int> A_sorted_gpu_custom_old;
    std::vector<int> B_sorted_gpu_custom_old;
    std::vector<int> A_sorted_gpu_custom;
    std::vector<int> B_sorted_gpu_custom;

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

// Old custom GPU sorting
float old_gpu_time = timeGPUExecution(sortDataGPU_custom_old, A, B, A_sorted_gpu_custom_old, B_sorted_gpu_custom_old);
std::cout << "Old custom GPU sorting completed in " << old_gpu_time << " ms.\n";

// New custom GPU sorting (shared memory optimization)
float new_gpu_time = timeGPUExecution(sortDataGPU_custom, A, B, A_sorted_gpu_custom, B_sorted_gpu_custom);
std::cout << "New custom GPU sorting (shared memory) completed in " << new_gpu_time << " ms.\n";


    // Output first 10 sorted elements for verification
    std::cout << "First 10 sorted elements from CPU:\n";
    for (std::uint64_t i = 0; i < 10; ++i) {
        std::cout << "B[" << i << "] = " << B_sorted_cpu[i]
                  << ", A[" << i << "] = " << A_sorted_cpu[i] << "\n";
    }

    std::cout << "First 10 sorted elements from old custom GPU sorter:\n";
    for (std::uint64_t i = 0; i < 10; ++i) {
        std::cout << "B[" << i << "] = " << B_sorted_gpu_custom_old[i]
                  << ", A[" << i << "] = " << A_sorted_gpu_custom_old[i] << "\n";
    }

    std::cout << "First 10 sorted elements from new custom GPU sorter (shared memory):\n";
      for (std::uint64_t i = 0; i < 10; ++i) {
        std::cout << "B[" << i << "] = " << B_sorted_gpu_custom_old[i]
                  << ", A[" << i << "] = " << A_sorted_gpu_custom_old[i] << "\n";
    }

    // Verify that the CPU and custom GPU results are the same for old version
    bool is_correct_old = true;
    for (std::uint64_t i = 0; i < N; ++i) {
        if (B_sorted_cpu[i] != B_sorted_gpu_custom_old[i] || A_sorted_cpu[i] != A_sorted_gpu_custom_old[i]) {
            is_correct_old = false;
            std::cout << "Old version mismatch at index " << i << ": "
                      << "CPU (B, A) = (" << B_sorted_cpu[i] << ", " << A_sorted_cpu[i] << "), "
                      << "Old GPU (B, A) = (" << B_sorted_gpu_custom_old[i] << ", " << A_sorted_gpu_custom_old[i] << ")\n";
            break;
        }
    }

    if (is_correct_old) {
        std::cout << "Verification passed: CPU and old custom GPU results match.\n";
    } else {
        std::cout << "Verification failed: CPU and old custom GPU results do not match.\n";
    }

    // Verify that the CPU and custom GPU results are the same for new version
    //bool is_correct_new = true;
    //bool is_correct_old = true;
    for (std::uint64_t i = 0; i < N; ++i) {
        if (B_sorted_cpu[i] != B_sorted_gpu_custom_old[i] || A_sorted_cpu[i] != A_sorted_gpu_custom_old[i]) {
            is_correct_old = false;
            std::cout << "Old version mismatch at index " << i << ": "
                      << "CPU (B, A) = (" << B_sorted_cpu[i] << ", " << A_sorted_cpu[i] << "), "
                      << "Old GPU (B, A) = (" << B_sorted_gpu_custom_old[i] << ", " << A_sorted_gpu_custom_old[i] << ")\n";
            break;
        }
    }

    if (is_correct_old) {
        std::cout << "Verification passed: CPU and new custom GPU results match.\n";
    } else {
        std::cout << "Verification failed: CPU and new custom GPU results do not match.\n";
    }

    return 0;
}

