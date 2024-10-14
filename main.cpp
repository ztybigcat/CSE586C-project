#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>

#include "data_generator.h"
#include "cpu_sorter.h"
#include "gpu_reference_sorter.h"
#include "gpu_custom_sorter.h"
#include "performance_timer.h"

int main() {
    // Size of the arrays
    const std::uint64_t N = 100000000; // 1 million elements

    // Generate test data
    std::vector<int> A;
    std::vector<int> B;
    generateTestData(A, B, N);

    // Prepare variables for sorted data
    std::vector<int> A_sorted_cpu;
    std::vector<int> B_sorted_cpu;
    std::vector<int> A_sorted_gpu_ref;
    std::vector<int> B_sorted_gpu_ref;
    std::vector<int> A_sorted_gpu_custom;
    std::vector<int> B_sorted_gpu_custom;

    // Measure performance for CPU sorting
    PerformanceTimer timer_cpu;
    timer_cpu.start();

    sortDataCPU(A, B, A_sorted_cpu, B_sorted_cpu);

    timer_cpu.stop();

    // Measure performance for GPU reference sorting
    PerformanceTimer timer_gpu_ref;
    timer_gpu_ref.start();

    sortDataGPU_reference(A, B, A_sorted_gpu_ref, B_sorted_gpu_ref);

    timer_gpu_ref.stop();

    // Measure performance for custom GPU sorting
    PerformanceTimer timer_gpu_custom;
    timer_gpu_custom.start();

    sortDataGPU_custom(A, B, A_sorted_gpu_custom, B_sorted_gpu_custom);

    timer_gpu_custom.stop();

    // Output timing
    std::cout << "CPU sorting completed in " << timer_cpu.getElapsedSeconds() << " seconds.\n";
    std::cout << "GPU reference sorting completed in " << timer_gpu_ref.getElapsedSeconds() << " seconds.\n";
    std::cout << "Custom GPU sorting completed in " << timer_gpu_custom.getElapsedSeconds() << " seconds.\n";

    // Output first 10 sorted elements for verification
    std::cout << "First 10 sorted elements from CPU:\n";
    for (std::uint64_t i = 0; i < 10; ++i) {
        std::cout << "B[" << i << "] = " << B_sorted_cpu[i]
                  << ", A[" << i << "] = " << A_sorted_cpu[i] << "\n";
    }

    std::cout << "First 10 sorted elements from GPU reference sorter:\n";
    for (std::uint64_t i = 0; i < 10; ++i) {
        std::cout << "B[" << i << "] = " << B_sorted_gpu_ref[i]
                  << ", A[" << i << "] = " << A_sorted_gpu_ref[i] << "\n";
    }

    std::cout << "First 10 sorted elements from custom GPU sorter:\n";
    for (std::uint64_t i = 0; i < 10; ++i) {
        std::cout << "B[" << i << "] = " << B_sorted_gpu_custom[i]
                  << ", A[" << i << "] = " << A_sorted_gpu_custom[i] << "\n";
    }

    // Verify that the CPU and custom GPU results are the same
    bool is_correct = true;
    for (std::uint64_t i = 0; i < N; ++i) {
        if (B_sorted_cpu[i] != B_sorted_gpu_custom[i] || A_sorted_cpu[i] != A_sorted_gpu_custom[i]) {
            is_correct = false;
            std::cout << "Mismatch at index " << i << ": "
                      << "CPU (B, A) = (" << B_sorted_cpu[i] << ", " << A_sorted_cpu[i] << "), "
                      << "Custom GPU (B, A) = (" << B_sorted_gpu_custom[i] << ", " << A_sorted_gpu_custom[i] << ")\n";
            break;
        }
    }

    if (is_correct) {
        std::cout << "Verification passed: CPU and custom GPU results match.\n";
    } else {
        std::cout << "Verification failed: CPU and custom GPU results do not match.\n";
    }

    return 0;
}
