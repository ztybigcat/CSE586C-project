#ifndef GPU_CUSTOM_SORTER_H
#define GPU_CUSTOM_SORTER_H

#include <vector>
#include <cstdint>

// Declaration of the new custom GPU sorter (with shared memory optimization)
void sortDataGPU_custom(const std::vector<int>& A, const std::vector<int>& B,
                        std::vector<int>& A_sorted, std::vector<int>& B_sorted);

// Declaration of the old custom GPU sorter (without shared memory optimization)
void sortDataGPU_custom_old(const std::vector<int>& A, const std::vector<int>& B,
                            std::vector<int>& A_sorted, std::vector<int>& B_sorted);

#endif // GPU_CUSTOM_SORTER_H
