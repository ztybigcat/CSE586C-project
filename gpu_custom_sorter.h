#ifndef GPU_CUSTOM_SORTER_H
#define GPU_CUSTOM_SORTER_H

#include <vector>
#include <cstdint>

void sortDataGPU_custom(const std::vector<int>& A, const std::vector<int>& B,
                        std::vector<int>& A_sorted, std::vector<int>& B_sorted);

#endif // GPU_CUSTOM_SORTER_H
