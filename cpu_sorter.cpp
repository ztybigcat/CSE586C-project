#include "cpu_sorter.h"
#include <algorithm>
#include <cstdint>


void sortDataCPU(const std::vector<int>& A, const std::vector<int>& B,
                 std::vector<int>& A_sorted, std::vector<int>& B_sorted) {
    std::uint64_t N = A.size();

    // Create a vector of pairs (B[i], A[i])
    std::vector<std::pair<int, int>> pairs(N);
    for (std::uint64_t i = 0; i < N; ++i) {
        pairs[i] = std::make_pair(B[i], A[i]);
    }

    // Sort the pairs based on B[i]
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  return a.first < b.first;
              });

    // Extract the sorted A[i] and B[i]
    A_sorted.resize(N);
    B_sorted.resize(N);
    for (std::uint64_t i = 0; i < N; ++i) {
        B_sorted[i] = pairs[i].first;
        A_sorted[i] = pairs[i].second;
    }
}
