#include "data_generator.h"
#include <random>
#include <algorithm> // For std::shuffle
#include <cstdint>   // For std::uint64_t

void generateTestData(std::vector<int>& A, std::vector<int>& B, std::uint64_t N) {
    // Random number generator
    static std::random_device rd;
    static std::mt19937 rng{rd()};
    std::uniform_int_distribution<int> distA(0, 1000000);

    // Resize vector A to hold N elements
    A.resize(N);

    // Generate random data for A
    for (std::uint64_t i = 0; i < N; ++i) {
        A[i] = distA(rng);
    }

    // Generate unique keys for B
    B.resize(N);
    for (std::uint64_t i = 0; i < N; ++i) {
        B[i] = static_cast<int>(i); // Ensure B[i] is within int range
    }

    // Shuffle B to randomize the keys
    std::shuffle(B.begin(), B.end(), rng);
}

