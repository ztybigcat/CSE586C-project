#include "gpu_reference_sorter.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdint>

void sortDataGPU_reference(const std::vector<int>& A, const std::vector<int>& B,
                           std::vector<int>& A_sorted, std::vector<int>& B_sorted) {
    std::uint64_t N = A.size();

    // Transfer data from host to device
    thrust::device_vector<int> d_A(A);
    thrust::device_vector<int> d_B(B);

    // Create zip iterator to sort A and B together based on B
    typedef thrust::device_vector<int>::iterator Iterator;
    typedef thrust::tuple<Iterator, Iterator> IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

    ZipIterator zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_B.begin(), d_A.begin()));
    ZipIterator zip_end   = thrust::make_zip_iterator(thrust::make_tuple(d_B.end(),   d_A.end()));

    // Sort the zipped iterators based on B
    thrust::sort(zip_begin, zip_end, thrust::less<thrust::tuple<int, int>>());

    // Transfer sorted data back to host
    B_sorted.resize(N);
    A_sorted.resize(N);

    thrust::copy(d_B.begin(), d_B.end(), B_sorted.begin());
    thrust::copy(d_A.begin(), d_A.end(), A_sorted.begin());
}
