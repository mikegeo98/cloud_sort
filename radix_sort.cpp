#include "radix_sort.hpp"
#include <algorithm>
#include <vector>
#include <utility>

// Single-threaded LSB-based Radix Sort implementation
// Processes 64-bit keys in BITS-sized passes

template <typename T>
void radix_sort_single_lsb(T* in, T* out, size_t N) {
    constexpr unsigned BITS = 11;
    constexpr unsigned BUCKETS = 1u << BITS;
    constexpr unsigned PASSES = (sizeof(T) * 8 + BITS - 1) / BITS;

    std::vector<uint32_t> hist(BUCKETS);
    std::vector<size_t> offsets(BUCKETS + 1);
    T* src = in;
    T* dst = out;

    for (unsigned pass = 0; pass < PASSES; ++pass) {
        unsigned shift = pass * BITS;
        std::fill(hist.begin(), hist.end(), 0);
        for (size_t i = 0; i < N; ++i) {
            uint32_t bucket = (src[i] >> shift) & (BUCKETS - 1);
            ++hist[bucket];
        }
        offsets[0] = 0;
        for (unsigned b = 0; b < BUCKETS; ++b) {
            offsets[b + 1] = offsets[b] + hist[b];
        }
        for (size_t i = 0; i < N; ++i) {
            uint32_t bucket = (src[i] >> shift) & (BUCKETS - 1);
            dst[offsets[bucket]++] = src[i];
        }
        std::swap(src, dst);
    }
    if (src != out) {
        std::copy(src, src + N, out);
    }
}

// Single-threaded MSD-based Radix Sort (skeleton)
// TODO: implement recursive or iterative most-significant-digit approach
template <typename T>
void radix_sort_single_msb(T* in, T* out, size_t N) {
    // Implementation pending: split by most significant bucket first,
    // recursively sort buckets or use an explicit stack.
    std::move(in, in + N, out);
}

// In-place Radix Sort (LSD or MSD) without extra buffers (skeleton)
// TODO: implement in-place algorithm using cycle-walking or index mapping
template <typename T>
void radix_sort_single_inplace(T* data, size_t N) {
    // Implementation pending: reorganize elements within data[]

}

// Multi-threaded Radix Sort (LSD or MSD) skeleton
// TODO: implement parallel histogram, prefix, and scatter phases
template <typename T>
void radix_sort_multi_threaded(T* in, T* out, size_t N, size_t threads) {
    // 1) Partition input among threads to compute local histograms
    // 2) Merge local histograms into global offsets
    // 3) Parallel scatter into out[] using the computed offsets
    std::move(in, in + N, out);
}

// NUMA/Chiplet-aware optimizations
// TODO: bind threads to cores and allocate buffers on local memory domains

// Explicit instantiations for uint64_t
template void radix_sort_single_lsb<uint64_t>(uint64_t*, uint64_t*, size_t);
template void radix_sort_single_msb<uint64_t>(uint64_t*, uint64_t*, size_t);
template void radix_sort_single_inplace<uint64_t>(uint64_t*, size_t);
template void radix_sort_multi_threaded<uint64_t>(uint64_t*, uint64_t*, size_t, size_t);
