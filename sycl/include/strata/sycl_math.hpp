// include/strata/sycl_math.hpp - device math the hardware has (DP4A) and the sampler's accumulator type.
#pragma once
#include "strata/port.hpp"

#if defined(SYCL_LANGUAGE_VERSION)
#include <cstdint>

namespace strata {
// CUDA's __dp4a(int, int, int): four signed byte products summed into c. One place to change it: the byte
// unpack + multiply-add form is what IGC pattern-matches into the DP4A instruction (sycl::ext::oneapi::dot_acc is
// the same emulation and its header defines non-inline functions, which breaks the link across TUs).
template <typename A, typename B, typename C>
inline int32_t dp4a(A a, B b, C c) {
    return dpct::dp4a((int32_t) a, (int32_t) b, (int32_t) c);
}
}  // namespace strata


#endif  // SYCL_LANGUAGE_VERSION

namespace strata {
// The accumulator of the sampler's top_p / temperature tail (exp, sum, cumulative scan over the <= 64 kept logits).  FP64 is
// emulated on an Arc Alchemist (an A750: the tail of one row costs ~450 us against 20 us in float, which made sampled
// decode ~40% slower than greedy) and is not hardware on any Arc.  Float is the default; -DSTRATA_SYCL_SAMPLER_FP64=1
// keeps the reference's double.
#if defined(STRATA_SYCL_SAMPLER_FP64) && STRATA_SYCL_SAMPLER_FP64
using samp_acc_t = double;
#else
using samp_acc_t = float;
#endif
}  // namespace strata
