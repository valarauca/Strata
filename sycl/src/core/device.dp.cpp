// src/core/device.cu - P2.S1: the CUDA side of the runtime core.
#include "strata/port.hpp"
#include "strata/core/device.hpp"

namespace strata::core {

namespace {

// A NaN pattern, not zero.  Zeros read from uninitialised memory are indistinguishable from real zeros in a
// dequantized weight or a masked attention score, which is exactly the kind of wrong-but-plausible value the
// Phase 1 harnesses kept catching.
__dpct_inline__ void poison_kernel(float *p, uint64_t n_floats) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint64_t i =
        (uint64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n_floats) p[i] = sycl::bit_cast<float>(0x7fc00000);
}

}  // namespace

#include "device.kern.inc"

}  // namespace strata::core
