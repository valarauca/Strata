// src/kernels/cuda/dequant_s2.cu - P2.S2's first naive kernel.
//
// S2 is Q2_0's canonical form and 31.64 GiB of the artifact is Q2_0, so this is the single hottest decode in
// the engine.  The canonical plane layout is `docs/pack-format.md`'s: codes packed 4 per byte LSB-first
// (element i at byte i/4, bits (i%4)*2 - Q2_0's own convention, kept for every form) and one FP32 scale per
// group (64 elements for S2).  The decode is `(code + code_bias) * scale` with `code_bias = -1`, the
// subtraction inside the INTEGER domain, which is what makes it bit-exact (docs/pack-format.md §3.1).
//
// Naive on purpose: one thread per block, no vectors, no shared memory, no unrolling.  Phase 3 changes this
// file and the parity test in src/kernels/dequant_s2_parity.cpp is what says whether a change is still right.
#include "strata/port.hpp"
#include "strata/kernels/dequant_s2.hpp"

namespace strata::kernels {
namespace {

constexpr int QK = 64;             // S2 group = Q2_0 block = 64 elements
constexpr int CODES_PER_BYTE = 4;

__dpct_inline__ void dequant_s2_kernel(const uint8_t *__restrict__ codes,
                                       const float *__restrict__ scales,
                                       float *__restrict__ out,
                                       long long n_blocks) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long b =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (b >= n_blocks) return;
    const float d = scales[b];
    const uint8_t* c = codes + b * (QK / CODES_PER_BYTE);
    float* y = out + b * QK;
    for (int j = 0; j < QK; ++j) {
        const int code = (c[j / CODES_PER_BYTE] >> ((j % CODES_PER_BYTE) * 2)) & 0x03;
        y[j] = (float)(code - 1) * d;      // the -1 is applied to the CODE, then ONE multiply
    }
}

}  // namespace

#include "dequant_s2.kern.inc"

}  // namespace strata::kernels
