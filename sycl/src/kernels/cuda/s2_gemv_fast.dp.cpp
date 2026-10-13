// src/kernels/cuda/s2_gemv_fast.cu - two levers on the hottest kernel, BOTH MEASURED NEGATIVE OR NEUTRAL.
//
// *** THIS FILE IS NAMED "fast" AND IS NOT FASTER.  DO NOT ADOPT IT AS THE FAST PATH. ***
// It is kept because the negative result is worth more than the file costs: it closes two plausible
// optimisations and it is the experiment that established the measurement floor.
//
// L5 got the S2 GEMV from 10.5 to 86.2 G weights/s by amortising the loads, and left the instruction-count
// argument confirmed but unfinished.  Two further reductions were obvious, so both were built and measured:
//
// LEVER 1 - STAGE `x` IN SHARED MEMORY.  `x` is n_in halves (5 KB at n_embd 2560) and every output row reads
// all of it, so the generic kernel issues n_out * n_in activation loads per GEMV.  Staging it once per block
// measured WORSE in every single paired comparison (59-76 G w/s staged against 64-87 G w/s unstaged, across
// four sweep passes and three thread counts).  The __syncthreads and the copy cost more than the L1/L2 hits
// they replace - the same 5 KB is already resident and already broadcast.
//
// LEVER 2 - A CONSTANT-MEMORY CODE TABLE.  Turn the per-element shifts, masks and int-to-float convert into
// one broadcast `float4` load per four elements.  Measured NEUTRAL: the two orderings swap between passes
// with no consistent winner.
//
// WHY THE NEGATIVE RESULT IS THE REAL FINDING: run-to-run variance on this machine is now ~20% within a
// single configuration (quads tpr=32 measured 64.3 to 79.5 G w/s across four passes), because `verify` holds
// ~97% of six CPU cores.  **The noise floor now EXCEEDS the effect sizes being chased**, so further
// micro-optimisation of this kernel cannot be evaluated here and should wait for a quiet machine rather than
// produce more numbers that cannot be distinguished from noise.
//
// Both changes alter the summation order or the expression, so this kernel is checked against the naive
// reference like every other one - being faster is never a reason to be trusted, and being neutral is not
// either.
#include "strata/port.hpp"
#include "strata/kernels/s_gemv.hpp"

namespace strata::kernels {
namespace {

constexpr int QK_S2 = 64;
constexpr int MAX_SHARED_HALVES = 4096;      // 8 KB of shared for x; n_embd 2560 fits with room

// byte -> the four code values with the -1 bias already applied, in element order (bits 0,2,4,6).
struct alignas(16) CodeLut {
    float v[256][4];
};

constexpr CodeLut make_code_lut() {
    CodeLut t{};
    for (int b = 0; b < 256; ++b) {
        for (int k = 0; k < 4; ++k) {
            t.v[b][k] = (float) (((b >> (2 * k)) & 3) - 1);
        }
    }
    return t;
}

constexpr CodeLut c_codes_lut = make_code_lut();

// One block per output row.  `staged` says whether x was copied to shared, so the SAME kernel covers both
// configurations and the bench can measure them against each other with nothing else changed.
template <bool STAGE_X>
__dpct_inline__ void s2_gemv_fast_kernel(
    const uint16_t *__restrict__ x, const uint8_t *__restrict__ codes,
    const float *__restrict__ scales, float *__restrict__ y, long long n_in,
    long long n_out, int threads_per_row, uint8_t *dpct_local,
    const float (*c_codes)[4]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto smem = (float *)dpct_local;
    sycl::half *sx = reinterpret_cast<sycl::half *>(
        smem); // [threads_per_row, ...] partials live after
    float* partial = smem + (STAGE_X ? (MAX_SHARED_HALVES / 2) : 0);

    const long long o = item_ct1.get_group(2);
    if (o >= n_out) return;
    const int tid = item_ct1.get_local_id(2);

    if (STAGE_X) {
        const sycl::half *xh = reinterpret_cast<const sycl::half *>(x);
#pragma unroll
        for (long long i = tid; i < n_in; i += threads_per_row) sx[i] = xh[i];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;
    const float* s = scales + o * (n_in / QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];                          // ONE load for four codes
        const float d = s[q >> 4];                          // (q*4) >> 6
        const sycl::float4 cv = *reinterpret_cast<const sycl::float4 *>(
            &c_codes[byte][0]); // one broadcast load
        if (STAGE_X) {
            const sycl::half2 h01 =
                *reinterpret_cast<const sycl::half2 *>(&sx[q * 4]);
            const sycl::half2 h23 =
                *reinterpret_cast<const sycl::half2 *>(&sx[q * 4 + 2]);
            a0 += cv.x() * d * h01[0];
            a1 += cv.y() * d * h01[1];
            a2 += cv.z() * d * h23[0];
            a3 += cv.w() * d * h23[1];
        } else {
            const sycl::uint2 xw2 =
                *reinterpret_cast<const sycl::uint2 *>(x + q * 4);
            const sycl::half2 h01 =
                *reinterpret_cast<const sycl::half2 *>(&xw2.x());
            const sycl::half2 h23 =
                *reinterpret_cast<const sycl::half2 *>(&xw2.y());
            a0 += cv.x() * d * h01[0];
            a1 += cv.y() * d * h01[1];
            a2 += cv.z() * d * h23[0];
            a3 += cv.w() * d * h23[1];
        }
    }
    partial[tid] = (a0 + a1) + (a2 + a3);
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int step = threads_per_row / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (tid == 0) y[o] = partial[0];
}

}  // namespace

#include "s2_gemv_fast.kern.inc"

}  // namespace strata::kernels
