// src/kernels/cuda/shared_expert.cu - P2.S2: the shared expert.
//
// `ref/moe.py::shared_expert`, transcribed:
//
//     h = silu(x @ gate_shexp.T) * (x @ up_shexp.T)      <- SILU GOES ON GATE, not on up
//     h = h @ down_shexp.T                                (nt, n_embd)
//     g = sigmoid(x @ gate_inp_shexp)                     (nt,) ONE SCALAR PER TOKEN
//     return h * g[:, None]
//
// TWO THINGS HERE WERE BELIEVED WRONG FOR MANY ROUNDS and `docs/semantics.md` records both.  SILU GOES ON THE
// GATE TENSOR: the opposite reading is plausible, produces the right shapes, and is wrong.  And
// `ffn_gate_inp_shexp` is `(n_embd,)` whose output is "one value per token" - a SCALAR gate obtained by dotting
// that vector with the hidden state, NOT a per-expert gate and NOT a per-dimension elementwise one.  Both were
// live readings of the same shapes until the source comment settled them.
//
// The result is ADDED to the routed output - not router-weighted, not renormalised against it.
//
// The legacy canonical projections use their explicit Q8_0/Q8_K activation images. The optional native
// BF16 path affects only the scalar gate: it reads the original F32 input and uses pinned CUDA MMVF plus
// an FP32 sigmoid. Native projection overrides independently select CUDA Q8_1 MMVQ and FP32 SwiGLU.
#include "strata/port.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"

namespace strata::kernels {
namespace {

// `f32_to_f16` used to live here as a private copy.  Round 198 found it - and the identical one in
// `quantize_act.cu` - returning a NaN for every FINITE value that overflows fp16 (1e30, 65536, 1e45) instead
// of saturating to inf, because `exp >= 31` conflates an f32 inf/NaN with an out-of-range finite exponent.
// This file's fixture is all O(1), so it could not see it.  The conversion now lives in one place, with the
// regimes spelled out.

// silu(x) = x / (1 + exp(-x)), written as `ref/moe.py` writes it, in DOUBLE then cast - the reference works in
// float64 and a float32 exp differs in the last bits.  The multiplication by `up` is the reference's order.
__dpct_inline__ void swiglu_kernel(const float *__restrict__ gate,
                                   const float *__restrict__ up,
                                   float *__restrict__ out, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    const double x = (double) gate[i];
    out[i] = (float)(x / (1.0 + sycl::exp(-x))) * up[i];
}

__dpct_inline__ void native_swiglu_kernel(const float *gate, const float *up,
                                          float *out, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    // Pinned CUDA unary.cuh op_silu, then unary_gated_op_kernel's multiply.
    // Explicit intrinsics reproduce its --use_fast_math operations without
    // changing compilation of the default legacy arithmetic in this file.
    out[i] = gate[i] / (1.0f + sycl::native::exp(-gate[i])) * up[i];
}


void to_f16_kernel(const float* __restrict__ in, uint16_t* __restrict__ out, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) out[i] = f16_from_f32(in[i]);
}

// The per-token scalar gate: sigmoid(dot(x, w)) with w = `ffn_gate_inp_shexp` (n_embd,).
//
// **ONE THREAD WAS THE LARGEST SINGLE KERNEL IN THE MODEL, AND THE COMMENT DEFENDING IT SAID WHY IT SHOULD
// NOT BE.**  It read: "n_embd is 2560, so this is a short serial loop and the parallelism is elsewhere."
// 2560 iterations is not short when each one is a DOUBLE multiply-add.  This card's double rate is **1/64** of
// its float rate, so the loop is a 2560-long dependency chain through a 1/64-rate unit - tens of microseconds
// from the chain alone, before counting 2560 sequential global loads issued by one thread with no coalescing.
//
// MEASURED, and this is what the number looks like from outside: `shared_expert` is **0.2374 ms of a
// 0.8883 ms block - 26.7% of the whole block, 11.4 ms per token** - for three projections that read 5.5 MB and
// whose memory floor is 0.009 ms.  It is seven kernels, and six of them are GEMVs or elementwise passes over
// 640-2560 elements.  This is the one that cannot be explained by bandwidth.
//
// THE DOUBLE ACCUMULATOR IS KEPT.  `ref/gr.py` reduces in float64 and `block_sum` in `gr.cu` exists for the
// same reason - the summation order of a tree is not the reference's, so the defence is enough precision that
// the order stops mattering.  What changes is the CHAIN: ten terms per thread instead of 2560, then a
// tree-reduce.  Through a sigmoid, a 1e-16 relative difference in the dot is not a difference.
//
// **BOTH OPERANDS ARE BF16, AND THAT IS THE REFERENCE'S CONTRACT, NOT A STORAGE DETAIL.**  `gate_inp_shexp` is a
// BF16 weight, so ggml converts the activation to its `vec_dot_type` - BF16 - exactly as it does for the router
// (`docs/activation-contract.md`).  This kernel used to take `const float* w` while the LOADER had already
// re-rounded the tensor to 2 B/elem, so it read 2560 floats out of a 5120-byte buffer: 5120 B past the end of
// the tensor, inside the 4.5 GiB arena, where nothing faults.  The dot came out astronomically large, sigmoid
// saturated to 1.0, and `scale_kernel` multiplied the shared expert's output by it - which is why the FIRST
// symptom was 2560 non-finite outputs rather than a wrong scalar.
//
// A wrong scalar here is the quiet failure mode: sigmoid bounds the damage to [0,1], so a gate that should be
// 0.5 and reads 1.0 scales the shared expert by 2x and produces perfectly finite, perfectly plausible logits.
__dpct_inline__ double warp_sum_d(double v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v +=
        dpct::experimental::shift_sub_group_left(
            0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            off);
    return dpct::experimental::select_from_sub_group(
        0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, 0);
}

__dpct_inline__ void scalar_gate_kernel(const uint16_t *__restrict__ x_bf16,
                                        const uint16_t *__restrict__ w_bf16,
                                        float *__restrict__ out, int n_embd) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &scratch = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<
        3>()); // 8 warps: the launch is <<<1, 256>>>
    double acc = 0.0;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n_embd;
         i += item_ct1.get_local_range(2))
        acc += (double) f32_from_bf16(x_bf16[i]) * (double) f32_from_bf16(w_bf16[i]);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    acc = warp_sum_d(acc);
    if (lane == 0) scratch[warp] = acc;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int nw = ((int)item_ct1.get_local_range(2) + 31) >> 5;
    if (warp == 0) {
        acc = (item_ct1.get_local_id(2) < nw)
                  ? scratch[item_ct1.get_local_id(2)]
                  : 0.0;
        acc = warp_sum_d(acc);
        if (item_ct1.get_local_id(2) == 0) out[0] =
            (float)(1.0 / (1.0 + sycl::exp(-acc)));
    }
}

__dpct_inline__ void scale_kernel(float *__restrict__ out,
                                  const float *__restrict__ g, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) out[i] *= g[0];
}

__dpct_inline__ void native_scalar_sigmoid_kernel(float *gate) {
    // Match the single-token CUDA sigmoid's FP32 fast-math operations without changing legacy kernels'
    // compilation flags. The dot product was already reduced by the pinned native MMVF implementation.
    gate[0] = 1.0f / (1.0f + sycl::native::exp(-gate[0]));
}
void native_scalar_sigmoid_multi_kernel(float* gate) {   // thread t = token t, same expression
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    gate[item_ct1.get_local_id(2)] =
        1.0f / (1.0f + sycl::native::exp(-gate[item_ct1.get_local_id(2)]));
}

/// The MoE block's final combination.  See the header for the two readings it exists to pin.
__dpct_inline__ void moe_combine_kernel(const float *__restrict__ parts,
                                        const float *__restrict__ weights,
                                        const float *__restrict__ shared,
                                        float *__restrict__ y, int n_embd,
                                        int k, int has_shared) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (j >= n_embd) return;
    // Accumulate in DOUBLE, in the reference's own order (`for i in range(k): out[t] += w[t,i] * g[0]`).
    // f32 would be defensible for ten terms, but the reference is float64 and this is one line.
    double acc = 0.0;
#pragma unroll
    for (int e = 0; e < k; ++e) acc +=
        (double)weights[e] * (double)parts[(size_t)e * n_embd + j];
    // The SHARED output is added PLAIN - not router-weighted, not renormalised against the routed sum.
    if (has_shared) acc += (double) shared[j];
    y[j] = (float) acc;
}

}  // namespace

namespace {
__dpct_inline__ void scale_rows_kernel(float *__restrict__ out,
                                       const float *__restrict__ g, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) out[(size_t) t * n + i] *= g[t];
}

__dpct_inline__ void sigmoid_scale_rows_kernel(float *__restrict__ out,
                                               const float *__restrict__ g,
                                               int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) {
        const float gt = 1.0f / (1.0f + sycl::native::exp(-g[t]));
        out[(size_t) t * n + i] *= gt;
    }
}

__dpct_inline__ void
sigmoid_scale_rows_vec4_kernel(sycl::float4 *__restrict__ out4,
                               const float *__restrict__ g, int n4) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n4) {
        const float gt = 1.0f / (1.0f + sycl::native::exp(-*(g + t)));
        sycl::float4 v = out4[(size_t)t * n4 + i];
        v.x() *= gt;
        v.y() *= gt;
        v.z() *= gt;
        v.w() *= gt;
        out4[(size_t) t * n4 + i] = v;
    }
}

}  // namespace

#include "shared_expert.kern.inc"

}  // namespace strata::kernels
