// src/kernels/cuda/bf16_gemv.cu - the BF16 GEMV.  See the header for why it is not `s_gemv`.
#include "strata/port.hpp"
#include "strata/kernels/bf16_gemv.hpp"

#include "strata/kernels/bf16_bits.hpp"

namespace strata::kernels {
namespace {

/// One thread per output row, walking it contiguously.  Kept as the naive reference the split version is
/// checked against, and used directly for the shapes where the output width is already large.
__dpct_inline__ void bf16_gemv_naive_kernel(const uint16_t *__restrict__ x,
                                            const uint16_t *__restrict__ w,
                                            float *__restrict__ y,
                                            long long n_in, long long n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long o =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (o >= n_out) return;
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
#pragma unroll
    for (long long i = 0; i < n_in; ++i)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    y[o] = acc;
}

/// One WARP per output row, lanes striding the reduction axis.  For a fixed `i` consecutive lanes touch
/// consecutive addresses in this layout, so the load is coalesced - the property that took `gr_read` from
/// 262 ms/token to 8.9.
__dpct_inline__ void bf16_gemv_warp_kernel(const uint16_t *__restrict__ x,
                                           const uint16_t *__restrict__ w,
                                           float *__restrict__ y,
                                           long long n_in, long long n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warps_per_block = (int)(item_ct1.get_local_range(2) >> 5);
    const long long o = (long long)item_ct1.get_group(2) * warps_per_block +
                        (item_ct1.get_local_id(2) >> 5);
    if (o >= n_out) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
#pragma unroll
    for (long long i = lane; i < n_in; i += 32)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc +=
        dpct::experimental::shift_sub_group_left(
            0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            acc, off);
    if (lane == 0) y[o] = acc;
}

/// TPR lanes cooperate on ONE row's reduction: `threads_per_row` threads each take a strided slice and the
/// block reduces through shared memory.  Used when the output width is too small to fill the machine.
__dpct_inline__ void bf16_gemv_split_kernel(const uint16_t *__restrict__ x,
                                            const uint16_t *__restrict__ w,
                                            float *__restrict__ y,
                                            long long n_in, long long n_out,
                                            int tpr, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto scratch = (float *)dpct_local;
    const long long o = item_ct1.get_group(2);
    if (o >= n_out) return;
    const int t = item_ct1.get_local_id(2); // 0 .. tpr-1
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
#pragma unroll
    for (long long i = t; i < n_in; i += tpr)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    // block reduction; `tpr` is at most a few hundred, so a tree in shared is enough
    scratch[t] = acc;
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int off = tpr >> 1; off > 0; off >>= 1) {
        if (t < off) scratch[t] += scratch[t + off];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (t == 0) y[o] = scratch[0];
}

}  // namespace

#include "bf16_gemv.kern.inc"

}  // namespace strata::kernels
