// Arithmetic adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp and ggml-cuda/{reduce_rows.cuh,sumrows.cu,unary.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/port.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/ngram.hpp"

namespace strata::kernels {
namespace {
constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;
inline float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) x +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            offset);
    return x;
}
__dpct_inline__ void gate_kernel(const float *key, const float *query,
                                 float *gate, float scale) {
    // SUM_ROWS selects 512 threads for four rows on the target GPU. Preserve
    // its eight partial lanes and materialized MUL rounding (never a dot FMA).
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    float sums[8] = {};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = item_ct1.get_local_id(2) + j * 512;
        const float p = d < N ? key[item_ct1.get_group(2) * N + d] *
                                    query[item_ct1.get_group(2) * N + d]
                              : 0.0f;
        sums[j] += p;
    }
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += sums[j];
    auto &partials =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    sum = warp_sum(sum);
    const int lane = item_ct1.get_local_id(2) % 32;
    if (!lane) partials[item_ct1.get_local_id(2) / 32] = sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    sum = lane < 16 ? partials[lane] : 0.0f;
    sum = warp_sum(sum);
    if (item_ct1.get_local_id(2) == 0) {
        const float s = sycl::fma(scale, sum, 0.0f); // ggml SCALE's zero bias
        const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(s), 1e-6f));
        const float sign = float((s > 0.0f) - (s < 0.0f));
        gate[item_ct1.get_group(2)] =
            1.0f / (1.0f + sycl::native::exp(-sign * mag));
    }
}
__dpct_inline__ void broadcast_kernel(const float *value, const float *gate,
                                      float *gated) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < D) gated[i] = value[i % N] * gate[i / N];
}
__dpct_inline__ void
conv_residual_kernel(const float *history, const float *normalized,
                     const uint16_t *weights, const float *hidden,
                     const float *gated, float *conv, float *result) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= D) return;
    float sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float x = k == 3 ? normalized[c] : history[c * HISTORY + 3 * k];
        const float w =
            sycl::vec<sycl::half, 1>(
                sycl::bit_cast<sycl::half, unsigned short>(weights[c * 4 + k]))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const float term = x * w;
        sum = k == 0 ? term : sum + term;
    }
    const float activation = sum / (1.0f + sycl::native::exp(-sum));
    conv[c] = activation;
    // Exact hidden/result alias is safe: each thread owns one element.
    result[c] = hidden[c] + gated[c] + activation;
}
// ---- the batch: T tokens, the same arithmetic per element as the kernels above
// weighted_rms_norm (native_gr_norm.cu) with the gamma row repeating every H rows (one token's H groups)
inline float norm_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            value, offset);
    return value;
}
__dpct_inline__ void rms_rep_kernel(const float *__restrict__ input,
                                    const float *__restrict__ gamma,
                                    float *__restrict__ output) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int BlockSize =
        1024; // native_gr_rms_norm_weighted's choice for 2560 columns
    const int tid = item_ct1.get_local_id(2);
    const size_t row_offset = size_t(item_ct1.get_group(2)) * N;
    input += row_offset;
    output += row_offset;
    gamma += size_t(item_ct1.get_group(2) % H) * N;
    float partial = 0.0f;
    for (int col = tid; col < N; col += BlockSize) {
        const float value = input[col];
        partial += value * value;
    }
    auto &sums =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    partial = norm_warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    partial = 0.0f;
    if (lane < BlockSize / 32) partial = sums[lane];
    partial = norm_warp_sum(partial);
    const float mean = partial / N;
    const float scale = sycl::rsqrt(mean + NG_RMS_EPS);
#pragma unroll
    for (int col = tid; col < N; col += BlockSize)
        output[col] = scale * input[col] * gamma[col];
}
__dpct_inline__ void broadcast_batch_kernel(const float *value,
                                            const float *gate, float *gated,
                                            int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= size_t(T) * D) return;
    const size_t t = i / D, d = i % D;
    gated[i] = value[t * N + d % N] * gate[t * H + d / N];
}
// the dilated conv (taps 9, 6, 3 tokens back and this one) and the residual; a tap before the chunk reads the history
__dpct_inline__ void conv_residual_batch_kernel(const float *history,
                                                const float *normalized,
                                                const uint16_t *weights,
                                                float *hidden,
                                                const float *gated, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const size_t i =
        size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= size_t(T) * D) return;
    const int t = int(i / D), c = int(i % D);
    float sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int p = t - 9 + 3 * k;             // the token this tap reads (k == 3: this one)
        const float x = p >= 0 ? normalized[size_t(p) * D + c] : history[size_t(c) * HISTORY + (9 + p)];
        const float wk =
            sycl::vec<sycl::half, 1>(
                sycl::bit_cast<sycl::half, unsigned short>(weights[c * 4 + k]))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const float term = x * wk;
        sum = k == 0 ? term : sum + term;
    }
    const float activation = sum / (1.0f + sycl::native::exp(-sum));
    hidden[i] = hidden[i] + gated[i] + activation;
}
// the history after the chunk: the last nine normalized rows (older ones from the history when T < 9)
__dpct_inline__ void history_batch_kernel(float *history,
                                          const float *normalized, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= D) return;
    float h[HISTORY];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) {
        const int p = T - HISTORY + r;
        h[r] = p >= 0 ? normalized[size_t(p) * D + c] : history[size_t(c) * HISTORY + (T + r)];
    }
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) history[size_t(c) * HISTORY + r] = h[r];
}
} // namespace

#include "native_ple_postops.kern.inc"

} // namespace strata::kernels
