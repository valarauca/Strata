// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu,unary.cuh,ssm-conv.cu,scale.cu}.
// Compile with --use_fast_math, as the pinned backend does.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/port.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include <cstddef>
#include <cstdint>

namespace strata::kernels {
namespace {
constexpr int S = 128;

__dpct_inline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            value, offset);
    return value;
}
__dpct_inline__ float norm_sum(float value, float *sums) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) % 32;
    value = warp_sum(value);
    if (lane == 0) sums[item_ct1.get_local_id(2) / 32] = value;
    item_ct1.barrier();
    value = lane < 8 ? sums[lane] : 0.0f;
    return warp_sum(value);
}
__dpct_inline__ float sigmoid(float value) {
    return 1.0f / (1.0f + sycl::native::exp(-value));
}

__dpct_inline__ void conv_silu(float *__restrict__ history,
                               const float *__restrict__ input,
                               const float *__restrict__ weights,
                               float *__restrict__ raw_output,
                               float *__restrict__ silu_output, int channels) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= channels) return;
    float values[4] = {history[c * 3], history[c * 3 + 1], history[c * 3 + 2], input[c]};
    float sum = 0.0f;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) sum += values[tap] * weights[c * 4 + tap];
    // The native SSM kernel adds its zero bias even when there is no bias input.
    sum = sum + 0.0f;
    raw_output[c] = sum;
    silu_output[c] = sum / (1.0f + sycl::native::exp(-sum));
#pragma unroll
    for (int tap = 0; tap < 3; ++tap) history[c * 3 + tap] = values[tap + 1];
}

__dpct_inline__ void l2_norm(float *input, float epsilon, float scale_after) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int col = item_ct1.get_local_id(2);
    input += size_t(item_ct1.get_group(2)) * S;
    const float value = col < S ? input[col] : 0.0f;
    float partial = 0.0f;
    if (col < S) partial += value * value;
    auto &sums =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    partial = norm_sum(partial, sums);
    const float scale = sycl::rsqrt(partial / S + epsilon);
    if (col < S) {
        // Preserve the FP32 store boundary between RMSNorm and ggml_scale.
        const float normalized = scale * value;
        input[col] = sycl::fma(normalized, scale_after, 0.0f);
    }
}

__dpct_inline__ void beta_sigmoid(float *beta, int count) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < count) beta[i] = sigmoid(beta[i]);
}
__dpct_inline__ void gate_softplus(const float *__restrict__ alpha,
                                   const float *__restrict__ dt,
                                   const float *__restrict__ ssm_a,
                                   float *__restrict__ gate, int count) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= count) return;
    const float value = alpha[i] + dt[i];
    const float softplus = value > 20.0f
                               ? value
                               : sycl::log1p(sycl::native::exp(
                                     value)); // 1 + e^v loses e^v below ~1e-7
    gate[i] = softplus * ssm_a[i];
}

__dpct_inline__ void out_norm(const float *__restrict__ input,
                              const float *__restrict__ z,
                              const float *__restrict__ gamma,
                              float *__restrict__ output, float epsilon) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int col = item_ct1.get_local_id(2);
    const size_t offset = size_t(item_ct1.get_group(2)) * S;
    const float value = col < S ? input[offset + col] : 0.0f;
    float partial = 0.0f;
    if (col < S) partial += value * value;
    auto &sums =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    partial = norm_sum(partial, sums);
    const float scale = sycl::rsqrt(partial / S + epsilon);
    if (col < S) {
        // RMSNorm+gamma is one pinned fused operator, followed by sigmoid*mul.
        const float weighted = scale * value * gamma[col];
        output[offset + col] = weighted * sigmoid(z[offset + col]);
    }
}

}

#include "native_gdn_preprocess.kern.inc"

} // namespace strata::kernels
