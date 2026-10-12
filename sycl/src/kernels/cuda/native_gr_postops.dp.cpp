// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{dsv4-hc.cu,scale.cu,unary.cu,unary.cuh}.
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
#include "strata/kernels/native_gr_postops.hpp"
#include <cstddef>
#include <cstdint>

namespace strata::kernels {
namespace {
__dpct_inline__ float sigmoid(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}
// ggml SCALE uses scale*x+bias, including its +0 bias. Retain this operation
// explicitly so compile-time zero does not change signed-zero behavior.
__dpct_inline__ float scale_zero_bias(float x, float scale) {
    return sycl::fma(scale, x, 0.0f);
}
__dpct_inline__ void down_silu(float *lo, int count, float scale) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const std::size_t i =
        std::size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= std::size_t(count)) return;
    const float x = scale_zero_bias(lo[i], scale);
    lo[i] = x / (1.0f + sycl::native::exp(-x));
}
template <bool Fused>
__dpct_inline__ void
pre_gated(const float *__restrict__ xn, float *__restrict__ gate,
          float *__restrict__ mixed, int n_embd, int hc, float scale) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const std::size_t d =
        std::size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (d >= std::size_t(n_embd)) return;
    const std::size_t t = item_ct1.get_group(1);
    xn += t * std::size_t(n_embd) * hc;
    gate += t * std::size_t(n_embd) * hc;
    mixed += t * std::size_t(n_embd);
    float sum = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const std::size_t i = std::size_t(c) * n_embd + d;
        const float x = xn[i], w = sigmoid(gate[i]);
        const float product = x * w;
        gate[i] = product;
        if constexpr (Fused) sum = sycl::fma(x, w, sum);
        else sum = c == 0 ? product : sum + product;
    }
    if constexpr (Fused) mixed[d] = scale * sum;
    else mixed[d] = scale_zero_bias(sum, scale);
}
__dpct_inline__ void post(const float *residual,
                          const float *__restrict__ block_out,
                          const float *__restrict__ inject, float *output,
                          int n_embd, int hc, float scale) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const std::size_t i =
        std::size_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= std::size_t(n_embd) * hc) return;
    const std::size_t t = item_ct1.get_group(1);
    residual += t * std::size_t(n_embd) * hc;
    block_out += t * std::size_t(n_embd);
    inject += t * std::size_t(hc);
    output += t * std::size_t(n_embd) * hc;
    const int c = int(i / n_embd), d = int(i % n_embd);
    const float weight = scale_zero_bias(sigmoid(scale_zero_bias(inject[c], scale)), 2.0f);
    // Exact residual/output alias is supported; no other thread reads residual[i].
    output[i] = sycl::fma(block_out[d], weight, residual[i]);
}
} // namespace

#include "native_gr_postops.kern.inc"

} // namespace strata::kernels
