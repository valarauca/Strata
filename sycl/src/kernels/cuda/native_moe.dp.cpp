// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
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
#include "strata/kernels/native_moe.hpp"
#include <cstddef>
#include <cstdint>

namespace strata::kernels {
namespace {
// GATED (S26 STRATA_LFUSE): `shared` is the shared expert's unscaled output and sg its raw gate logit per token;
// the row is scaled here as shared_expert_multi did (native_scalar_sigmoid_multi_kernel's expression, then one
// rounded multiply = scale_rows_kernel's `out *= g`), then added as before
template <bool ZADD = false, bool GATED = false>
__dpct_inline__ void
combine(const float *__restrict__ parts, const float *__restrict__ weights,
        const float *__restrict__ shared, float *__restrict__ output,
        int64_t n_embd, int k, const float *__restrict__ sg = nullptr) {
    // blockIdx.y = the token of a multi-token launch (0 for the single one)
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t tk = item_ct1.get_group(1);
    parts += tk * k * n_embd; weights += tk * k; if (shared) shared += tk * n_embd; output += tk * n_embd;
    const int64_t col =
        int64_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (col >= n_embd) return;
    // ZADD: a part is 0.0f + hit (the zeroed row plus moe_hit_add of the verify window), rounded as that add
    float p0 = parts[col];
    if (ZADD) p0 = 0.0f + p0;
    float sum = p0 * weights[0];
    for (int expert = 1; expert < k; ++expert) {
        float p = parts[int64_t(expert) * n_embd + col];
        if (ZADD) p = 0.0f + p;
        sum += p * weights[expert];
    }
    if constexpr (GATED) {
        // contraction off in this block only: the product is rounded on its own (scale_rows' store), then added
#pragma clang fp contract(off)
        const float g = 1.0f / (1.0f + sycl::native::exp(-sg[tk]));
        const float sh = shared[col] * g;
        sum += sh;
    } else {
        if (shared) sum += shared[col];
    }
    output[col] = sum;
}
__dpct_inline__ void combine_k10_vec4(const sycl::float4 *__restrict__ parts4,
                                      const float *__restrict__ weights,
                                      const sycl::float4 *__restrict__ shared4,
                                      sycl::float4 *__restrict__ output4,
                                      int64_t n4) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t tk = item_ct1.get_group(1);
    parts4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 =
        int64_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (c4 >= n4) return;
    const float w0 = *(weights + 0);
    const sycl::float4 p0 = parts4[c4];
    sycl::float4 sum =
        sycl::float4(p0.x() * w0, p0.y() * w0, p0.z() * w0, p0.w() * w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        const float w = *(weights + expert);
        const sycl::float4 p = parts4[int64_t(expert) * n4 + c4];
        // the documented contract spelled out: the first product rounded, then one FMA per expert in order. Left to the
        // compiler's contraction, a build may fuse a different product (gfx1151: the float4 kernel then differed from the
        // scalar one, native_multi_parity)
        sum.x() = sycl::fma((float)(p.x()), (float)w, sum.x());
        sum.y() = sycl::fma((float)(p.y()), (float)w, sum.y());
        sum.z() = sycl::fma((float)(p.z()), (float)w, sum.z());
        sum.w() = sycl::fma((float)(p.w()), (float)w, sum.w());
    }
    const sycl::float4 sh = shared4[c4];
    sum.x() += sh.x();
    sum.y() += sh.y();
    sum.z() += sh.z();
    sum.w() += sh.w();
    output4[c4] = sum;
}
}

#include "native_moe.kern.inc"

}
