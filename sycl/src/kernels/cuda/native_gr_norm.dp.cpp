// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh}. Scope: contiguous weighted F32 RMSNorm.
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
#include "strata/kernels/native_gr_norm.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels {
namespace {

__dpct_inline__ float norm_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            value, offset);
    }
    return value;
}

template <int BlockSize>
__dpct_inline__ void weighted_rms_norm(const float *__restrict__ input,
                                       const float *__restrict__ gamma,
                                       float *__restrict__ output, int n_cols,
                                       float epsilon) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid = item_ct1.get_local_id(2);
    const std::size_t row =
        std::size_t(item_ct1.get_group(1)) * item_ct1.get_group_range(2) +
        item_ct1.get_group(2);
    const std::size_t row_offset = row * n_cols;
    input += row_offset;
    gamma += std::size_t(item_ct1.get_group(2)) * n_cols;
    output += row_offset;
    float partial = 0.0f;
    for (int col = tid; col < n_cols; col += BlockSize) {
        const float value = input[col];
        partial += value * value;
    }

    // Pinned block_reduce<SUM,BlockSize>: every warp repeats the final XOR
    // reduction. There is no warp-0-only broadcast or downward-shuffle tree.
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

    const float mean = partial / n_cols;
    const float scale = sycl::rsqrt(mean + epsilon);
#pragma unroll
    for (int col = tid; col < n_cols; col += BlockSize) {
        output[col] = scale * input[col] * gamma[col];
    }
}

} // namespace

#include "native_gr_norm.kern.inc"

} // namespace strata::kernels
