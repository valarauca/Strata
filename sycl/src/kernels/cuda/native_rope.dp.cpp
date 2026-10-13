// Numerical contract: pinned ggml/src/ggml-cuda/rope.cu, rope_multi/rope_yarn.
// Text positions are equal across the four IMRoPE sections, so section routing
// reduces to the one position associated with each contiguous row.
// MIT License
//
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
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"
#include <cstdint>

namespace strata::kernels {
namespace {
// TAB (#280, STRATA_ROPE_TABLE=1): the angles from the session's float64 table.  The host launches <false> whenever
// no table applies - the default - so the default kernel is 0.1.31's code exactly (the table read is not in it;
// with it merely skipped at run time, the compiled default path changed its results).
template <bool TAB>
__dpct_inline__ void apply(const float *x, float *out, int rows, int width,
                           int n_rot, float theta_scale, float freq_scale,
                           float corr_low, float corr_high, float ext_factor,
                           float mscale, const int *positions,
                           const int32_t *mtab, RopeTab rt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(1);
    const int pair = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    if (row >= rows || pair >= width / 2) return;
    const size_t start = size_t(row) * width;
    if (pair >= n_rot / 2) {
        if (x != out) {
            out[start + 2 * pair] = x[start + 2 * pair];
            out[start + 2 * pair + 1] = x[start + 2 * pair + 1];
        }
        return;
    }
    float c, s;
    if (!(TAB && rope_tab_cs(rt, mrope_pos(mtab, positions[row], pair), pair, c, s))) {
        const float theta_extrap = mrope_pos(mtab, positions[row], pair) *
                                   dpct::pow(theta_scale, float(pair));
        rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
    }
    const float a = x[start + pair], b = x[start + pair + n_rot / 2];
    out[start + pair] = a * c - b * s;
    out[start + pair + n_rot / 2] = a * s + b * c;
}

template <bool TAB>
__dpct_inline__ void
norm_rope_kernel(const float *x, int in_stride, const float *__restrict__ gamma,
                 float *out, int n_cols, int n_rot, float epsilon,
                 float theta_scale, float freq_scale, float corr_low,
                 float corr_high, float ext_factor, float mscale,
                 const int *positions, const int32_t *mtab, RopeTab rt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid = item_ct1.get_local_id(2);
    const int row = item_ct1.get_group(2);
    const float* in_row = x + size_t(row) * in_stride;
    float* out_row = out + size_t(row) * n_cols;
    float partial = 0.0f;
    for (size_t col = tid; col < size_t(n_cols); col += 256) {
        const float xv = in_row[col];
        partial += xv * xv;
    }
    const int half_rot = n_rot / 2;
    const float x_tid = tid < n_cols ? in_row[tid] : 0.0f;
    const float x_hi  = tid < half_rot ? in_row[tid + half_rot] : 0.0f;
    auto &sums =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        partial += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            partial, offset);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    item_ct1.barrier();
    partial = lane < 256 / 32 ? sums[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        partial += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            partial, offset);
    const float mean = partial / n_cols;
    const float scale = sycl::rsqrt(mean + epsilon);
    if (tid >= n_rot && tid < n_cols) {
        out_row[tid] = (scale * x_tid) * gamma[tid];
    } else if (tid < half_rot) {
        const int pair = tid;
        float a = (scale * x_tid) * gamma[pair];
        float b = (scale * x_hi) * gamma[pair + half_rot];
#if defined(DPCT_COMPATIBILITY_TEMP) && !defined(__HIP_DEVICE_COMPILE__)
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        asm volatile("mov.f32 %0, %1;" : "=f"(a) : "f"(a));
#else
        a = a;
#endif
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        asm volatile("mov.f32 %0, %1;" : "=f"(b) : "f"(b));
#else
        b = b;
#endif
#endif
        float c, s;
        if (!(TAB && rope_tab_cs(rt, mrope_pos(mtab, positions[row], pair), pair, c, s))) {
            const float theta_extrap = mrope_pos(mtab, positions[row], pair) *
                                       dpct::pow(theta_scale, float(pair));
            rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
        }
        float bs = b * s;
        float bc = b * c;
#if defined(DPCT_COMPATIBILITY_TEMP) && !defined(__HIP_DEVICE_COMPILE__)
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        asm volatile("mov.f32 %0, %1;" : "=f"(bs) : "f"(bs));
#else
        bs = bs;
#endif
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
        asm volatile("mov.f32 %0, %1;" : "=f"(bc) : "f"(bc));
#else
        bc = bc;
#endif
#endif
        out_row[pair] = sycl::fma(a, c, -bs);
        out_row[pair + half_rot] = sycl::fma(a, s, bc);
    }
}
}

#include "native_rope.kern.inc"

}
