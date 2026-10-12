// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
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
#include "strata/sycl_math.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "s26_tsum.dp.hpp"

#include <cstdint>

namespace strata::kernels {
namespace {

#ifndef STRATA_MMVQ_UNROLL
#define STRATA_MMVQ_UNROLL 2
#endif
constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;

struct Q5KBlock {
    sycl::half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    sycl::half2 ds;
    int8_t qs[32];
};
struct Q20Block {
    sycl::half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    sycl::half d;
};
struct IQ4XSBlock {
    sycl::half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    sycl::half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    sycl::half d;
};
struct Q40Block {
    sycl::half d;
    uint8_t qs[16];
};
struct Q50Block {
    sycl::half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    sycl::half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    sycl::half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

__dpct_inline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += dpct::experimental::permute_sub_group_by_xor(
            0xffffffff, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            offset);
    }
    return x;
}

template <int SG> __dpct_inline__ float sg_sum(float x) {   // SYCL port: warp_sum for a SG-wide sub-group
#pragma unroll
    for (int offset = SG / 2; offset > 0; offset >>= 1)
        x += sycl::permute_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset);
    return x;
}

__dpct_inline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = sycl::fmax(x,
                       dpct::experimental::permute_sub_group_by_xor(
                           0xffffffff,
                           sycl::ext::oneapi::this_work_item::get_sub_group(),
                           x, offset));
    }
    return x;
}

__dpct_inline__ void native_quantize_q8_1_kernel(const float *__restrict__ x,
                                                 Q81Block *__restrict__ y,
                                                 int n_in) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = int(item_ct1.get_group(2)) * QUANT_THREADS +
                  int(item_ct1.get_local_id(2));
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(sycl::fabs(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

__dpct_inline__ void
native_swiglu_quantize_q8_1_kernel(const float *__restrict__ gate,
                                   const float *__restrict__ up,
                                   Q81Block *__restrict__ y, int n_in) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = int(item_ct1.get_group(2)) * QUANT_THREADS +
                  int(item_ct1.get_local_id(2));
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float gi = gate[i];
    const float xi = gi / (1.0f + sycl::native::exp(-gi)) * up[i];
    const float amax = warp_max(sycl::fabs(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606, as native_quantize_q8_1_kernel: finite blocks bit for bit
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__dpct_inline__ float
q5_q8_dot_impl(const int *__restrict__ vl, const int *__restrict__ vh,
               const int *__restrict__ u, const uint8_t *__restrict__ sc,
               const uint8_t *__restrict__ m, const sycl::half2 &dm5,
               const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = STRATA_DP4A(v0i, u[2 * i], STRATA_DP4A(v1i, u[2 * i + 1], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i], STRATA_DP4A(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const sycl::float2 dm5f =
        dm5.template convert<float, sycl::rounding_mode::automatic>();
    return dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}

__dpct_inline__ float q5_q8_dot(const Q5KBlock *__restrict__ bq5,
                                const Q81Block *__restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = bq8i->ds[0];
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, m, bq5->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template <bool SmallK>
__dpct_inline__ void native_q5_k_mmvq_kernel(const Q5KBlock *__restrict__ w,
                                             const Q81Block *__restrict__ x,
                                             float *__restrict__ y, int n_in,
                                             int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.
// The weight block is only 2-byte aligned, so qs is intentionally loaded as
// int16_t, unlike the naturally 4-byte aligned activation codes.
__dpct_inline__ float q2_q8_dot(const Q20Block *__restrict__ w,
                                const Q81Block *__restrict__ x, int iqs) {
    const float d2 = w->d;
    const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
    const Q81Block* chunk = x + iqs;
    const int* q8 = reinterpret_cast<const int*>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2];
        const int v = q8[j * 2 + 1];
        const int qe = dpct::byte_level_permute(0x020100ff, 0x020100ff, q >> 0);
        const int qo = dpct::byte_level_permute(0x020100ff, 0x020100ff, q >> 2);
        const int qx = dpct::byte_level_permute(qe, qo, 0x5140);
        const int qy = dpct::byte_level_permute(qe, qo, 0x7362);
        sumi = STRATA_DP4A(u, qx, sumi);
        sumi = STRATA_DP4A(v, qy, sumi);
    }
    const float d8 = chunk->ds[0];
    return d2 * d8 * sumi;
}

// Q2_0 generic MMVQ: QK=64, QI=2, VDR=1, 64 blocks per iteration.
// Preserve the same outer accumulation and cross-warp reduction as the oracle.
template <bool SmallK>
__dpct_inline__ void native_q2_0_mmvq_kernel(const Q20Block *__restrict__ w,
                                             const Q81Block *__restrict__ x,
                                             float *__restrict__ y, int n_in,
                                             int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 2;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 64;
    float tmp[ROWS] = {};
    for (int kbx = tid / 2; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 2;
        const int kqs = tid % 2;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q2_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
__dpct_inline__ int load_int_b2(const void *ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__dpct_inline__ float q3_q8_dot_impl(int vl, int vh, const int *__restrict__ u,
                                     const uint8_t *__restrict__ scales,
                                     int scale_offset, float d3,
                                     const float *__restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi =
            dpct::vectorized_binary<sycl::char4>(vil, vih, dpct::sub_sat());
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

__dpct_inline__ float q3_q8_dot(const Q3KBlock *__restrict__ w,
                                const Q81Block *__restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = w->d;
    const int vl = load_int_b2(w->qs, iqs);
    const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        d8[i] = x[bq8_offset + i].ds[0];
    }
    return q3_q8_dot_impl(vl, vh, u, w->scales, scale_offset, d, d8);
}

// Q3_K generic MMVQ: QK=256, QI=16, VDR=1, eight blocks per iteration.
template <bool SmallK>
__dpct_inline__ void native_q3_k_mmvq_kernel(const Q3KBlock *__restrict__ w,
                                             const Q81Block *__restrict__ x,
                                             float *__restrict__ y, int n_in,
                                             int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 16;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 16;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q3_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// The pinned nonlinear IQ4 codebook and its CUDA two-stage byte lookup. The
// explicit alignment satisfies the four 32-bit table loads; values are unchanged.
alignas(4) static constexpr int8_t kIq4nlTable[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__dpct_inline__ sycl::int2 iq4_table_lookup(int q4, int8_t *iq4nl_values) {
#if defined(STRATA_HIP_GFX906)
    // AMD: llama.cpp's HIP lookup (see iq_kernels.cu get_int_from_table_16): 4 v_perm_b32 per 8 values
    const uint32_t* v32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    const uint32_t q_even = (uint32_t) q4, q_odd = (uint32_t) q4 >> 4;
    const uint32_t el = __builtin_amdgcn_perm(v32[1], v32[0], q_even & 0x07070707u);
    const uint32_t ol = __builtin_amdgcn_perm(v32[1], v32[0], q_odd & 0x07070707u);
    const uint32_t eh = __builtin_amdgcn_perm(v32[3], v32[2], q_even & 0x07070707u);
    const uint32_t oh = __builtin_amdgcn_perm(v32[3], v32[2], q_odd & 0x07070707u);
    return make_int2((int) __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1)),
                     (int) __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1)));
#else
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low =
            dpct::byte_level_permute(table32[0], table32[1], q4 >> shift);
        const uint32_t high =
            dpct::byte_level_permute(table32[2], table32[3], q4 >> shift);
        tmp[i] = dpct::byte_level_permute(low, high,
                                          low_high_selection_indices >> shift);
    }
    return sycl::int2(dpct::byte_level_permute(tmp[0], tmp[1], 0x6420),
                      dpct::byte_level_permute(tmp[0], tmp[1], 0x7531));
#endif
}

// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock,
// computes integer dot products, applies signed scale in the integer domain,
// then multiplies the two half scales and integer sum in the original order.
__dpct_inline__ float iq4_xs_q8_dot(const IQ4XSBlock *__restrict__ w,
                                    const Q81Block *__restrict__ x, int iqs,
                                    int8_t *iq4nl_values) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = reinterpret_cast<const int*>(w->qs)[iqs + j];
        const sycl::int2 v = iq4_table_lookup(aux_q4, iq4nl_values);
        const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
        const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        sumi = STRATA_DP4A(v.x(), u0, sumi);
        sumi = STRATA_DP4A(v.y(), u1, sumi);
    }
    const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
                   (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = sycl::vec<sycl::half, 1>(w->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    x[iqs / 4].ds[0];
    return d * sumi;
}

// IQ4_XS generic MMVQ: QK=256, QI=32, VDR=4,16 blocks per iteration.
template <bool SmallK>
__dpct_inline__ void
native_iq4_xs_mmvq_kernel(const IQ4XSBlock *__restrict__ w,
                          const Q81Block *__restrict__ x, float *__restrict__ y,
                          int n_in, int n_out, int8_t *iq4nl_values) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 4 * WARPS * WARP / 32;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 8; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = 4 * (tid % 8);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += iq4_xs_q8_dot(w + block, x + kby, kqs, iq4nl_values);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
__dpct_inline__ float
q4_q8_dot_impl(const int *__restrict__ v, const int *__restrict__ u,
               const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
               const sycl::half2 &dm4, const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = STRATA_DP4A(v1i, u[2 * i + 1], STRATA_DP4A(v0i, u[2 * i], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i + 1], STRATA_DP4A(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const sycl::float2 dm4f =
        dm4.template convert<float, sycl::rounding_mode::automatic>();
    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

__dpct_inline__ float q4_q8_dot(const Q4KBlock *__restrict__ bq4,
                                const Q81Block *__restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = bq8i->ds[0];
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, m, bq4->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template <bool SmallK>
__dpct_inline__ void native_q4_k_mmvq_kernel(const Q4KBlock *__restrict__ w,
                                             const Q81Block *__restrict__ x,
                                             float *__restrict__ y, int n_in,
                                             int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q4_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
__dpct_inline__ float q6_q8_dot_impl(int vl, int vh, const int *__restrict__ u,
                                     const int8_t *__restrict__ scales, float d,
                                     const float *__restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = dpct::vectorized_binary<sycl::char4>(
            vil | vih, 0x20202020, dpct::sub_sat());
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

__dpct_inline__ float q6_q8_dot(const Q6KBlock *__restrict__ w,
                                const Q81Block *__restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = x[bq8_offset + 2 * i].ds[0];
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}

// Q6_K generic MMVQ: QK=256, QI=32, VDR=1, four blocks per iteration.
template <bool SmallK>
__dpct_inline__ void native_q6_k_mmvq_kernel(const Q6KBlock *__restrict__ w,
                                             const Q81Block *__restrict__ x,
                                             float *__restrict__ y, int n_in,
                                             int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 32;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q6_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// The four 32-element formats use native two-byte loads and VDR=2. The affine
// Q4_0/Q5_0 correction consumes the original-input sum stored in Q8_1, exactly
// as the pinned CUDA dot does; a signed-integer code substitution would differ.
__dpct_inline__ float small_q8_dot(const Q40Block *__restrict__ w,
                                   const Q81Block *__restrict__ x, int iqs,
                                   int8_t *iq4nl_values) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = STRATA_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        sumi = STRATA_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const sycl::float2 ds =
        (x->ds).template convert<float, sycl::rounding_mode::automatic>();
    const float d = w->d;
    return d * (sumi * ds.x() - 4 * ds.y());
}

__dpct_inline__ float small_q8_dot(const Q50Block *__restrict__ w,
                                   const Q81Block *__restrict__ x, int iqs,
                                   int8_t *iq4nl_values) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = STRATA_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = STRATA_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const sycl::float2 ds =
        (x->ds).template convert<float, sycl::rounding_mode::automatic>();
    const float d = w->d;
    return d * (sumi * ds.x() - 8 * ds.y());
}

__dpct_inline__ float small_q8_dot(const Q80Block *__restrict__ w,
                                   const Q81Block *__restrict__ x, int iqs,
                                   int8_t *iq4nl_values) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = STRATA_DP4A(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = x->ds[0];
    return d0 * d1 * float(sumi);
}

__dpct_inline__ float small_q8_dot(const IQ4NLBlock *__restrict__ w,
                                   const Q81Block *__restrict__ x, int iqs,
                                   int8_t *iq4nl_values) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const sycl::int2 v =
            iq4_table_lookup(load_int_b2(w->qs, iqs + i), iq4nl_values);
        sumi = STRATA_DP4A(v.x(), q8[i], sumi);
        sumi = STRATA_DP4A(v.y(), q8[i + 4], sumi);
    }
    const float d = sycl::vec<sycl::half, 1>(w->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    x->ds[0];
    return d * sumi;
}

// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0. With VDR=2 this preserves
// the pinned 64/32-block iteration and 2048/1024-element small-K thresholds.
template <typename Weight, int Qi, bool SmallK>
__dpct_inline__ void native_small_mmvq_kernel(const Weight *__restrict__ w,
                                              const Q81Block *__restrict__ x,
                                              float *__restrict__ y, int n_in,
                                              int n_out, int8_t *iq4nl_values) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int row0 = ROWS * int(item_ct1.get_group(2));
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs, iq4nl_values);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS - 1][ROWS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            partial[item_ct1.get_local_id(1) - 1][i][item_ct1.get_local_id(2)] =
                tmp[i];
    }
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] +=
            partial[l][i][item_ct1.get_local_id(2)];
        tmp[i] = warp_sum(tmp[i]);
        if (item_ct1.get_local_id(2) == i && row0 + i < n_out) y[row0 + i] =
            tmp[i];
    }
}

// ============================ plan v0.3 P3: ncols = 2..8 (speculative verify, small batches) ============================
//
// One generic kernel for every format, parameterized by the format's iteration traits below, which are
// transcribed from the ncols = 1 kernels above (same thread-to-block mapping, same blocks per iteration, same
// small-K rule). Column j reads activation blocks x + j * (n_in / 32) and writes y + j * n_out. Each (column,
// row) value is accumulated over kbx in the same order, summed across warps in the same order and reduced with
// the same warp tree as the ncols = 1 kernel, so every column is BITWISE equal to a single-column call on that
// column (checked by bench/micro/native_mmvq_multi.cpp). The ncols = 1 kernels are untouched.
constexpr int MAX_NCOLS = 8;

// Each format splits its dot product into `load` (everything that depends only on the weight block: codes,
// unpacked scales, block scale) and `apply` (the activation loads and the original *_impl expression). `load` runs
// once per (row, block) and `apply` once per column, so adding columns adds only activation work. `apply` calls
// the same impl functions, in the same order, with the same values as the ncols = 1 dot, which is what keeps
// every column bitwise equal to it.
struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W {
        int vl[2], vh[2]; uint16_t aux[2]; sycl::half2 dm; int bq8_offset;
    };
    static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; sycl::half2 dm; int bq8_offset; };
    static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = bq8i->ds[0];
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4], qy[4]; float d2; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.d2 = w->d;
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe =
                dpct::byte_level_permute(0x020100ff, 0x020100ff, q >> 0);
            const int qo =
                dpct::byte_level_permute(0x020100ff, 0x020100ff, q >> 2);
            r.qx[j] = dpct::byte_level_permute(qe, qo, 0x5140);
            r.qy[j] = dpct::byte_level_permute(qe, qo, 0x7362);
        }
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const Q81Block* chunk = x + iqs;
        const int* q8 = reinterpret_cast<const int*>(chunk->qs);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = STRATA_DP4A(q8[j * 2], r.qx[j], sumi);
            sumi = STRATA_DP4A(q8[j * 2 + 1], r.qy[j], sumi);
        }
        const float d8 = chunk->ds[0];
        return r.d2 * d8 * sumi;
    }
};
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    static int kqs(int tid) { return tid % 16; }
    struct W { int vl, vh; float d; const uint8_t* scales; int scale_offset, bq8_offset; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d;
        r.vl = load_int_b2(w->qs, iqs);
        r.vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
        r.scales = w->scales;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + i].qs)[iqs % 8];
            d8[i] = x[r.bq8_offset + i].ds[0];
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    static int kqs(int tid) { return tid % 32; }
    // SYCL port: the weight unpack (shifts, masks, the saturating byte subtract) is column-independent, so it is
    // done once in load(); apply() is two dp4a and two fmas per column.
    struct W { int vi[2]; float dsc[2]; int bq8_offset; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        const int vl = load_int_b2(w->ql, iqs);
        const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        const float d = w->d;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
            r.vi[i] = dpct::vectorized_binary<sycl::char4>(vil | vih, 0x20202020, dpct::sub_sat());
            r.dsc[i] = d * (float) w->scales[scale_offset + 4 * i];
        }
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            sumf += x[r.bq8_offset + 2 * i].ds[0] * r.dsc[i] * (float) strata::dp4a(r.vi[i], u, 0);
        }
        return sumf;
    }
    // SYCL port, row-blocked kernel: the activation's part for this (block, lane), loaded once per column and
    // reused across the warp's rows.
    struct A { int u[2]; float ds[2]; };
    static A acts(const Q81Block* __restrict__ x, int iqs) {
        A a;
        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            a.u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
            a.ds[i] = x[bq8_offset + 2 * i].ds[0];
        }
        return a;
    }
    static float dot(const W& r, const A& a) {
        return a.ds[0] * r.dsc[0] * (float) strata::dp4a(r.vi[0], a.u[0], 0) +
               a.ds[1] * r.dsc[1] * (float) strata::dp4a(r.vi[1], a.u[1], 0);
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { sycl::int2 v[4]; int ls; float dw; };
    static W load(const Block* __restrict__ w, int iqs) { return load(w, iqs, const_cast<int8_t*>(kIq4nlTable)); }
    static W load(const Block* __restrict__ w, int iqs, int8_t *iq4nl_values) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(
            reinterpret_cast<const int *>(w->qs)[iqs + j], iq4nl_values);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = sycl::vec<sycl::half, 1>(w->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
            sumi = STRATA_DP4A(r.v[j].x(), u0, sumi);
            sumi = STRATA_DP4A(r.v[j].y(), u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * x[iqs / 4].ds[0];
        return d * sumi;
    }
};
// The four 32-element formats: `load` decodes the two 32-bit weight chunks and block scale once per (row, block),
// and `apply` runs the exact same STRATA_DP4A and scale arithmetic per activation column as `small_q8_dot`.
template<typename Weight, int Qi>
struct SmallTraits;

template<>
struct SmallTraits<Q40Block, 4> {
    using Block = Q40Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int vi0[2], vi1[2]; float d; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = load_int_b2(w->qs, iqs + i);
            r.vi0[i] = (v >> 0) & 0x0f0f0f0f;
            r.vi1[i] = (v >> 4) & 0x0f0f0f0f;
        }
        r.d = w->d;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.vi0[i], reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
            sumi = STRATA_DP4A(r.vi1[i], reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
        }
        const sycl::float2 ds =
            (x->ds).template convert<float, sycl::rounding_mode::automatic>();
        return r.d * (sumi * ds.x() - 4 * ds.y());
    }
};

template<>
struct SmallTraits<Q50Block, 4> {
    using Block = Q50Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int vi0[2], vi1[2]; float d; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
        const int qh = load_int_b2(w->qh, 0);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vl = load_int_b2(w->qs, iqs + i);
            const int vh = qh >> (4 * (iqs + i));
            int vi0 = (vl >> 0) & 0x0f0f0f0f;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            r.vi0[i] = vi0;
            int vi1 = (vl >> 4) & 0x0f0f0f0f;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            r.vi1[i] = vi1;
        }
        r.d = w->d;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.vi0[i], reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
            sumi = STRATA_DP4A(r.vi1[i], reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
        }
        const sycl::float2 ds =
            (x->ds).template convert<float, sycl::rounding_mode::automatic>();
        return r.d * (sumi * ds.x() - 8 * ds.y());
    }
};

template<>
struct SmallTraits<Q80Block, 8> {
    using Block = Q80Block;
    static constexpr int DIV = 32, T = 4, KBY = 1, BPI = 2 * WARPS * WARP / 8;
    static int kqs(int tid) { return 2 * (tid % 4); }
    struct W { int v[2]; float d0; };
    static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = load_int_b2(w->qs, iqs + i);
        r.d0 = w->d;
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
            sumi = STRATA_DP4A(r.v[i], u, sumi);
        }
        const float d1 = x->ds[0];
        return r.d0 * d1 * float(sumi);
    }
};

template<>
struct SmallTraits<IQ4NLBlock, 4> {
    using Block = IQ4NLBlock;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { sycl::int2 v[2]; float dw; };
    static W load(const Block* __restrict__ w, int iqs) { return load(w, iqs, const_cast<int8_t*>(kIq4nlTable)); }
    static W load(const Block* __restrict__ w, int iqs, int8_t *iq4nl_values) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] =
            iq4_table_lookup(load_int_b2(w->qs, iqs + i), iq4nl_values);
        r.dw = sycl::vec<sycl::half, 1>(w->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.v[i].x(), q8[i], sumi);
            sumi = STRATA_DP4A(r.v[i].y(), q8[i + 4], sumi);
        }
        const float d = r.dw * x->ds[0];
        return d * sumi;
    }
};


// S26 STRATA_LFUSE (PAIR): blocks n_out.. compute the same rows of w2 into y2 (two matrices of one shape on one
// input in one launch; each output's code is the single matrix's)
template <typename F, int NCOLS, int NW, int ROWS, bool TS = false,
          bool PAIR = false, int SG = WARP>
__dpct_inline__ void
native_mmvq_multi_kernel(const typename F::Block *__restrict__ w,
                         const Q81Block *__restrict__ x, float *__restrict__ y,
                         int n_in, int n_out,
                         const typename F::Block *__restrict__ w2 = nullptr,
                         float *__restrict__ y2 = nullptr) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int BPI =
        F::BPI * NW * SG / (WARPS * WARP); // blocks per iteration scale with the thread count
    const int tid =
        SG * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    int bxi = int(item_ct1.get_group(2));
    if constexpr (PAIR) {
        if (bxi >= n_out / ROWS) { bxi -= n_out / ROWS; w = w2; y = y2; }
    }
    const int row0 = ROWS * bxi;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
    float tmp[NCOLS][ROWS] = {};
#pragma unroll STRATA_MMVQ_UNROLL   // SYCL port: several 256-blocks' loads in flight per warp (bytes in flight, not ALU)
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                const typename F::W wv =
                    F::load(w + block, kqs); // once per (row, block)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)                         // then per column
                    tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][SG]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                partial[item_ct1.get_local_id(1) - 1][j][i]
                       [item_ct1.get_local_id(2)] = tmp[j][i];
    }
    item_ct1.barrier();
    if (item_ct1.get_local_id(1) > 0) return;
    if constexpr (TS) {
        constexpr int V = NCOLS * ROWS, P = s26ts::pow2_ceil(V);
        float v[P];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
#pragma unroll
                for (int l = 0; l < NW - 1; ++l) tmp[j][i] +=
                    partial[l][j][i][item_ct1.get_local_id(2)];
                v[j * ROWS + i] = tmp[j][i];
            }
#pragma unroll
        for (int k = V; k < P; ++k) v[k] = 0.0f;
        const float sum = s26ts::tsum<P>(v, int(item_ct1.get_local_id(2)));
        const int k = s26ts::tsum_token<P>(int(item_ct1.get_local_id(2)));
        const int j = k / ROWS, i = k % ROWS;
        if (int(item_ct1.get_local_id(2)) == s26ts::tsum_lane<P>(k) && k < V &&
            row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = sum;
        return;
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] +=
                partial[l][j][i][item_ct1.get_local_id(2)];
            tmp[j][i] = sg_sum<SG>(tmp[j][i]);
            if (item_ct1.get_local_id(2) == i && row0 + i < n_out)
                y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

// ---------------------------------------------------------------- SYCL port: row-blocked mmvq
//
// Measured on the B70 (mmvq_bench): the shared kernel above spends its time re-reading the activation - every
// row's work-group loads all NCOLS columns of it again, and a 4-warp work-group per row ends in a barrier and an
// SLM reduction for ten blocks of work. Here each warp owns RPW rows outright: per 256-block it loads the
// activation parts once for all columns and the RPW weight parts once, does RPW x NCOLS dots, and reduces its
// own lanes at the end. No barrier, no SLM. Only for traits with acts()/dot() (Q6_K so far).
template <typename F, int NCOLS, int RPW>
void native_mmvq_rowwarp_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row0 = (int(item.get_group(2)) * WARPS + warp) * RPW;
    const int blocks_per_row = n_in / F::DIV, x_stride = n_in / Q8K;
    const int kqs = F::kqs(lane);
    float tmp[RPW][NCOLS] = {};
    for (int kbx = lane / F::T; kbx < blocks_per_row; kbx += WARP / F::T) {
        const int kby = kbx * F::KBY;
        typename F::A a[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) a[j] = F::acts(x + std::size_t(j) * x_stride + kby, kqs);
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            if (row0 + r >= n_out) break;
            const typename F::W wv = F::load(w + std::size_t(row0 + r) * blocks_per_row + kbx, kqs);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) tmp[r][j] += F::dot(wv, a[j]);
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = tmp[r][j];
#pragma unroll
            for (int o = WARP / 2; o > 0; o >>= 1) v += dpct::experimental::permute_sub_group_by_xor(0xffffffffu, sg, v, o);
            if (lane == 0 && row0 + r < n_out) y[std::size_t(j) * n_out + row0 + r] = v;
        }
}

// ---------------------------------------------------------------- SYCL port: Q6_K with 16-byte loads
//
// bw probe on the B70: 4-byte loads per lane stream at 395 GB/s, 16-byte at 596; the shared kernel above does
// 4-byte loads and measures 130-140 GB/s on the 8192-wide Q6_K projections. Here 8 lanes share a 256-block:
// lane g takes positions iqs = 4g..4g+3, so its ql (16 B), qh (16 B) and each column's q8 ints (16 B) are one
// vector load each; a 32-lane sub-group has four blocks in flight. One row per warp, WARPS rows per group.
// SYCL port: a 16-byte load from a 2-byte aligned address as two aligned 16-byte loads and a shift. A Q6_K block
// is 210 bytes, so its ql/qh runs start 2-byte aligned, and the B70 splits a misaligned vector load into pieces:
// the wide kernel streamed 145 GB/s that way against 640 GB/s for the same blocks repacked at a 16-byte stride.
// Both aligned loads stay inside the 210-byte block for the ql and qh runs (last byte read <= block + 207).
__dpct_inline__ sycl::int4 load16_a2(const void* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const sycl::int4* q = reinterpret_cast<const sycl::int4*>(a & ~uintptr_t(15));
    const int k = int(a >> 2) & 3;          // dword offset within the first aligned chunk
    const bool half = (a & 2) != 0;          // and a 2-byte shift (the address is always 2-byte aligned)
    // the second chunk only when the address is unaligned: then it holds needed bytes, so (16-byte chunks never
    // straddle a page) it cannot fault even at the very end of an allocation
    const sycl::int4 lo = q[0], hi = (a & 15) ? q[1] : lo;
    const uint32_t d[8] = {(uint32_t) lo.x(), (uint32_t) lo.y(), (uint32_t) lo.z(), (uint32_t) lo.w(),
                           (uint32_t) hi.x(), (uint32_t) hi.y(), (uint32_t) hi.z(), (uint32_t) hi.w()};
    uint32_t w[5];
#pragma unroll
    for (int i = 0; i < 5; ++i)              // w = d[k .. k+4] by selects (no indexed register access)
        w[i] = k == 0 ? d[i] : k == 1 ? d[i + 1] : k == 2 ? d[i + 2] : d[(i + 3) & 7];
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = half ? (w[i] >> 16) | (w[i + 1] << 16) : w[i];
    return sycl::int4((int) o[0], (int) o[1], (int) o[2], (int) o[3]);
}

template <int NCOLS, int RPW, int SG = WARP, int BS = (int) sizeof(Q6KBlock), bool A2 = false>   // BS: block stride
void native_mmvq_q6k_wide_kernel(const Q6KBlock* __restrict__ w, const Q81Block* __restrict__ x,
                                 float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    constexpr int NWG = WARPS * WARP / SG;              // sub-groups per work-group
    const int row0 = (int(item.get_group(2)) * NWG + warp) * RPW;
    if (row0 >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;            // position group within the block, block within the iteration
    const int vh_shift = 2 * ((g & 3) >> 1);
    const int qh4_idx = 2 * (g >> 2) + (g & 1);
    const int scale_offset = 8 * (g >> 2) + (g & 3);
    const int bq8_offset = 4 * (g >> 2) + ((g & 3) >> 1);
    const int u_int4 = g & 1;                           // ints 4(g%2)..+3 of the q8 block = one int4
    float acc[RPW][NCOLS] = {};
    for (int kbx = sub; kbx < blocks_per_row; kbx += SG / 8) {
        const int kby = kbx * (QK / Q8K);
        sycl::int4 ql4[RPW], qh4[RPW]; float dsc0[RPW], dsc1[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const Q6KBlock* b = reinterpret_cast<const Q6KBlock*>(reinterpret_cast<const uint8_t*>(w) +
                (std::size_t(row0 + (row0 + r < n_out ? r : 0)) * blocks_per_row + kbx) * BS);
            if constexpr (A2) {
                ql4[r] = load16_a2(b->ql + 16 * g);
                qh4[r] = load16_a2(b->qh + 16 * qh4_idx);
            } else {
                ql4[r] = reinterpret_cast<const sycl::int4*>(b->ql)[g];
                qh4[r] = reinterpret_cast<const sycl::int4*>(b->qh)[qh4_idx];
            }
            const float d = b->d;
            dsc0[r] = d * (float) b->scales[scale_offset]; dsc1[r] = d * (float) b->scales[scale_offset + 4];
        }
        // the activations once per column, shared by the RPW rows
        sycl::int4 u0[NCOLS], u1[NCOLS];
        float ds0[NCOLS], ds1[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * x_stride + kby + bq8_offset;
            if constexpr (A2) {   // Q8_1 qs sits at offset 4 of a 36-byte block: 4-byte aligned, so four int loads
                const int* p0 = reinterpret_cast<const int*>(xb[0].qs) + 4 * u_int4;
                const int* p1 = reinterpret_cast<const int*>(xb[2].qs) + 4 * u_int4;
                u0[j] = sycl::int4(p0[0], p0[1], p0[2], p0[3]);
                u1[j] = sycl::int4(p1[0], p1[1], p1[2], p1[3]);
            } else {
                u0[j] = reinterpret_cast<const sycl::int4*>(xb[0].qs)[u_int4];
                u1[j] = reinterpret_cast<const sycl::int4*>(xb[2].qs)[u_int4];
            }
            ds0[j] = xb[0].ds[0]; ds1[j] = xb[2].ds[0];
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int vl[4] = {ql4[r].x(), ql4[r].y(), ql4[r].z(), ql4[r].w()};
            const int vh[4] = {qh4[r].x() >> vh_shift, qh4[r].y() >> vh_shift, qh4[r].z() >> vh_shift, qh4[r].w() >> vh_shift};
            int vi0[4], vi1[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                vi0[p] = dpct::vectorized_binary<sycl::char4>((vl[p] & 0x0f0f0f0f) | ((vh[p] << 4) & 0x30303030), 0x20202020, dpct::sub_sat());
                vi1[p] = dpct::vectorized_binary<sycl::char4>(((vl[p] >> 4) & 0x0f0f0f0f) | (((vh[p] >> 4) << 4) & 0x30303030), 0x20202020, dpct::sub_sat());
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                int s0 = 0, s1 = 0;
                s0 = strata::dp4a(vi0[0], u0[j].x(), s0); s0 = strata::dp4a(vi0[1], u0[j].y(), s0);
                s0 = strata::dp4a(vi0[2], u0[j].z(), s0); s0 = strata::dp4a(vi0[3], u0[j].w(), s0);
                s1 = strata::dp4a(vi1[0], u1[j].x(), s1); s1 = strata::dp4a(vi1[1], u1[j].y(), s1);
                s1 = strata::dp4a(vi1[2], u1[j].z(), s1); s1 = strata::dp4a(vi1[3], u1[j].w(), s1);
                acc[r][j] += ds0[j] * dsc0[r] * (float) s0 + ds1[j] * dsc1[r] * (float) s1;
            }
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = acc[r][j];
#pragma unroll
            for (int o = SG / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
            if (lane == 0 && row0 + r < n_out) y[std::size_t(j) * n_out + row0 + r] = v;
        }
}
// SYCL port: the wide Q6_K kernel with its weight loads issued U steps ahead. The wide kernel's 2560-wide rows
// are ~3 steps of load -> wait -> dot per sub-group, so each sub-group has one 16-byte load per lane in flight
// at a time and the kernel is latency-bound (~146 GB/s). Here a sub-group first issues the ql/qh/scale loads
// of U steps (4U blocks), then does their dots; the activation reads stay inline (they are L1 hits, shared by
// every row). One row per sub-group, WARPS rows per group.
template <int NCOLS, int U>
void native_mmvq_q6k_pf_kernel(const Q6KBlock* __restrict__ w, const Q81Block* __restrict__ x,
                               float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;
    const int vh_shift = 2 * ((g & 3) >> 1);
    const int qh4_idx = 2 * (g >> 2) + (g & 1);
    const int scale_offset = 8 * (g >> 2) + (g & 3);
    const int bq8_offset = 4 * (g >> 2) + ((g & 3) >> 1);
    const int u_int4 = g & 1;
    const Q6KBlock* wr = w + std::size_t(row) * blocks_per_row;
    float acc[NCOLS] = {};
    for (int kb0 = 0; kb0 < blocks_per_row; kb0 += 4 * U) {
        sycl::int4 ql4[U], qh4[U]; float dsc0[U], dsc1[U]; bool ok[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {      // all U steps' weight loads first
            const int kbx = kb0 + 4 * u + sub;
            ok[u] = kbx < blocks_per_row;
            const Q6KBlock* b = wr + (ok[u] ? kbx : 0);
            ql4[u] = reinterpret_cast<const sycl::int4*>(b->ql)[g];
            qh4[u] = reinterpret_cast<const sycl::int4*>(b->qh)[qh4_idx];
            const float d = b->d;
            dsc0[u] = d * (float) b->scales[scale_offset]; dsc1[u] = d * (float) b->scales[scale_offset + 4];
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (!ok[u]) continue;
            const int kby = (kb0 + 4 * u + sub) * (QK / Q8K);
            const int vl[4] = {ql4[u].x(), ql4[u].y(), ql4[u].z(), ql4[u].w()};
            const int vh[4] = {qh4[u].x() >> vh_shift, qh4[u].y() >> vh_shift, qh4[u].z() >> vh_shift, qh4[u].w() >> vh_shift};
            int vi0[4], vi1[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                vi0[p] = dpct::vectorized_binary<sycl::char4>((vl[p] & 0x0f0f0f0f) | ((vh[p] << 4) & 0x30303030), 0x20202020, dpct::sub_sat());
                vi1[p] = dpct::vectorized_binary<sycl::char4>(((vl[p] >> 4) & 0x0f0f0f0f) | (((vh[p] >> 4) << 4) & 0x30303030), 0x20202020, dpct::sub_sat());
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * x_stride + kby + bq8_offset;
                const sycl::int4 a0 = reinterpret_cast<const sycl::int4*>(xb[0].qs)[u_int4];
                const sycl::int4 a1 = reinterpret_cast<const sycl::int4*>(xb[2].qs)[u_int4];
                int s0 = 0, s1 = 0;
                s0 = strata::dp4a(vi0[0], a0.x(), s0); s0 = strata::dp4a(vi0[1], a0.y(), s0);
                s0 = strata::dp4a(vi0[2], a0.z(), s0); s0 = strata::dp4a(vi0[3], a0.w(), s0);
                s1 = strata::dp4a(vi1[0], a1.x(), s1); s1 = strata::dp4a(vi1[1], a1.y(), s1);
                s1 = strata::dp4a(vi1[2], a1.z(), s1); s1 = strata::dp4a(vi1[3], a1.w(), s1);
                acc[j] += (float) xb[0].ds[0] * dsc0[u] * (float) s0 + (float) xb[2].ds[0] * dsc1[u] * (float) s1;
            }
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}
// ---------------------------------------------------------------- SYCL port: wide kernels for Q4_K, Q5_K, IQ4_XS
//
// The Q6_K wide kernel's shape for the other 256-element formats: 8 lanes share a block, each lane's weights are one
// or two 16-byte loads, a 32-lane sub-group has four blocks in flight, one row per sub-group. The multi kernel these
// replace does 4-byte loads (Q4_K/Q5_K, 230-280 GB/s) or global-memory codebook lookups (IQ4_XS, ~100 GB/s).
// Lane g of a block: Q4_K/Q5_K take qs bytes [16g, 16g+16) = sub-blocks 2(g/2) (low nibbles) and 2(g/2)+1 (high
// nibbles) at positions 16(g%2)..+15; IQ4_XS takes sub-block g (16 bytes, low nibbles = positions 0-15).
__dpct_inline__ sycl::int4 ld_q8_16(const Q81Block* b, int half) {   // Q8_1 qs is 4-byte aligned: four int loads
    const int* q = reinterpret_cast<const int*>(b->qs) + 4 * half;
    return sycl::int4(q[0], q[1], q[2], q[3]);
}
__dpct_inline__ int dp4a4(const sycl::int4 a, const sycl::int4 b, int acc) {
    acc = strata::dp4a(a.x(), b.x(), acc); acc = strata::dp4a(a.y(), b.y(), acc);
    acc = strata::dp4a(a.z(), b.z(), acc); return strata::dp4a(a.w(), b.w(), acc);
}
__dpct_inline__ void k4_scale_min(const uint8_t* sc, int j, float& s, float& m) {   // ggml get_scale_min_k4
    if (j < 4) { s = (float) (sc[j] & 63); m = (float) (sc[j + 4] & 63); }
    else { s = (float) ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)); m = (float) ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)); }
}
struct WideQ4K {
    using Block = Q4KBlock;
    struct W { sycl::int4 lo, hi; float dl, dh, ml, mh; int sb, half; };
    static W load(const Block* b, int g) {
        W r; r.sb = 2 * (g >> 1); r.half = g & 1;
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];          // 16-byte aligned (144 = 9 x 16)
        r.lo = sycl::int4(q.x() & 0x0f0f0f0f, q.y() & 0x0f0f0f0f, q.z() & 0x0f0f0f0f, q.w() & 0x0f0f0f0f);
        r.hi = sycl::int4((q.x() >> 4) & 0x0f0f0f0f, (q.y() >> 4) & 0x0f0f0f0f, (q.z() >> 4) & 0x0f0f0f0f, (q.w() >> 4) & 0x0f0f0f0f);
        const sycl::float2 dm = b->dm.convert<float, sycl::rounding_mode::automatic>();
        float s0, m0, s1, m1; k4_scale_min(b->scales, r.sb, s0, m0); k4_scale_min(b->scales, r.sb + 1, s1, m1);
        r.dl = dm.x() * s0; r.ml = dm.y() * m0; r.dh = dm.x() * s1; r.mh = dm.y() * m1;
        return r;
    }
    static float apply(const W& r, const Q81Block* xb) {   // xb = this column's Q8_1 blocks of the 256-block
        const Q81Block* a = xb + r.sb; const Q81Block* c = xb + r.sb + 1;
        const sycl::int4 ua = ld_q8_16(a, r.half), uc = ld_q8_16(c, r.half);
        const sycl::int4 ones(0x01010101, 0x01010101, 0x01010101, 0x01010101);
        const float da = (float) a->ds[0], dc = (float) c->ds[0];
        return da * (r.dl * (float) dp4a4(r.lo, ua, 0) - r.ml * (float) dp4a4(ones, ua, 0)) +
               dc * (r.dh * (float) dp4a4(r.hi, uc, 0) - r.mh * (float) dp4a4(ones, uc, 0));
    }
};
struct WideQ5K {
    using Block = Q5KBlock;
    using W = WideQ4K::W;
    static W load(const Block* b, int g) {
        W r; r.sb = 2 * (g >> 1); r.half = g & 1;
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];          // qs at 48, 176 = 11 x 16
        const sycl::int4 h = reinterpret_cast<const sycl::int4*>(b->qh)[r.half];     // qh bytes of positions 16h..+15
        const int sl = r.sb, sh = r.sb + 1;
        auto hb = [](int v, int bit) { return ((v >> bit) & 0x01010101) << 4; };
        r.lo = sycl::int4((q.x() & 0x0f0f0f0f) | hb(h.x(), sl), (q.y() & 0x0f0f0f0f) | hb(h.y(), sl),
                          (q.z() & 0x0f0f0f0f) | hb(h.z(), sl), (q.w() & 0x0f0f0f0f) | hb(h.w(), sl));
        r.hi = sycl::int4(((q.x() >> 4) & 0x0f0f0f0f) | hb(h.x(), sh), ((q.y() >> 4) & 0x0f0f0f0f) | hb(h.y(), sh),
                          ((q.z() >> 4) & 0x0f0f0f0f) | hb(h.z(), sh), ((q.w() >> 4) & 0x0f0f0f0f) | hb(h.w(), sh));
        const sycl::float2 dm = b->dm.convert<float, sycl::rounding_mode::automatic>();
        float s0, m0, s1, m1; k4_scale_min(b->scales, r.sb, s0, m0); k4_scale_min(b->scales, r.sb + 1, s1, m1);
        r.dl = dm.x() * s0; r.ml = dm.y() * m0; r.dh = dm.x() * s1; r.mh = dm.y() * m1;
        return r;
    }
    static float apply(const W& r, const Q81Block* xb) { return WideQ4K::apply(r, xb); }
};
__dpct_inline__ uint32_t iq4_lut4(uint32_t q4) {   // four nibbles -> four codebook bytes, codebook in registers
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t n = (q4 >> (8 * i)) & 0xF;
        const uint32_t t = n < 4 ? t0 : n < 8 ? t1 : n < 12 ? t2 : t3;
        r |= ((t >> (8 * (n & 3))) & 0xFF) << (8 * i);
    }
    return r;
}
struct WideIQ4XS {
    using Block = IQ4XSBlock;
    struct W { sycl::int4 lo, hi; float d; int sb; };
    static W load(const Block* b, int g) {
        W r; r.sb = g;
        const sycl::int2* q2 = reinterpret_cast<const sycl::int2*>(b->qs + 16 * g);   // 8-byte aligned (136 = 17 x 8)
        const sycl::int2 qa = q2[0], qb = q2[1];
        const uint32_t v[4] = {(uint32_t) qa.x(), (uint32_t) qa.y(), (uint32_t) qb.x(), (uint32_t) qb.y()};
        int lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { lo[i] = (int) iq4_lut4(v[i] & 0x0f0f0f0f); hi[i] = (int) iq4_lut4((v[i] >> 4) & 0x0f0f0f0f); }
        r.lo = sycl::int4(lo[0], lo[1], lo[2], lo[3]); r.hi = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
        const int ls = ((b->scales_l[g / 2] >> (4 * (g & 1))) & 0x0f) | (((b->scales_h >> (2 * g)) & 0x03) << 4);
        r.d = (float) b->d * (float) (ls - 32);
        return r;
    }
    static float apply(const W& r, const Q81Block* xb) {
        const Q81Block* a = xb + r.sb;
        return r.d * (float) a->ds[0] * (float) dp4a4(r.hi, ld_q8_16(a, 1), dp4a4(r.lo, ld_q8_16(a, 0), 0));
    }
};
template <typename F, int NCOLS>
void native_mmvq_wide_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                             float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;
    const typename F::Block* wr = w + std::size_t(row) * blocks_per_row;
    float acc[NCOLS] = {};
    for (int kbx = sub; kbx < blocks_per_row; kbx += 4) {
        const typename F::W wv = F::load(wr + kbx, g);
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] += F::apply(wv, x + std::size_t(j) * x_stride + kbx * (QK / Q8K));
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}
// SYCL port: wide kernels for the 32-element formats Q8_0 (34-byte blocks) and IQ4_NL (18). Both are only 2-byte
// aligned, so the weights come through load16_a2 (page-safe). Q8_0: 2 lanes per block (16 quants each), 16 blocks
// per sub-group step; IQ4_NL: 1 lane per block (its 16 bytes = 32 nibbles), 32 blocks per step. One row per
// sub-group. The multi/small kernels these replace measured 26 GB/s (IQ4_NL 640 -> 2560) to ~300 GB/s.
struct Wide32Q8 {
    using Block = Q80Block;
    static constexpr int LPB = 2;
    struct W { sycl::int4 q; float d; int half; };
    static W load(const Block* b, int l) {
        W r; r.half = l; r.q = load16_a2(b->qs + 16 * l); r.d = (float) b->d; return r;
    }
    static float apply(const W& r, const Q81Block* xb) {
        return r.d * (float) xb->ds[0] * (float) dp4a4(r.q, ld_q8_16(xb, r.half), 0);
    }
};
struct Wide32IQ4NL {
    using Block = IQ4NLBlock;
    static constexpr int LPB = 1;
    struct W { sycl::int4 lo, hi; float d; };
    static W load(const Block* b, int) {
        W r;
        const sycl::int4 q = load16_a2(b->qs);
        const uint32_t v[4] = {(uint32_t) q.x(), (uint32_t) q.y(), (uint32_t) q.z(), (uint32_t) q.w()};
        int lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { lo[i] = (int) iq4_lut4(v[i] & 0x0f0f0f0f); hi[i] = (int) iq4_lut4((v[i] >> 4) & 0x0f0f0f0f); }
        r.lo = sycl::int4(lo[0], lo[1], lo[2], lo[3]); r.hi = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
        r.d = (float) b->d;
        return r;
    }
    static float apply(const W& r, const Q81Block* xb) {
        return r.d * (float) xb->ds[0] * (float) dp4a4(r.hi, ld_q8_16(xb, 1), dp4a4(r.lo, ld_q8_16(xb, 0), 0));
    }
};
template <typename F, int NCOLS>
void native_mmvq_wide32_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                               float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    constexpr int BPS = WARP / F::LPB;                  // blocks per sub-group step
    const int blocks_per_row = n_in / 32;
    const int l = lane % F::LPB, sub = lane / F::LPB;
    const typename F::Block* wr = w + std::size_t(row) * blocks_per_row;
    float acc[NCOLS] = {};
    for (int kb = sub; kb < blocks_per_row; kb += BPS) {
        const typename F::W wv = F::load(wr + kb, l);
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] += F::apply(wv, x + std::size_t(j) * blocks_per_row + kb);
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}
#if defined(STRATA_HIP_GFX906)
// ---- AMD (wave64) layout: one wavefront per R rows, four wavefronts per block, the row's blocks strided over the
// 64 lanes exactly as the CUDA kernels stride them over a block (kbx = lane / T, stride 64 / T), and a 64-lane
// butterfly instead of the LDS partials and __syncthreads of the one-block-per-row layout: on gfx906 a block per
// 2 KB row spent most of its time being launched and joined.  The SAME kernel serves every column count 1..8, so a
// column's sums do not depend on how many columns (verify tokens) ride along.  STRATA_MMVQ_WAVE=0: the CUDA layout.
template<typename F, int NCOLS, int R>
__launch_bounds__(256)
__global__ void native_mmvq_wave_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPIW = 64 / F::T;
    static_assert(64 % F::T == 0, "a block's threads must tile the wavefront");
    const int lane = int(threadIdx.x) & 63;
    const int row0 = (int(blockIdx.x) * 4 + (int(threadIdx.x) >> 6)) * R;
    if (row0 >= n_out) return;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    const int kqs = F::kqs(lane);
    float tmp[NCOLS][R] = {};
    for (int kbx = lane / F::T; kbx < blocks_per_row; kbx += BPIW) {
        const int kby = kbx * F::KBY;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float v = tmp[j][i];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
            if (lane == 0 && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = v;
        }
}
#endif

// ============================ STRATA_Q8_PACKED=1: the packed Q8_0 decode layout (opt-in) ============================
//
// A lossless load-time repack of a Q8_0 matrix: the SAME int8 values in a qs plane (row-major, n_in bytes per row)
// followed by the SAME fp16 scales in a d plane (row-major, n_in / 32 per row) - no requantization, same bytes.
// The GGUF's 34-byte blocks leave every int 2-byte aligned (load_int_b2 = two 16-bit loads); the planes give each
// thread one aligned 8-byte run, read with streaming (non-temporal) loads. Every thread owns exactly the bytes of
// the ncols = 1 / multi-column EXACT layout (block kbx = tid / 4 + 32 i, ints 2 (tid % 4) and +1), accumulates them
// in the same order with the same expression, and the cross-warp sum and XOR tree are the same: every output is
// BITWISE equal to native_small_mmvq_kernel / native_mmvq_multi_kernel<SmallTraits<Q80Block, 8>, NC, 4, 1>.
// Eligible: 1024 <= n_in <= 4096 and n_out >= 2048, run as 640 persistent workgroups (a grid-stride over rows, two
// partial buffers). Measured on gfx1151 (S26, 8 real UD-IQ4_XS layers per shape, paired, 1-5 columns): 2560 ->
// 6144/10240/12288 +10-12% per call (engine trace: -10.7% on those calls). 6144 -> 2560 (one workgroup per row)
// was +2-4% standalone but -3% in the engine, and rows < 2048 lose: neither is packed.
#if defined(__HIPCC__)
#define STRATA_Q8P_LOAD(p) __builtin_nontemporal_load(p)
#else
#define STRATA_Q8P_LOAD(p) (*(p))
#endif

template <int NCOLS, bool TS = false>
__dpct_inline__ void
native_q8_0_packed_kernel(const int8_t *__restrict__ qs,
                          const sycl::half *__restrict__ dpl,
                          const Q81Block *__restrict__ x, float *__restrict__ y,
                          int n_in, int n_out, int row_step) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int nb = n_in / 32;
    const int kqs = 2 * (tid % 4);
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[2][WARPS - 1][NCOLS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int buf = 0;
    for (int row = int(item_ct1.get_group(2)); row < n_out;
         row += row_step, buf ^= 1) {
        float tmp[NCOLS] = {};
        for (int kbx = tid / 4; kbx < nb; kbx += 2 * WARPS * WARP / 8) {
            const std::size_t block = std::size_t(row) * nb + kbx;
            const int* p = reinterpret_cast<const int*>(qs + block * 32) + kqs;
            const int v0 = STRATA_Q8P_LOAD(p);
            const int v1 = STRATA_Q8P_LOAD(p + 1);
            const float d0 = dpl[block];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * nb + kbx;
                const int* u = reinterpret_cast<const int*>(xb->qs);
                int sumi = 0;
                sumi = STRATA_DP4A(v0, u[kqs], sumi);
                sumi = STRATA_DP4A(v1, u[kqs + 1], sumi);
                const float d1 = xb->ds[0];
                tmp[j] += d0 * d1 * float(sumi);
            }
        }
        if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j)
                partial[buf][item_ct1.get_local_id(1) - 1][j]
                       [item_ct1.get_local_id(2)] = tmp[j];
        }
        item_ct1.barrier();
        if (TS && item_ct1.get_local_id(1) ==
                      0) { // S26 STRATA_TSUM=1: one transposed butterfly,
                           // bitwise the same sums
            constexpr int P = s26ts::pow2_ceil(NCOLS);
            float v[P];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] +=
                    partial[buf][l][j][item_ct1.get_local_id(2)];
                v[j] = tmp[j];
            }
#pragma unroll
            for (int j = NCOLS; j < P; ++j) v[j] = 0.0f;
            const float sum = s26ts::tsum<P>(v, int(item_ct1.get_local_id(2)));
            const int j = s26ts::tsum_token<P>(int(item_ct1.get_local_id(2)));
            if (int(item_ct1.get_local_id(2)) == s26ts::tsum_lane<P>(j) &&
                j < NCOLS) y[std::size_t(j) * n_out + row] = sum;
        } else if (item_ct1.get_local_id(1) == 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] +=
                    partial[buf][l][j][item_ct1.get_local_id(2)];
                tmp[j] = warp_sum(tmp[j]);
                if (item_ct1.get_local_id(2) == 0)
                    y[std::size_t(j) * n_out + row] = tmp[j];
            }
        }
    }
}

// ============================ STRATA_Q6_PACKED=1: the packed Q6_K head layout (opt-in) ============================
//
// The output heads (the main head, 248320 rows, and the MTP draft head's row subset) are Q6_K matrices of 2560-wide
// rows: 10 blocks of 210 bytes each, so every int load is 2-byte aligned and a row is 2100 bytes. The packed copy
// holds the SAME bytes as four planes - ql (128 B per block), qh (64 B), scales (16 B), d (2 B) - so a thread's
// two words are aligned dword loads (non-temporal). Thread ownership (block tid / 32 + 4 i, word tid % 32), the
// q6_q8_dot_impl expression, the per-thread order and the reduction are those of native_q6_k_mmvq_kernel<false> /
// native_mmvq_multi_kernel<Q6KTraits, NC, 4, 1>: every output is bitwise equal. Persistent workgroups.
template <int NCOLS, bool TS = false>
__dpct_inline__ void native_q6_k_packed_kernel(
    const int *__restrict__ ql, const int *__restrict__ qh,
    const int8_t *__restrict__ sc, const sycl::half *__restrict__ dpl,
    const Q81Block *__restrict__ x, float *__restrict__ y, int n_in, int n_out,
    int row_step) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid =
        WARP * int(item_ct1.get_local_id(1)) + int(item_ct1.get_local_id(2));
    const int nb = n_in / 256;
    const int x_stride = n_in / Q8K;
    const int iqs = tid % 32;
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int qh_word = 8 * (iqs / 16) + iqs % 8;
    auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[2][WARPS - 1][NCOLS][WARP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int buf = 0;
    for (int row = int(item_ct1.get_group(2)); row < n_out;
         row += row_step, buf ^= 1) {
        float tmp[NCOLS] = {};
        for (int kbx = tid / 32; kbx < nb; kbx += WARPS * WARP / 32) {
            const std::size_t block = std::size_t(row) * nb + kbx;
            const int vl = STRATA_Q8P_LOAD(ql + block * 32 + iqs);
            const int vh = STRATA_Q8P_LOAD(qh + block * 16 + qh_word) >> vh_shift;
            const int8_t* scales = sc + block * 16 + scale_offset;
            const float d = dpl[block];
            const int kby = kbx * 8;
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * x_stride + kby;
                int u[2];
                float d8[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    u[i] = reinterpret_cast<const int*>(xb[bq8_offset + 2 * i].qs)[iqs % 8];
                    d8[i] = xb[bq8_offset + 2 * i].ds[0];
                }
                tmp[j] += q6_q8_dot_impl(vl, vh, u, scales, d, d8);
            }
        }
        if (item_ct1.get_local_id(1) > 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j)
                partial[buf][item_ct1.get_local_id(1) - 1][j]
                       [item_ct1.get_local_id(2)] = tmp[j];
        }
        item_ct1.barrier();
        if (TS && item_ct1.get_local_id(1) ==
                      0) { // S26 STRATA_TSUM=1: one transposed butterfly,
                           // bitwise the same sums
            constexpr int P = s26ts::pow2_ceil(NCOLS);
            float v[P];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] +=
                    partial[buf][l][j][item_ct1.get_local_id(2)];
                v[j] = tmp[j];
            }
#pragma unroll
            for (int j = NCOLS; j < P; ++j) v[j] = 0.0f;
            const float sum = s26ts::tsum<P>(v, int(item_ct1.get_local_id(2)));
            const int j = s26ts::tsum_token<P>(int(item_ct1.get_local_id(2)));
            if (int(item_ct1.get_local_id(2)) == s26ts::tsum_lane<P>(j) &&
                j < NCOLS) y[std::size_t(j) * n_out + row] = sum;
        } else if (item_ct1.get_local_id(1) == 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] +=
                    partial[buf][l][j][item_ct1.get_local_id(2)];
                tmp[j] = warp_sum(tmp[j]);
                if (item_ct1.get_local_id(2) == 0)
                    y[std::size_t(j) * n_out + row] = tmp[j];
            }
        }
    }
}

} // namespace

#include "native_mmvq.kern.inc"

} // namespace strata::kernels
