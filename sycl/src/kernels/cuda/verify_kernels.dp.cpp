// src/kernels/cuda/verify_kernels.cu - see include/strata/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
#include "strata/port.hpp"
#include "strata/sycl_doorbell.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/dp4a.hpp"
#include "verify_kernels_shared.hpp"

#ifndef STRATA_PLAN_LOCAL
#define STRATA_PLAN_LOCAL 1   // 0: the original one-thread plan kernel (A/B)
#endif
namespace strata::kernels {
namespace {

// `commit` (one token): the history then keeps it - [hist1, hist2, x_0], each thread its channel's, after reading them
__dpct_inline__ void gdn_conv_l2_multi_kernel(float *hist,
                                              const float *__restrict__ qkv,
                                              const float *__restrict__ w,
                                              float *__restrict__ h, int C,
                                              int qk_heads, float eps,
                                              int t_begin, bool commit) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = t_begin + item_ct1.get_group(1);
    const int c = item_ct1.get_group(2) * S + item_ct1.get_local_id(2);
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    if (commit) {
        hist[c * 3] = v1;
        hist[c * 3 + 1] = v2;
        hist[c * 3 + 2] = x;
    }
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + sycl::native::exp(-sum));
    if ((int)item_ct1.get_group(2) < qk_heads) {
        float sq = y * y;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                sq, o);
        if ((item_ct1.get_local_id(2) & 31) == 0)
            part[item_ct1.get_local_id(2) >> 5] = sq;
        item_ct1.barrier();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= sycl::rsqrt(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

__dpct_inline__ void
gdn_conv_commit_kernel(float *__restrict__ hist, const float *__restrict__ qkv,
                       int C, const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

template <int MAX_T = kVerifyMaxT, bool EXACT_T = false>
__dpct_inline__ void gdn_ab_multi_kernel(
    const float *__restrict__ x, const uint16_t *__restrict__ wa,
    const uint16_t *__restrict__ wb, const float *__restrict__ dt,
    const float *__restrict__ ssm_a, float *__restrict__ gate,
    float *__restrict__ beta, int n, int h_v, int T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * 2 + (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(
        (is_beta ? wb : wa) + (size_t)r * n);
    float acc[MAX_T];
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const sycl::uint4 wv = *(w4 + j);
        const float w0 = sycl::bit_cast<float>(wv.x() << 16),
                    w1 = sycl::bit_cast<float>(wv.x() & 0xffff0000u);
        const float w2 = sycl::bit_cast<float>(wv.y() << 16),
                    w3 = sycl::bit_cast<float>(wv.y() & 0xffff0000u);
        const float w4f = sycl::bit_cast<float>(wv.z() << 16),
                    w5 = sycl::bit_cast<float>(wv.z() & 0xffff0000u);
        const float w6 = sycl::bit_cast<float>(wv.w() << 16),
                    w7 = sycl::bit_cast<float>(wv.w() & 0xffff0000u);
#pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            if (!EXACT_T && t >= T) break;
            const float* xt = x + (size_t) t * n;
            const sycl::float4 xa = *reinterpret_cast<const sycl::float4 *>(xt + j * 8);
            const sycl::float4 xb = *reinterpret_cast<const sycl::float4 *>(xt + j * 8 + 4);
            float a = acc[t];
            a = sycl::fma((float)w0, (float)(xa.x()), a);
                a = sycl::fma((float)w1, (float)(xa.y()), a);
            a = sycl::fma((float)w2, (float)(xa.z()), a);
                a = sycl::fma((float)w3, (float)(xa.w()), a);
            a = sycl::fma((float)w4f, (float)(xb.x()), a);
                a = sycl::fma((float)w5, (float)(xb.y()), a);
            a = sycl::fma((float)w6, (float)(xb.z()), a);
                a = sycl::fma((float)w7, (float)(xb.w()), a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) {
        if (!EXACT_T && t >= T) break;
        float a = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t)t * h_v + r] = 1.0f / (1.0f + sycl::native::exp(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : sycl::log1p(sycl::native::exp(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

// State-only commit kernel: each head's 128 independent state columns are split across 4 blocks of 32 columns
// (48 * 4 = 192 blocks of 128 threads across all SMs, vs 48 blocks of 512 threads), with no sq/o/norm/z/y work.
__dpct_inline__ void gdn_step_commit_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta, int h_k,
    int h_v, const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int n = *n_keep;
    if (n <= 0) return;
    auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int col_local = item_ct1.get_local_id(2);
    const int col = item_ct1.get_group(1) * 32 + col_local;
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * 32 + col_local;    // 0 .. 127
    const int qh = head % h_k;
    const int qk = S * h_k;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        item_ct1.barrier();
        sk[tid] = ht[qk + qh * S + tid];
        item_ct1.barrier();
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][col_local] = kv;
        item_ct1.barrier();
        const float kv_col = red[0][col_local] + red[1][col_local] + red[2][col_local] + red[3][col_local];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

// S26 (STRATA_QFUSE=1): the q8_1 image of a 32-column block, written by the warp that holds those columns with
// native_quantize_q8_1_kernel's quantizer (the same XOR-tree max and sum, d = amax / 127, roundf(x / d), ds = (d, sum)).
struct GdnQ81 { sycl::half2 ds; int8_t qs[32]; };
__dpct_inline__ void gdn_q8_1_store(GdnQ81 *__restrict__ xq, size_t idx,
                                    float xi) {
    // no contraction: xi is a product here, and fma(a, b, shfl) would round the first sum unlike the separate
    // quantizer, which reads xi from memory (S26 harness: ds.sum differed by 1 ulp without this)
#pragma clang fp contract(off)
    float amax = sycl::fabs(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = sycl::fmax(
        amax, dpct::experimental::permute_sub_group_by_xor(
                  0xffffffffu,
                  sycl::ext::oneapi::this_work_item::get_sub_group(), amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            sum, o);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : sycl::round(xi / d);
    xq[idx / 32].qs[idx % 32] = q;
    if (idx % 32 == 0) xq[idx / 32].ds = sycl::half2(d, sum);
}

template <bool ALL_OUT, bool Q>
__dpct_inline__ void gdn_step_norm_multi_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta,
    const float *__restrict__ z, const float *__restrict__ gamma, float eps,
    float *__restrict__ y, int h_k, int h_v, int T,
    const int32_t *__restrict__ n_keep, int t_out_begin,
    GdnQ81 *__restrict__ xq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][S]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red_kv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red_o =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &wsum =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S / 32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int col = item_ct1.get_local_id(2);
    const int rg = item_ct1.get_local_id(1);
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = ALL_OUT ? T : (n_keep ? *n_keep : T);
    if (n > 0) {
        if (rg == 0) sk[0][col] = hbuf[qk + qh * S + col];
        else if (rg == 1 && (ALL_OUT || 0 >= t_out_begin)) sq[0][col] = hbuf[qh * S + col];
    }
    const float gam = (rg == 0) ? gamma[col] : 0.0f;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = ALL_OUT ? base[r * row_stride]
                                                 : base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const int cur = t & 1, nxt = (t + 1) & 1;
        const float* ht = hbuf + (size_t) t * C;
        const bool need_out = ALL_OUT || (t >= t_out_begin);
        item_ct1.barrier();
        if (t + 1 < n) {
            const float* ht_next = ht + C;
            const bool need_next = ALL_OUT || (t + 1 >= t_out_begin);
            if (rg == 0) sk[nxt][col] = ht_next[qk + qh * S + col];
            else if (rg == 1 && need_next) sq[nxt][col] = ht_next[qh * S + col];
        }
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[cur][rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        item_ct1.barrier();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        if (!need_out) {
#pragma unroll
            for (int r = 0; r < RPG; ++r)
                s[r] = sycl::fma((float)g, s[r], sk[cur][rg * RPG + r] * delta);
            continue;
        }
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[cur][rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[cur][rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        item_ct1.barrier();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] +
                  red_o[3][col]) *
                 sycl::rsqrt((float)S);
            sq_part = oc * oc;
        }
        // every thread takes the shuffles, as the single-token kernel does: inside the `rg == 0` branch hipcc on
        // gfx1151 gave another rounding for 3% of the outputs (gdn_parity section 5 on Aurora)
#pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                sq_part, o2);
        if (rg == 0 && (col & 31) == 0) wsum[col >> 5] = sq_part;
        item_ct1.barrier();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = sycl::rsqrt(ss / (float)S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            const float yv =
                oc * scale * gam * (1.0f / (1.0f + sycl::native::exp(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;
            if constexpr (Q) gdn_q8_1_store(xq, (size_t) (t - t_out_begin) * value_dim + head * S + col, yv);
        }
    }
    if (!ALL_OUT && n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

// S25 (STRATA_GDN_SPLIT=1): the same recurrence with each head's 128 state columns over 4 blocks of 32 columns
// (128 threads: the same (column, row group) threads, each with the same 32 state rows), so 4x the blocks of
// gdn_step_norm_multi_kernel - which ran one 512-thread block per head (32 blocks) and read the 2 MB state of a layer
// at ~60 GB/s. Every column's arithmetic is the original's; the output norm, which needs all 128 columns of a head,
// moves to gdn_out_norm_kernel: this kernel leaves the unnormalized output `oc` in y and the second kernel sums the
// squares by the same warps (columns 32w..32w+31, the same butterfly) in the same order.
// S26: OUT = false (the commit, whose outputs nobody reads) drops the output reduction and its two barriers; the
// norm kernel below has the old kernel's code shape (S x RG threads, rg 0 holding oc), which makes it bitwise equal -
// the 128-thread version differed from the old kernel by 1 ulp in ~7% of the outputs.
template <bool OUT>
__dpct_inline__ void
gdn_step_split_kernel(float *__restrict__ state, const float *__restrict__ hbuf,
                      int C, const float *__restrict__ gate,
                      const float *__restrict__ beta, float *__restrict__ y,
                      int h_k, int h_v, int T,
                      const int32_t *__restrict__ n_keep, int t_out_begin) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[RG][GS_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int c0 = item_ct1.get_group(1) * GS_COLS;
    const int lc = item_ct1.get_local_id(2); // local column
    const int col = c0 + lc;
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * GS_COLS + lc;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        item_ct1.barrier();
        if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        item_ct1.barrier();
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][lc] = kv;
        item_ct1.barrier();
        const float kv_col = red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[rg * RPG + r], o);
        }
        if (!OUT) continue;
        item_ct1.barrier();
        red[rg][lc] = o;
        item_ct1.barrier();
        if (rg == 0 && t >= t_out_begin)
            y[(size_t)t * value_dim + head * S + col] =
                (red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc]) *
                sycl::rsqrt((float)S);
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}


// S26: the state-only replay (the commit) with its per-token inputs prefetched - gate, beta and the v column of every
// row up front, the next row's k / q while the current one computes. The arithmetic and its order are
// gdn_step_split_kernel<false>'s (gfx1151 harness: state bitwise equal; 44.5 -> 39.1 us at 1 row, 47.4 -> 40.5 at 2,
// 53.8 -> 42.9 at 4, h_v 48). With outputs the prefetch lost (register pressure), so the window keeps the plain split.
template <int MAXT>
__dpct_inline__ void gdn_state_replay_kernel(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta, int h_k,
    int h_v, int T, const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[RG][GS_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int c0 = item_ct1.get_group(1) * GS_COLS;
    const int lc = item_ct1.get_local_id(2);
    const int col = c0 + lc;
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * GS_COLS + lc;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int n = n_keep ? *n_keep : T;
    float gv[MAXT], bv[MAXT], vv[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t < n) {
            gv[t] = gate[(size_t) t * h_v + head];
            bv[t] = beta[(size_t) t * h_v + head];
            vv[t] = hbuf[(size_t) t * C + 2 * qk + head * S + col];
        }
    }
    float kn = 0.0f, qn = 0.0f;
    if (n > 0) { kn = hbuf[qk + qh * S + tid]; qn = hbuf[qh * S + tid]; }
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t >= n) break;
        item_ct1.barrier();
        sk[tid] = kn; sq[tid] = qn;
        item_ct1.barrier();
        if (t + 1 < n) {
            const float* hn = hbuf + (size_t) (t + 1) * C;
            kn = hn[qk + qh * S + tid]; qn = hn[qh * S + tid];
        }
        const float g = sycl::native::exp(gv[t]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][lc] = kv;
        item_ct1.barrier();
        const float kv_col = red[0][lc] + red[1][lc] + red[2][lc] + red[3][lc];
        const float delta = (vv[t] - g * kv_col) * bv[t];
        float o = 0.0f;   // unused: kept because the measured code had it (without it: 51 us instead of 39 at 1 row)
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[rg * RPG + r], o);
        }
        (void) o;
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}
// y = oc * rsqrt(mean(oc^2) + eps) * gamma * sigmoid(z) per head and token: gdn_step_norm_multi_kernel's tail with its
// code shape (S x RG threads; rg 0 holds oc, the other row groups contribute 0 to wsum), reading oc from y
template <bool Q>
__dpct_inline__ void gdn_out_norm_kernel(
    const float *__restrict__ z, const float *__restrict__ gamma, float eps,
    float *__restrict__ y, int h_v, int T, const int32_t *__restrict__ n_keep,
    int t_out_begin, GdnQ81 *__restrict__ xq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &wsum =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S * RG / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2),
              t = item_ct1.get_group(1) + t_out_begin,
              col = item_ct1.get_local_id(2), rg = item_ct1.get_local_id(1);
    const int tid = rg * S + col;
    const int n = n_keep ? *n_keep : T;
    if (t >= n) return;
    const int value_dim = S * h_v;
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = y[(size_t) t * value_dim + head * S + col];
        sq_part = oc * oc;
    }
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            sq_part, o2);
    if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
    item_ct1.barrier();
    if (rg == 0) {
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float scale = sycl::rsqrt(ss / (float)S + eps);
        const float zz = z[(size_t) t * value_dim + head * S + col];
        if constexpr (Q) {
            const float yv = oc * scale * gamma[col] *
                             (1.0f / (1.0f + sycl::native::exp(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;
            gdn_q8_1_store(xq, (size_t) (t - t_out_begin) * value_dim + head * S + col, yv);
        } else {
            y[(size_t)t * value_dim + head * S + col] =
                oc * scale * gamma[col] *
                (1.0f / (1.0f + sycl::native::exp(-zz)));
        }
    }
}

__dpct_inline__ void embedding_gather_dev_kernel(
    const uint8_t *__restrict__ codes, const float *__restrict__ scales,
    const float *__restrict__ offsets, const int32_t *__restrict__ tokens,
    int64_t n, int code_bits, int code_bias, int group_elems,
    unsigned long long row_codes, unsigned long long row_groups,
    float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const unsigned long long token = (unsigned long long) tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = (float)(code + code_bias) * sc[group];
    out[(size_t)t * n + i] = product + (of ? of[group] : 0.0f);
}

__dpct_inline__ void broadcast_streams_kernel(const float *__restrict__ x,
                                              float *__restrict__ R, int64_t n,
                                              int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

__dpct_inline__ void copy_indexed_kernel(float *__restrict__ dst,
                                         const float *__restrict__ src,
                                         int64_t stride,
                                         const int32_t *__restrict__ index,
                                         int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int idx = *index;
    if (idx < 0) return;
#pragma unroll
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n; i += (int64_t)item_ct1.get_group_range(2) *
                     item_ct1.get_local_range(2))
        dst[i] = src[(size_t) idx * stride + i];
}

__dpct_inline__ void
fetch_blobs_kernel(const unsigned long long *__restrict__ src,
                   const int32_t *__restrict__ n, sycl::uint4 *__restrict__ dst,
                   long long per) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long total = (long long)*n * per;
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < total; i += (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2)) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const sycl::uint4 *)src[k])[off];
    }
}

__dpct_inline__ void rebase_ptrs_kernel(unsigned long long *ptr,
                                        const int32_t *n,
                                        unsigned long long base,
                                        long long bytes) {
    const int k =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

__dpct_inline__ void add_streams_broadcast_kernel(const float *__restrict__ h,
                                                  const float *__restrict__ e,
                                                  float *__restrict__ R,
                                                  int64_t n, int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

__dpct_inline__ void ident_hits_kernel(const int32_t *__restrict__ ids, int n,
                                       int32_t *__restrict__ slot,
                                       int32_t *__restrict__ dst,
                                       int32_t *__restrict__ count) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i < n) { slot[i] = ids[i]; dst[i] = i; }
    if (i == 0) *count = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template <typename E>
__dpct_inline__ void gather_rows_kernel(const E *__restrict__ src,
                                        long long row_e,
                                        const int32_t *__restrict__ ids,
                                        long long n, E *__restrict__ dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long total = n * row_e;
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < total; i += (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2)) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

__dpct_inline__ void map_ids_kernel(int32_t *ids,
                                    const int32_t *__restrict__ table, int n) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i < n) ids[i] = table[ids[i]];
}

__dpct_inline__ void row_top_prob_kernel(const float *__restrict__ logits,
                                         int n_vocab,
                                         const int32_t *__restrict__ ids,
                                         float *__restrict__ probs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_group(2);
    const float* l = logits + (size_t) t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n_vocab;
         i += item_ct1.get_local_range(2)) s += sycl::native::exp(l[i] - m);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), s,
            o);
    if ((item_ct1.get_local_id(2) & 31) == 0)
        part[item_ct1.get_local_id(2) >> 5] = s;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < (int)(item_ct1.get_local_range(2) >> 5); ++w) tot +=
            part[w];
        probs[t] = 1.0f / tot;
    }
}

__dpct_inline__ void
mtp_select_kernel(const float *__restrict__ R_src, int64_t stride,
                  const int32_t *__restrict__ ids,
                  const int32_t *__restrict__ row_dev,
                  float *__restrict__ R_dst, int32_t *__restrict__ tok_dst,
                  int32_t *out, int j, const float *probs, float *out_p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = *row_dev;
#pragma unroll
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < stride; i += (int64_t)item_ct1.get_group_range(2) *
                          item_ct1.get_local_range(2))
        R_dst[i] = R_src[(size_t) row * stride + i];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::system);
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
    }
}

__dpct_inline__ void dense_steps_kernel(const int32_t *__restrict__ cells,
                                        int n, int32_t *__restrict__ steps) {
    const int i =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

// --pipeline-windows: the drafter's chain, teacher forced (see force_token)
__dpct_inline__ void force_token_kernel(int32_t *tok, const int32_t *force,
                                        int j) {
    const int32_t f = ((const volatile int32_t*) force)[j];
    if (f >= 0) *tok = f;
}


__dpct_inline__ void
copy_row_to_first_kernel(const int32_t *__restrict__ row_dev, float *a,
                         int64_t a_n, float *b, int64_t b_n, float *c,
                         int64_t c_n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t row = *row_dev;
    if (row == 0) return;
    const int64_t i0 = (int64_t)item_ct1.get_group(2) *
                           item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2),
                  st = (int64_t)item_ct1.get_group_range(2) *
                       item_ct1.get_local_range(2);
#pragma unroll
    for (int64_t i = i0; i < a_n; i += st) a[i] = a[row * a_n + i];
#pragma unroll
    for (int64_t i = i0; i < b_n; i += st) b[i] = b[row * b_n + i];
#pragma unroll
    for (int64_t i = i0; i < c_n; i += st) c[i] = c[row * c_n + i];
}
}  // namespace

#if !defined(__HIPCC__)   // warp-32 kernels: an AMD card keeps the one-block row_top_prob and sampler_greedy_kernel
namespace {


// the larger value, on equality the lower index: sampler_greedy_kernel's order
__dpct_inline__ void arg_take(float ov, int oi, float &bv, int &bi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// a block's (value, index) pick, in thread 0; `none` is the index of no candidate
__dpct_inline__ void arg_block(float &bv, int &bi, int none) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &si = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = dpct::experimental::shift_sub_group_left(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), bv,
            off);
        const int oi = dpct::experimental::shift_sub_group_left(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), bi,
            off);
        arg_take(ov, oi, bv, bi);
    }
    const int warp = (int)(item_ct1.get_local_id(2) >> 5),
              lane = (int)(item_ct1.get_local_id(2) & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    item_ct1.barrier();
    if (warp == 0) {
        const int nw = (int)(item_ct1.get_local_range(2) >> 5);
        bv = lane < nw ? sv[lane] : sycl::bit_cast<float, int>(0xff800000);
        bi = lane < nw ? si[lane] : none;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bv, off);
            const int oi = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bi, off);
            arg_take(ov, oi, bv, bi);
        }
    }
}

// grid (blocks, rows): block b scans [b * per_block, +per_block) of its row
__dpct_inline__ void argmax_rows_kernel(const float *__restrict__ logits, int n,
                                        int per_block,
                                        ArgRow *__restrict__ rows,
                                        int32_t *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = (int)item_ct1.get_group(1), b = (int)item_ct1.get_group(2),
              nb = (int)item_ct1.get_group_range(2);
    unsigned* counter = &rows[row].counter;
    float* pv = rows[row].v;
    int* pi = rows[row].i;
    const float* l = logits + (size_t) row * n;
    const int lo = b * per_block, hi = sycl::min(n, lo + per_block);
    float bv = sycl::bit_cast<float, int>(0xff800000); // -inf
    int bi = n;
    for (int v = lo + (int)item_ct1.get_local_id(2); v < hi;
         v += (int)item_ct1.get_local_range(2)) {
        const float s = l[v];
        if (s > bv) { bv = s; bi = v; }
    }
    arg_block(bv, bi, n);
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<bool>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) {
        pv[b] = bv;
        pi[b] = bi;
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::device);
        last =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                counter, 1u) == (unsigned)(nb - 1);
    }
    item_ct1.barrier();
    if (!last) return;
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    bv = sycl::bit_cast<float, int>(0xff800000);
    bi = n;
    if ((int)item_ct1.get_local_id(2) < nb) {
        bv = ((volatile const float *)pv)[item_ct1.get_local_id(2)];
        bi = ((volatile const int *)pi)[item_ct1.get_local_id(2)];
    }
    arg_block(bv, bi, n);
    if (item_ct1.get_local_id(2) == 0) {
        out[row] = bi < n ? bi : 0;   // no value above -inf: 0, as the one-block kernel
        *counter = 0;
    }
}

// grid (kTopBlocks, rows), 1024 / kTopBlocks threads: thread j of block b is row_top_prob's thread b * blockDim + j
__dpct_inline__ void row_top_prob_split_kernel(const float *__restrict__ logits,
                                               int n_vocab,
                                               const int32_t *__restrict__ ids,
                                               float *__restrict__ probs,
                                               TopRow *__restrict__ rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = (int)item_ct1.get_group(1);
    float* parts = rows[row].parts;
    const float* l = logits + (size_t) row * n_vocab;
    const float m = l[ids[row]];
    const int vt = (int)(item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                         item_ct1.get_local_id(2));
    float s = 0.0f;
#pragma unroll
    for (int i = vt; i < n_vocab; i += 1024) s += sycl::native::exp(l[i] - m);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), s,
            o);
    if ((item_ct1.get_local_id(2) & 31) == 0) parts[vt >> 5] = s;
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    item_ct1.barrier();
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<bool>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) last =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &rows[row].counter, 1u) ==
        (unsigned)(item_ct1.get_group_range(2) - 1);
    item_ct1.barrier();
    if (!last || item_ct1.get_local_id(2) != 0) return;
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 32; ++w) tot += ((volatile const float *)parts)[w];
    probs[row] = 1.0f / tot;
    rows[row].counter = 0;
}

}  // namespace
#endif

namespace {
__dpct_inline__ void window_ids_kernel(int32_t *steps, int window, int32_t *ids,
                                       long long stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int q = item_ct1.get_group(1);
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
#pragma unroll
    for (int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
         j < width;
         j += item_ct1.get_group_range(2) * item_ct1.get_local_range(2))
        ids[q * stride + j] = start + j;
    item_ct1.barrier();
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) == 0) st[3] =
        width;
}

__dpct_inline__ void wait_flag_ge_kernel(const volatile uint32_t *flag,
                                         uint32_t value, uint32_t spin_max) {
    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

__dpct_inline__ void
resident_plan_kernel(const int32_t *__restrict__ ids, int n, int k,
                     const int32_t *__restrict__ res, int n_expert,
                     const uint8_t *cache_base,
                     const unsigned long long *slot_off, long long blob,
                     int32_t *__restrict__ pl, long long capx, uint32_t *skip,
                     uint32_t ring, const unsigned long long *__restrict__ mir,
                     volatile uint32_t *plan_err, int32_t *usage,
                     unsigned long long *stats, int32_t *lru_res,
                     int32_t *lru_owner, uint32_t *lru_last,
                     uint32_t *lru_state, unsigned long long *lru_jobs,
                     uint32_t *lru_njobs, int lru_layer, uint32_t lru_guard,
                     uint32_t lru_scan, uint32_t lru_sample) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_ids = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    int32_t[kResidentPlanMax]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_excl = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int32_t[kResidentPlanMax]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_wsum =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[4]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_bad = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int tid = item_ct1.get_local_id(2);
    if (tid == 0) s_bad = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);

    int32_t eid = -1;
    int32_t slot = -1;
    unsigned long long maddr = 0;   // SYCL port: a host-mirrored expert (not in VRAM): its pinned mirror, read over PCIe
    if (tid < n) {
        eid = ids[tid];
        s_ids[tid] = eid;
        slot = (eid >= 0 && eid < n_expert) ? res[eid] : -1;
        if (slot < 0 && eid >= 0 && eid < n_expert && mir != nullptr) maddr = mir[eid];
        if (usage != nullptr && eid >= 0 && eid < n_expert)   // the adaptive tier's routing counts (device side)
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(usage[eid]).fetch_add(1);
        if (slot < 0 && maddr == 0)
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                &s_bad, 1);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (stats != nullptr) {
        const bool valid = tid < n && eid >= 0 && eid < n_expert;
        const int n_valid = sycl::reduce_over_group(item_ct1.get_group(), valid ? 1 : 0, sycl::plus<int>());
        const int n_hit = sycl::reduce_over_group(item_ct1.get_group(), valid && slot >= 0 ? 1 : 0,
                                                  sycl::plus<int>());
        if (tid == 0) {
            sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(stats[1])
                .fetch_add((unsigned long long) n_hit);
            sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(stats[2])
                .fetch_add((unsigned long long) n_valid);
        }
    }
    if (s_bad) {
        if (tid == 0 && skip != nullptr) *skip = 0;
        if (tid == 0 && skip == nullptr) {   // #871: the all-resident graph has no host plan to fall back on
            pl[0] = 0; pl[1] = 0; pl[2] = 0;   // an empty plan: no expert runs on a stale pointer
            if (plan_err != nullptr) {
                    *plan_err = 1;
                    sycl::atomic_fence(sycl::memory_order::acq_rel,
                                       sycl::memory_scope::system);
            }
        }
        return;
    }

    if (lru_owner != nullptr && lru_layer >= 0) {
        auto &s_slot = *sycl::ext::oneapi::group_local_memory_for_overwrite<
            int32_t[kResidentPlanMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
        if (tid < n) s_slot[tid] = slot;
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (tid == 0) {
            int32_t* res_l = lru_res + (size_t) lru_layer * (size_t) n_expert;
            const uint32_t t = lru_state[0] + 1u;
            const uint32_t lru_end = lru_state[2];
            lru_state[0] = t;
            uint32_t hand = lru_state[1] < lru_end ? lru_state[1] : 0u;
            uint32_t jobs = 0;
            for (int j = 0; j < n; ++j)
                if (s_slot[j] >= 0) lru_last[s_slot[j]] = t;
            for (int j = 0; j < n && lru_end > 0; ++j) {
                if (s_slot[j] >= 0) continue;
                const int32_t e = s_ids[j];
                if (e < 0 || e >= n_expert) continue;
                const int32_t cur = res_l[e];
                if (cur >= 0) { s_slot[j] = cur; continue; }
                const unsigned long long src = mir != nullptr ? mir[e] : 0ull;
                if (src == 0ull) continue;
                int32_t chosen = -1;
                uint32_t best = 0xffffffffu;
                for (uint32_t step = 0; step < lru_sample; ++step) {
                    const uint32_t sl = hand;
                    hand = hand + 1u >= lru_end ? 0u : hand + 1u;
                    const uint32_t ls = lru_last[sl];
                    if (t - ls > lru_guard && (chosen < 0 || ls < best)) { chosen = (int32_t) sl; best = ls; }
                }
                for (uint32_t step = lru_sample; chosen < 0 && step < lru_scan; ++step) {
                    const uint32_t sl = hand;
                    hand = hand + 1u >= lru_end ? 0u : hand + 1u;
                    if (t - lru_last[sl] > lru_guard) chosen = (int32_t) sl;
                }
                if (chosen < 0) continue;
                const int32_t o = lru_owner[chosen];
                if (o >= 0 && lru_res[o] == chosen) lru_res[o] = -1;
                res_l[e] = chosen;
                lru_owner[chosen] = lru_layer * n_expert + e;
                lru_last[chosen] = t;
                lru_jobs[2 * jobs] = src;
                lru_jobs[2 * jobs + 1] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[chosen]
                                                                                      : (size_t) chosen * (size_t) blob));
                ++jobs;
                s_slot[j] = chosen;
            }
            lru_state[1] = hand;
            *lru_njobs = jobs;
            if (stats != nullptr) stats[0] += (unsigned long long) jobs;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (tid < n && s_slot[tid] >= 0) { slot = s_slot[tid]; maddr = 0; }
    }

    int first_j = tid;
    int rank_in_group = 0;
    int count_same = 0;
    if (tid < n) {
#pragma unroll
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == eid) {
                if (j < first_j) first_j = j;
                if (j < tid) ++rank_in_group;
                ++count_same;
            }
        }
    }
    const bool is_first = (tid < n && first_j == tid && (slot >= 0 || maddr != 0));
    const int my_cnt = is_first ? count_same : 0;
    const int my_pack = (my_cnt << 16) | (is_first ? 1 : 0);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int pref = my_pack;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int up = dpct::experimental::shift_sub_group_right(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            pref, d);
        if (lane >= d) pref += up;
    }
    s_excl[tid] = pref - my_pack;
    if (lane == 31) {
        s_wsum[warp] = pref;
    }
    item_ct1.barrier();

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    if (is_first) {
        int tot = s_excl[tid];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < warp) tot += s_wsum[w];
        }
        const int grp_idx = tot & 0xffff;
        const int ent_start = tot >> 16;
        ptr[grp_idx] = slot >= 0 ? (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob))
                                 : maddr;
        start[grp_idx] = ent_start;
    }
    if (tid < n) {
        const int fj_warp = first_j >> 5;
        int fj_tot = s_excl[first_j];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < fj_warp) fj_tot += s_wsum[w];
        }
        const int out_idx = (fj_tot >> 16) + rank_in_group;
        dst[out_idx] = tid;
        tok[out_idx] = tid / k;
    }
    if (tid == 0) {
        const int groups = (s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3]) & 0xffff;
        start[groups] = n;
        start2[0] = n;
        counts[0] = groups;
        counts[1] = n;
        counts[2] = 0;
        if (skip != nullptr) {
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::device);
            strata::sys_store(skip, ring);
        }
    }
}
__dpct_inline__ void lru_copy_kernel(const unsigned long long *jobs, const uint32_t *njobs, long long chunks) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long total = (long long) *njobs * chunks;
    const long long stride = (long long) it.get_global_range(2);
    for (long long i = (long long) it.get_global_id(2); i < total; i += stride) {
        const long long j = i / chunks, c = i - j * chunks;
        const sycl::uint4 *src = (const sycl::uint4 *) jobs[2 * j];
        sycl::uint4 *dst = (sycl::uint4 *) jobs[2 * j + 1];
        dst[c] = src[c];
    }
}
// The same plan in one block of 128 threads (n <= 128): thread i owns entry i. Groups are the distinct experts in
// order of first occurrence; group g's entries are its occurrences in increasing i - exactly the loop above
// (S26: the one-thread loop took ~44 us per call on gfx1151, 48 per window).
void resident_plan_par_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                         int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                         long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                         uint32_t ring) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_id = *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[128]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_first =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_size =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[128]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_bad = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int i = item_ct1.get_local_id(2);
    if (i == 0) s_bad = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int32_t e = -1;
    if (i < n) {
        e = ids[i];
        s_id[i] = e;
        if (e < 0 || e >= n_expert || res[e] < 0) s_bad = 1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (s_bad) { if (i == 0) *skip = 0; return; }
    int first = i, rank = 0, size = 0;
    if (i < n) {
        for (int j = 0; j < n; ++j) {
            const bool same = s_id[j] == e;
            if (same && j < first) first = j;
            if (same && j < i) ++rank;
            size += same;
        }
    }
    if (i < n) { s_first[i] = first; s_size[i] = size; }
    item_ct1.barrier();
    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;
    if (i < n) {
        // group of entry i = the number of first occurrences before `first`; its start = their sizes' sum
        int group = 0, gstart = 0;
#pragma unroll
        for (int j = 0; j < first; ++j)
            if (s_first[j] == j) { ++group; gstart += s_size[j]; }
        if (i == first) {
            const int32_t slot = res[e];
            ptr[group] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
            start[group] = gstart;
        }
        dst[gstart + rank] = i;
        tok[gstart + rank] = i / k;
        if (i == 0) {
            int groups = 0;
#pragma unroll
            for (int j = 0; j < n; ++j) groups += s_first[j] == j;
            start[groups] = n;
            start2[0] = n;
            counts[0] = groups;
            counts[1] = n;
            counts[2] = 0;
        }
    }
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    item_ct1.barrier();
    if (i == 0) *skip = ring;
}
__dpct_inline__ void wait_flag_ge_or_kernel(const volatile uint32_t *flag,
                                            uint32_t value,
                                            const volatile uint32_t *skip, uint32_t spin_max) {
    if (strata::sys_load(skip) == value) return;
    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
__dpct_inline__ void copy_i32_unless_kernel(int32_t *__restrict__ dst,
                                            const volatile int32_t *src, int n,
                                            const uint32_t *skip,
                                            uint32_t value) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (strata::sys_load(skip) == value) return;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) dst[i] = src[i];
}
__dpct_inline__ void copy_or_zero_kernel(sycl::float4 *__restrict__ dst,
                                         const sycl::float4 *src, long long n4,
                                         const uint32_t *skip, uint32_t value, bool pair) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const bool zero = *skip == value;
#pragma unroll
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n4; i += (long long)item_ct1.get_group_range(2) *
                      item_ct1.get_local_range(2))
        dst[i] = zero ? sycl::float4(0.f, 0.f, 0.f, 0.f)
                      : (pair ? strata::load_mapped_float4(src + i) : const_cast<const sycl::float4 *>(src)[i]);
}

__dpct_inline__ void gpu_stamp_kernel(unsigned long long *buf, int i, const uint64_t *word) {
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    buf[i] = strata::sys_load_mapped(word);
}
}  // namespace

#include "verify_kernels.kern.inc"

}  // namespace strata::kernels
