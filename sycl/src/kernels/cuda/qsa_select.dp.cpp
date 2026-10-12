// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#include "strata/port.hpp"
#include "strata/kernels/qsa_select.hpp"

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__dpct_inline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = sycl::bit_cast<unsigned int>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__dpct_inline__ void block_scores_kernel(const float *__restrict__ pooled,
                                         const float *__restrict__ dead,
                                         const float *__restrict__ q_idx,
                                         const int32_t *__restrict__ steps,
                                         int64_t max_blocks,
                                         float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t qi = item_ct1.get_group(1);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t)item_ct1.get_group(2) * SCORE_WARPS +
                      (item_ct1.get_local_id(2) >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const sycl::float4 k4 =
        *reinterpret_cast<const sycl::float4 *>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const sycl::float4 q4 =
            *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
        float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
                  k4.w() * q4.w();
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

__dpct_inline__ void block_topk_kernel(const float *__restrict__ scores,
                                       const int32_t *__restrict__ steps,
                                       int64_t max_blocks, int64_t cap,
                                       int32_t *__restrict__ ids) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &hist = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_a =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TOPK_T]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_b =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TOPK_T]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_digit = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_above = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item_ct1.get_local_id(2);
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
#pragma unroll
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        item_ct1.barrier();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                &hist[(k >> shift) & 255], w);
        }
        item_ct1.barrier();
        if (t == 0) {
            int cum = above, d = 255;
#pragma unroll
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        item_ct1.barrier();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        item_ct1.barrier();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    item_ct1.barrier();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    item_ct1.barrier();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    item_ct1.barrier();
    s_a[t] = sel;
    item_ct1.barrier();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    item_ct1.barrier();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
#pragma unroll
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t)(b * R + c);
        } else if (k == thr) {
#pragma unroll
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
}


// ---- QSA select on tensor cores (perf-review, after D-1): the block scores of many queries are one GEMM,
// rows (query, indexer head) x columns (blocks), K = 128, with relu per head summed. 3xTF32 (each operand split
// into a TF32 hi and lo part, hi*hi + hi*lo + lo*hi) keeps FP32-level accuracy; the summation order differs from
// the warp kernel, so a score can move in its last bits and a near-tie can select differently (not bitwise).
// Blocks < n_bid only; the tail block n_bid (the `dead` key, +1e9) is scored by the warp kernel's own code.
constexpr int TC_QT = 16;                 // queries per CTA (one m16 tile per indexer head)
constexpr int TC_NB = 32;                 // blocks per tile (4 warps x n8)
constexpr int TC_ITER = 4;                // tiles per CTA (the query tile is loaded once)
constexpr int TC_QS = IDX_HEADS * IDX_DIM + 4;   // query row stride in floats
constexpr int TC_KS = IDX_DIM + 4;               // key row stride

// TF32 conversion and MMA need sm_80: below it they compile to a trap and qsa_block_scores_tc refuses the device
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the warp kernel (below)
#define STRATA_SEL_SM80 0
#elif 0   // SYCL: inline PTX (mma/ldmatrix/cp.async) - the XMX port is pending; see tools/fixups.py
#define STRATA_SEL_SM80 1
#else
#define STRATA_SEL_SM80 0
#endif
__dpct_inline__ uint32_t tf32_hi(float x) {
#if STRATA_SEL_SM80
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    return sycl::bit_cast<uint32_t>(x);
#endif
}
__dpct_inline__ void mma_tf32(float *c, const uint32_t *a, const uint32_t *b) {
#if !STRATA_SEL_SM80
    /* unreachable on SYCL: the launcher refuses this device */
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]));
#endif
}

__dpct_inline__ void block_scores_tc_kernel(
    const float *__restrict__ pooled, const float *__restrict__ q_idx,
    const int32_t *__restrict__ steps, int64_t nq, int64_t max_blocks,
    int64_t reach, float *__restrict__ out, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sm = (float *)dpct_local;
    float* sQ = sm;                                   // [TC_QT][TC_QS]
    float* sK = sm + TC_QT * TC_QS;                   // [TC_NB][TC_KS]
    auto &s_nbid =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TC_QT]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5,
              gid = lane >> 2, tig = lane & 3;
    const int64_t q0 = (int64_t)item_ct1.get_group(1) * TC_QT;
    for (int i = t; i < TC_QT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int r = i / (IDX_HEADS * IDX_DIM / 4), c = i % (IDX_HEADS * IDX_DIM / 4);
        sycl::float4 v = sycl::float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) v = reinterpret_cast<const sycl::float4 *>(
            q_idx + (q0 + r) * IDX_HEADS * IDX_DIM)[c];
        *reinterpret_cast<sycl::float4 *>(sQ + r * TC_QS + c * 4) = v;
    }
    if (t < TC_QT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    int lo_nbid = 0x7fffffff, hi_nbid = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int i = 0; i < TC_QT; ++i) {
        if (q0 + i >= nq) break;
        lo_nbid = sycl::min(lo_nbid, s_nbid[i]);
        hi_nbid = sycl::max(hi_nbid, s_nbid[i]);
    }
    for (int it = 0; it < TC_ITER; ++it) {
        const int64_t b0 =
            ((int64_t)item_ct1.get_group(2) * TC_ITER + it) * TC_NB;
        if (b0 >= reach || b0 >= hi_nbid) break;      // blocks >= every query's n_bid: nothing to score
        item_ct1.barrier(); // the previous tile's reads are done
        for (int i = t; i < TC_NB * IDX_DIM / 4; i += 128) {
            const int r = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
            sycl::float4 v = sycl::float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + r < hi_nbid) v = reinterpret_cast<const sycl::float4 *>(
                pooled + (b0 + r) * IDX_DIM)[c];
            *reinterpret_cast<sycl::float4 *>(sK + r * TC_KS + c * 4) = v;
        }
        item_ct1.barrier();
        float acc[IDX_HEADS][4];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
        const float* kr = sK + (warp * 8 + gid) * TC_KS;
#pragma unroll 4
        for (int k0 = 0; k0 < IDX_DIM; k0 += 8) {
            const float kx0 = kr[k0 + tig], kx1 = kr[k0 + tig + 4];
            uint32_t bh[2], bl[2];
            bh[0] = tf32_hi(kx0);
            bh[1] = tf32_hi(kx1);
            bl[0] = tf32_hi(kx0 - sycl::bit_cast<float>(bh[0]));
            bl[1] = tf32_hi(kx1 - sycl::bit_cast<float>(bh[1]));
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float* qa = sQ + h * IDX_DIM + k0 + tig;
                const float x0 = qa[gid * TC_QS], x1 = qa[(gid + 8) * TC_QS];
                const float x2 = qa[gid * TC_QS + 4], x3 = qa[(gid + 8) * TC_QS + 4];
                uint32_t ah[4], al[4];
                ah[0] = tf32_hi(x0); ah[1] = tf32_hi(x1); ah[2] = tf32_hi(x2); ah[3] = tf32_hi(x3);
                al[0] = tf32_hi(x0 - sycl::bit_cast<float>(ah[0]));
                al[1] = tf32_hi(x1 - sycl::bit_cast<float>(ah[1]));
                al[2] = tf32_hi(x2 - sycl::bit_cast<float>(ah[2]));
                al[3] = tf32_hi(x3 - sycl::bit_cast<float>(ah[3]));
                mma_tf32(acc[h], al, bh);
                mma_tf32(acc[h], ah, bl);
                mma_tf32(acc[h], ah, bh);
            }
        }
        // relu per head, heads added in order (as the warp kernel), written where the block completed for the query
        const int64_t bc = b0 + warp * 8 + 2 * tig;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int qr = gid + half * 8;
            const int64_t qi = q0 + qr;
            if (qi >= nq) continue;
            const int nb = s_nbid[qr];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int64_t b = bc + j;
                if (b >= nb || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][half * 2 + j];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

#if defined(__HIPCC__)
// ---- gfx12 (RDNA4: gfx1200 / gfx1201) scorer on v_wmma_f32_16x16x16_bf16 (wave32).  Same GEMM as above: for a tile of 16
// queries, every block's score is sum over the 4 indexer heads of relu(q_h . k), K = 128.  FP32-level accuracy from a
// three-way bf16 split of both operands (x = hi + mid + lo EXACTLY - the top 8, next 8 and last 8 bits of the fp32
// mantissa, so bf16's fp32-sized exponent range needs no scaling) and the six products of order <= 2, smallest first;
// like the TF32 kernel it sums in another order than the warp kernel (not bitwise).  The tail block n_bid is the warp
// kernel's arithmetic (block_scores_tail_kernel).
//   D = A x B with A = 16 key blocks (row l%16 of a lane, k = 8*(l/16)+i), B = 16 queries (column l%16, same k):
//   a lane ends up with 8 consecutive blocks of ONE query, written as one run.  No LDS for keys: a lane reads its
//   32-byte slice of the key row from global (the 16 query tiles of a launch re-read them from L2); the queries are
//   split once per CTA into LDS.
// gfx11 (RDNA3 / RDNA3.5) has the same instruction with another fragment layout: a lane holds all 16 k of its row
// (A) or column (B) - lanes 16..31 repeat lanes 0..15 - and D[2i + l/16][l%16]; SEL_KL is the k per lane, SEL_KOFF(g)
// the first, SEL_ROW(i, g) the D row of element i.  Same products, same order per element as on gfx12.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_SEL_GFX12 1
#define STRATA_SEL_GFX11 0
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || \
    defined(__gfx1151__)
#define STRATA_SEL_GFX12 0
#define STRATA_SEL_GFX11 1
#else
#define STRATA_SEL_GFX12 0
#define STRATA_SEL_GFX11 0
#endif
#if STRATA_SEL_GFX11
constexpr int SEL_KL = 16;
#define SEL_KOFF(g) 0
#define SEL_ROW(i, g) (2 * (i) + (g))
#else
constexpr int SEL_KL = 8;
#define SEL_KOFF(g) ((g) * 8)
#define SEL_ROW(i, g) ((g) * 8 + (i))
#endif
typedef short sel_s8 __attribute__((ext_vector_type(SEL_KL)));
typedef float sel_f8 __attribute__((ext_vector_type(8)));
typedef uint32_t sel_u4 __attribute__((ext_vector_type(SEL_KL / 2)));
constexpr int WQT = 16;                    // queries per CTA (the N of one WMMA)
constexpr int WITER = 4;                   // key tiles per warp (the CTA covers 4 warps * WITER * 16 blocks)
constexpr int WQS = IDX_DIM + 8;           // bf16 elements per LDS row: 272 bytes, conflict-free 16-byte reads

__device__ __forceinline__ sel_f8 wmma_bf16(const sel_s8& a, const sel_s8& b, const sel_f8& c) {
#if STRATA_SEL_GFX12
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#elif STRATA_SEL_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
    /* unreachable on SYCL: the launcher refuses this device */
    return c;
#endif
}
// the bf16 pieces of x as fp32 bit patterns whose upper 16 bits are the bf16 (lower 16 are zero)
__device__ __forceinline__ void split3(float x, uint32_t& hi, uint32_t& mid, uint32_t& lo) {
    hi = sycl::bit_cast<uint32_t>(x) & 0xffff0000u;
    const float r1 = x - __uint_as_float(hi);                   // exact
    mid = sycl::bit_cast<uint32_t>(r1) & 0xffff0000u;
    lo = sycl::bit_cast<uint32_t>(r1 - __uint_as_float(mid)) & 0xffff0000u;   // exact: at most 8 significant bits left
}
__device__ __forceinline__ uint32_t pack_bf16x2(uint32_t a, uint32_t b) {   // low half = a's bf16, high half = b's
#if STRATA_SEL_GFX12 || STRATA_SEL_GFX11
    return __builtin_amdgcn_perm(b, a, 0x07060302u);
#else
    return (a >> 16) | (b & 0xffff0000u);
#endif
}

__global__ void __launch_bounds__(128) block_scores_wmma_kernel(const float* __restrict__ pooled,
                                                                const float* __restrict__ q_idx,
                                                                const int32_t* __restrict__ steps, int64_t nq,
                                                                int64_t max_blocks, int64_t reach,
                                                                float* __restrict__ out) {
    __shared__ __align__(16) uint16_t sq[3][IDX_HEADS][WQT][WQS];   // hi / mid / lo, [head][query][dim]
    __shared__ int s_nbid[WQT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, l16 = lane & 15, g = lane >> 4;
    const int64_t q0 = (int64_t) blockIdx.y * WQT;
    for (int i = t; i < WQT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int row = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
        const int qr = row / IDX_HEADS, h = row % IDX_HEADS;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + qr < nq) v = reinterpret_cast<const float4*>(q_idx + (q0 + qr) * IDX_HEADS * IDX_DIM)[h * (IDX_DIM / 4) + c];
        const float x[4] = {v.x, v.y, v.z, v.w};
        uint32_t hb[4], mb[4], lb[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) split3(x[j], hb[j], mb[j], lb[j]);
        *reinterpret_cast<uint2*>(&sq[0][h][qr][c * 4]) = make_uint2(pack_bf16x2(hb[0], hb[1]), pack_bf16x2(hb[2], hb[3]));
        *reinterpret_cast<uint2*>(&sq[1][h][qr][c * 4]) = make_uint2(pack_bf16x2(mb[0], mb[1]), pack_bf16x2(mb[2], mb[3]));
        *reinterpret_cast<uint2*>(&sq[2][h][qr][c * 4]) = make_uint2(pack_bf16x2(lb[0], lb[1]), pack_bf16x2(lb[2], lb[3]));
    }
    if (t < WQT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    __syncthreads();
    int hi_nbid = 0;
#pragma unroll
    for (int i = 0; i < WQT; ++i) hi_nbid = max(hi_nbid, s_nbid[i]);
    const int64_t qi = q0 + l16;
    const int nb_q = s_nbid[l16];
    for (int it = 0; it < WITER; ++it) {
        const int64_t b0 = (((int64_t) blockIdx.x * WITER + it) * 4 + warp) * 16;
        if (b0 >= reach || b0 >= hi_nbid) break;          // warp-uniform; later tiles start higher
        const int64_t row = b0 + l16;
        const bool rv = row < hi_nbid && row < max_blocks;
        const float* kp = pooled + (rv ? row : 0) * IDX_DIM + SEL_KOFF(g);
        // acc: the hi*hi products; cor: the five smaller ones (<= 2^-8 of it). Kept apart and added once at the end: a
        // correction summed into the big accumulator is rounded to ITS ulp at every one of the 48 steps (measured: 8e-7
        // of the score scale; apart, near the warp kernel's)
        sel_f8 acc[IDX_HEADS], cor[IDX_HEADS];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h] = cor[h] = sel_f8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
        for (int kk = 0; kk < IDX_DIM / 16; ++kk) {
            float x[SEL_KL];
#pragma unroll
            for (int j = 0; j < SEL_KL / 4; ++j) {
                float4 kv = make_float4(0.f, 0.f, 0.f, 0.f);
                if (rv) kv = *reinterpret_cast<const float4*>(kp + kk * 16 + 4 * j);
                x[4 * j] = kv.x; x[4 * j + 1] = kv.y; x[4 * j + 2] = kv.z; x[4 * j + 3] = kv.w;
            }
            uint32_t hb[SEL_KL], mb[SEL_KL], lb[SEL_KL];
#pragma unroll
            for (int j = 0; j < SEL_KL; ++j) split3(x[j], hb[j], mb[j], lb[j]);
            sel_u4 ah, am, al;
#pragma unroll
            for (int j = 0; j < SEL_KL / 2; ++j) {
                ah[j] = pack_bf16x2(hb[2 * j], hb[2 * j + 1]);
                am[j] = pack_bf16x2(mb[2 * j], mb[2 * j + 1]);
                al[j] = pack_bf16x2(lb[2 * j], lb[2 * j + 1]);
            }
            const sel_s8 Ah = __builtin_bit_cast(sel_s8, ah), Am = __builtin_bit_cast(sel_s8, am),
                         Al = __builtin_bit_cast(sel_s8, al);
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const sel_s8 Bh = *reinterpret_cast<const sel_s8*>(&sq[0][h][l16][kk * 16 + SEL_KOFF(g)]);
                const sel_s8 Bm = *reinterpret_cast<const sel_s8*>(&sq[1][h][l16][kk * 16 + SEL_KOFF(g)]);
                const sel_s8 Bl = *reinterpret_cast<const sel_s8*>(&sq[2][h][l16][kk * 16 + SEL_KOFF(g)]);
                cor[h] = wmma_bf16(Al, Bh, cor[h]);
                cor[h] = wmma_bf16(Ah, Bl, cor[h]);
                cor[h] = wmma_bf16(Am, Bm, cor[h]);
                cor[h] = wmma_bf16(Ah, Bm, cor[h]);
                cor[h] = wmma_bf16(Am, Bh, cor[h]);
                acc[h] = wmma_bf16(Ah, Bh, acc[h]);
            }
        }
        // relu per head, heads added in order (as the warp kernel); this lane: query qi, blocks b0 + SEL_ROW(i, g)
        if (qi < nq) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int64_t b = b0 + SEL_ROW(i, g);
                if (b >= nb_q || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][i] + cor[h][i];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

#endif  // __HIPCC__

// the tail block n_bid of each query: exactly block_scores_kernel's arithmetic for that block
__dpct_inline__ void block_scores_tail_kernel(const float *__restrict__ dead,
                                              const float *__restrict__ q_idx,
                                              const int32_t *__restrict__ steps,
                                              int64_t max_blocks,
                                              float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    if (n_bid >= max_blocks) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const sycl::float4 k4 =
        *reinterpret_cast<const sycl::float4 *>(dead + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const sycl::float4 q4 =
            *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
        float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
                  k4.w() * q4.w();
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + n_bid] = score;
    }
}

// ---- the same top-k with each query's keys read once: 1,024 threads hold up to TK_PER consecutive blocks' keys in
// registers (contexts up to 4 * 1024 * TK_PER cells), per-warp histograms, block-wide scans. The selection rule is
// block_topk_kernel's (radix threshold, ties to the lowest index, cells ascending): identical ids.
constexpr int TK_T = 1024;
constexpr int TK_PER = 33;
#if defined(__HIPCC__)
// AMD (RDNA, wave32): a 1,024-thread block may use 192 VGPRs per lane, so the keys of 66 blocks per thread fit - contexts
// up to 4 * 1024 * 66 = 270,336 cells (the 262,144 --max-context) keep the register kernel. NVIDIA's 64 registers per
// thread at 1,024 threads allow only TK_PER.
constexpr int TK_PER_MAX = 66;
#else
constexpr int TK_PER_MAX = TK_PER;
#endif

__dpct_inline__ int block_excl_scan(int v, int *s_warp, int &total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = dpct::experimental::shift_sub_group_right(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    item_ct1.barrier();
    if (warp == 0) {
        int w = s_warp[lane];
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;               // exclusive per warp
        if (lane == 31) s_warp[32] = z;     // total
    }
    item_ct1.barrier();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    item_ct1.barrier();
    return r;
}

template <int PER>
__dpct_inline__ void block_topk_reg_kernel(const float *__restrict__ scores,
                                           const int32_t *__restrict__ steps,
                                           int64_t max_blocks, int64_t cap,
                                           int32_t *__restrict__ ids) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &hist =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TK_T / 32][256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_warp =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[33]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_digit = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_above = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
#pragma unroll
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // <= PER (the caller checks)
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if ((key[j] & hi_mask) == (prefix & hi_mask))
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &hist[warp][(key[j] >> shift) & 255], w);
        }
        item_ct1.barrier();
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int s = 0;
#pragma unroll
            for (int w2 = 0; w2 < TK_T / 32; ++w2) s += hist[w2][t];
            hist[0][t] = s;
        }
        item_ct1.barrier();
        if (t == 0) {
            int cum = above, d = 255;
#pragma unroll
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
        item_ct1.barrier();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        item_ct1.barrier();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) gt += w;
        else if (key[j] == thr) eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = block_excl_scan(sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
#pragma unroll
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t)(b * R + c);
        } else if (key[j] == thr) {
#pragma unroll
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
}

#if !defined(__HIPCC__)
// ---- a query with more blocks than the register kernel holds (contexts past 4 * 1024 * TK_PER cells): the register
// kernel's threads, per-warp histograms and warp scans, with each key read from memory again on every pass.  A
// histogram has no order, so on the four radix passes thread t reads blocks t, t + 1024, ...: a warp reads 32
// neighbours at a time.  The cells are emitted ascending in the order (warp, row, lane): warp w holds the `per` rows
// of 32 consecutive blocks from block w * 32 * per.  block_topk_kernel's selection rule (radix threshold, ties to the
// lowest index): identical ids.
__dpct_inline__ void block_topk_wide_kernel(const float *__restrict__ scores,
                                            const int32_t *__restrict__ steps,
                                            int64_t max_blocks, int64_t cap,
                                            int32_t *__restrict__ ids) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &hist =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TK_T / 32][256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_gt =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TK_T / 32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_eq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TK_T / 32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_digit = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_above = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
#pragma unroll
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // rows of 32 blocks per warp
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = t; b < nb; b += TK_T) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                &hist[warp][(k >> shift) & 255], w);
        }
        item_ct1.barrier();
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int sum = 0;
#pragma unroll
            for (int w2 = 0; w2 < TK_T / 32; ++w2) sum += hist[w2][t];
            hist[0][t] = sum;
        }
        item_ct1.barrier();
        if (t == 0) {
            int cum = above, d = 255;
#pragma unroll
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
        item_ct1.barrier();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        item_ct1.barrier();
    }
    const uint32_t thr = prefix;
    const int eq_budget = (int) (width - above);      // cells equal to thr that fit, lowest index first
    const int64_t w0 = (int64_t) warp * 32 * per;
    int gt = 0, eq = 0;
    for (int64_t b = w0 + lane; b < nb && b < w0 + 32 * per; b += 32) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), gt,
            o);
        eq += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), eq,
            o);
    }
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    item_ct1.barrier();
    if (t == 0) {                                     // per warp: the selected cells and the tied cells before it
        int eb = 0, sb = 0;
        for (int w2 = 0; w2 < TK_T / 32; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb;
            s_eq[w2] = eb;
            sb += g2 + take;
            eb += e2;
        }
    }
    item_ct1.barrier();
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    for (int64_t r0 = w0; r0 < nb && r0 < w0 + 32 * per; r0 += 32) {
        const int64_t b = r0 + lane;
        const int w = b < nb ? weight(b) : 0;
        const uint32_t k = w ? order_key(sc[b]) : 0u;
        const int my_gt = (w && k > thr) ? w : 0, my_eq = (w && k == thr) ? w : 0;
        if (sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(),
                (0xffffffffu &
                 (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_id())) &&
                        (my_gt | my_eq) != 0
                    ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())
                    : 0,
                sycl::ext::oneapi::plus<>()) ==
            0u) continue; // a row without a selected block
        int pe = my_eq;                               // tied cells up to and with this lane
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                pe, o);
            if (lane >= o) pe += y;
        }
        const int left = eq_budget - (run_eq + pe - my_eq);
        const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
        const int my_sel = my_gt + take;
        int ps = my_sel;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                ps, o);
            if (lane >= o) ps += y;
        }
        int32_t* dst = out + run_sel + ps - my_sel;
#pragma unroll
        for (int c = 0; c < my_sel; ++c) dst[c] = (int32_t)(b * R + c);
        run_eq += dpct::experimental::select_from_sub_group(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), pe,
            31);
        run_sel += dpct::experimental::select_from_sub_group(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), ps,
            31);
    }
}
#endif


// Block scores with every key block read ONCE for all of a call's queries (block_scores_kernel's grid is
// (max_blocks / 8) x nq: ~24,600 mostly-idle blocks per layer at a decode window, each key re-read per query).  A fixed
// grid strides over the blocks; per (block, query) the same arithmetic in the same order as block_scores_kernel.
// qsa_block_scores takes it for every call without an active-block count and at most MQ queries: the captured decode
// window, the uncaptured decode, and prefill's pooled16 call.
constexpr int MQ = 8;
__dpct_inline__ void block_scores_multi_kernel(
    const float *__restrict__ pooled, const float *__restrict__ dead,
    const float *__restrict__ q_idx, const int32_t *__restrict__ steps, int nq,
    int64_t max_blocks, float *__restrict__ out, bool early_exit) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &qs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[MQ * IDX_HEADS * IDX_DIM]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_nkv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[MQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_nbid =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[MQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (!early_exit)
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < nq * IDX_HEADS * IDX_DIM;
             i += item_ct1.get_local_range(2)) qs[i] = q_idx[i];
    if (item_ct1.get_local_id(2) < nq) {
        s_nkv[item_ct1.get_local_id(2)] =
            steps[item_ct1.get_local_id(2) * kStepCount + kStepNKv];
        s_nbid[item_ct1.get_local_id(2)] =
            steps[item_ct1.get_local_id(2) * kStepCount + kStepNBid];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int64_t top = 0;
#pragma unroll
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if (early_exit) {
        // (#783, stuchapin909) a CTA whose first block is past the batch's largest n_bid has no work: the grid is a
        // fixed 256 CTAs and a decode window reaches only a few of them, so those return before staging the queries.
        // Every thread of the CTA takes the same branch (top and blockIdx are CTA-uniform), so the barrier is safe.
        if ((int64_t)item_ct1.get_group(2) * SCORE_WARPS > top) return;
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < nq * IDX_HEADS * IDX_DIM;
             i += item_ct1.get_local_range(2)) qs[i] = q_idx[i];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    const int lane = item_ct1.get_local_id(2) & 31;
    const int64_t wstride = (int64_t)item_ct1.get_group_range(2) * SCORE_WARPS;
    for (int64_t b = (int64_t)item_ct1.get_group(2) * SCORE_WARPS +
                     (item_ct1.get_local_id(2) >> 5);
         b <= top && b < max_blocks; b += wstride) {
        const sycl::float4 kp = *reinterpret_cast<const sycl::float4 *>(
            pooled + b * IDX_DIM + lane * 4);
        const sycl::float4 kd =
            *reinterpret_cast<const sycl::float4 *>(dead + lane * 4);
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const sycl::float4 k4 = (b == n_bid) ? kd : kp;
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const sycl::float4 q4 =
                    *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
                float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
                          k4.w() * q4.w();
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) d +=
                    dpct::experimental::permute_sub_group_by_xor(
                        0xffffffffu,
                        sycl::ext::oneapi::this_work_item::get_sub_group(), d,
                        o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

#if !defined(__HIPCC__)
// ---- the decode top-k on a thread-block cluster (sm_90+; S19).  One CTA per query (block_topk_reg_kernel, or
// block_topk_kernel above 33,792 blocks: a --max-context over ~135K) makes four radix passes and two scans over up to
// 65,538 blocks on ONE SM while the rest of the GPU idles - a decode window has 1-5 queries.  Here a cluster of CL_N
// CTAs shares a query: CTA r holds the keys of blocks [r * per, (r + 1) * per) in shared memory, builds the digit
// histogram of its keys, and PUSHES it into slot r of every CTA's `hin` (distributed shared memory); after one cluster
// barrier each CTA sums the CL_N slots itself, so all of them take the same digit.  The emit pass needs each CTA's
// cells above / at the threshold: pushed the same way, then each CTA offsets its own by the ranks before it.
// RTX 5070, per call at capacity = context (decode_cluster_parity --bench): 21.9 -> 15.6 us at 32K, 58 -> 18 at 128K,
// 200 -> 22 at 262K (the one-CTA kernels' digit search was serial, too: one thread over 256 bins per pass).
//
// IDENTICAL IDS, by construction rather than by luck: the threshold thr (the width-th largest key, cells counted with
// their weights) and `above` (cells with a larger key) are pure functions of the multiset of (key, weight) - integer
// histograms, summed in any order - and the digit rule is block_topk_kernel's, written as a scan (the largest digit d
// whose cells at or above it reach `need`; 0 when none). The cells emitted are then fixed: every cell of a block with
// key > thr, and the first eq_budget = width - above cells at thr in ascending order (a block at thr may be cut),
// written in ascending cell order. Position of a thread's first cell = (cells above thr before it) + min(cells at thr
// before it, eq_budget): the telescoped sum of the reference's per-thread clamp. NaN keys are 0 (order_key) as there.
//
// Barriers: barrier.cluster arrive / wait (each CTA's threads all take part; it also orders the CTA's own shared
// memory, so it doubles as __syncthreads). Phase 0 is a relaxed arrive at entry, waited before the first remote store
// (a CTA's shared memory exists once it runs). Pushes go out before the arrive that releases them, and nothing reads
// another CTA's memory after the last wait - so a CTA may exit without a closing barrier (a wait counts the threads
// that have not exited). `hin` alternates by pass parity: CTA x reads slot buffer p&1 in pass p before arriving at
// pass p+1's barrier, and nobody writes buffer p&1 again (pass p+2) before waiting on that barrier.
constexpr int CL_N = 8;        // CTAs per query (the portable cluster size)
constexpr int CL_T = 1024;     // threads per CTA
constexpr int CL_MAXQ = 16;    // larger calls (prefill sub-batches) fill the GPU with one CTA per query already
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 900
#define STRATA_SEL_CLUSTER 1
#else
#define STRATA_SEL_CLUSTER 0   // an older target's code is a trap; qsa_block_topk_cluster never launches it there
#endif

#if STRATA_SEL_CLUSTER
__dpct_inline__ void cl_arrive_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
__dpct_inline__ void cl_arrive() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n" :: : "memory");
}
__dpct_inline__ void cl_wait() {
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" :: : "memory");
}
__dpct_inline__ unsigned cl_rank() {
    unsigned r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}
template <typename T>
__dpct_inline__ T *cl_map(T *p,
                          unsigned rank) { // p in CTA `rank`'s shared memory
    uint64_t o;
    asm volatile("mapa.u64 %0, %1, %2;\n"
                 : "=l"(o)
                 : "l"((uint64_t)p), "r"(rank));
    return reinterpret_cast<T*>(o);
}
// exclusive block scan of a 64-bit value over CL_T threads; `total` = the CTA's sum
__dpct_inline__ unsigned long long cl_excl_scan64(unsigned long long v,
                                                  unsigned long long *s_warp,
                                                  unsigned long long &total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    unsigned long long x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned long long y = dpct::experimental::shift_sub_group_right(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    item_ct1.barrier();
    if (warp == 0) {
        const unsigned long long w = s_warp[lane];
        unsigned long long z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned long long y =
                dpct::experimental::shift_sub_group_right(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;
        if (lane == 31) s_warp[32] = z;
    }
    item_ct1.barrier();
    total = s_warp[32];
    return s_warp[warp] + x - v;
}
#endif

// grid (CL_N, nq), cluster (CL_N, 1, 1), CL_T threads, dynamic shared memory: ceil(max_blocks / CL_N) keys
void block_topk_cluster_kernel(const float *__restrict__ scores,
                               const int32_t *__restrict__ steps,
                               int64_t max_blocks, int64_t cap,
                               int32_t *__restrict__ ids, uint8_t *dpct_local) {
#if STRATA_SEL_CLUSTER
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto keys = (uint32_t *)dpct_local; // this CTA's blocks' keys
    auto &hin =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[2][CL_N][256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // the cluster's histograms, slot = the pushing rank
    auto &hloc =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[2][256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // this CTA's histogram
    auto &cnt = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        unsigned long long[CL_N]>(
        sycl::ext::oneapi::this_work_item::get_work_group<
            3>()); // per rank: its cells above thr (low 32 bits) and at it
                   // (high)
    auto &s_warp = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        unsigned long long[33]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_prefix =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_above = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const unsigned rank = cl_rank();
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    cl_arrive_relaxed();                                // phase 0: this CTA runs
    const int64_t qi = item_ct1.get_group(1);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    if (n_kv <= width) {                                // the identity, split over the cluster; no remote access
#pragma unroll
        for (int64_t j = (int64_t)rank * CL_T + t; j < n_kv;
             j += (int64_t)CL_N * CL_T) out[j] = (int32_t)j;
        return;
    }
    // the tail block n_bid weighs its cells (0..3); with none it is not a candidate (the reference skips w == 0)
    const int wtail = (int) (n_kv - n_bid * R);
    const int64_t nbe = n_bid + (wtail > 0 ? 1 : 0);
    const int64_t per = (nbe + CL_N - 1) / CL_N;       // <= ceil(max_blocks / CL_N), the keys the host sized
    const int64_t lo = (int64_t) rank * per < nbe ? (int64_t) rank * per : nbe;
    const int64_t hi = lo + per < nbe ? lo + per : nbe;
    const int n = (int) (hi - lo);
    const int tail_i = (wtail > 0 && n_bid >= lo && n_bid < hi) ? (int) (n_bid - lo) : -1;
    const float* sc = scores + qi * max_blocks + lo;
#pragma unroll
    for (int i = t; i < n; i += CL_T) keys[i] = order_key(sc[i]);
#pragma unroll
    for (int i = t; i < 2 * 256; i += CL_T)(&hloc[0][0])[i] = 0;
    item_ct1.barrier();
    uint32_t prefix = 0;
    int above = 0;                                      // cells strictly above the digits fixed so far
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        const uint32_t hmask = pass == 0 ? 0u : (0xffffffffu << (shift + 8));
        int* hl = hloc[pass & 1];
        // warp-aggregated: the scores share their top bits, so plain atomics would queue on a few bins
        for (int i0 = 0; i0 < n; i0 += CL_T) {
            const int i = i0 + t;
            int bin = -1;
            if (i < n) {
                const uint32_t k = keys[i];
                if ((k & hmask) == prefix) bin = (int) ((k >> shift) & 255);
            }
            const unsigned same = dpct::match_any_over_sub_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(), 0xffffffffu,
                bin);
            if (bin >= 0 && lane == dpct::ffs<int>(same) - 1)
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &hl[bin], sycl::popcount(same) * R);
            if (bin >= 0 && i == tail_i) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(&hl[bin],
                                                            wtail - R);
        }
        item_ct1.barrier();
        if (pass == 0) cl_wait();                       // phase 0 done: every CTA's shared memory is there
        for (int i = t; i < CL_N * 64; i += CL_T) {     // the histogram into slot `rank` of every CTA, as int4
            const int dst = i >> 6, c = i & 63;
            *cl_map(reinterpret_cast<sycl::int4 *>(&hin[pass & 1][rank][0]) + c,
                    (unsigned)dst) =
                reinterpret_cast<const sycl::int4 *>(hl)[c];
        }
#pragma unroll
        for (int i = t; i < 256; i += CL_T) hloc[(pass + 1) & 1][i] = 0;
        cl_arrive();
        cl_wait();
        if (warp == 0) {
            // lane L owns digits 255 - 8L down to 248 - 8L; the digit holding the need-th cell from the top
            const int need = (int) width - above;
            int tot[8], part = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int d = 255 - 8 * lane - k;
                int v = 0;
#pragma unroll
                for (int r = 0; r < CL_N; ++r) v += hin[pass & 1][r][d];
                tot[k] = v;
                part += v;
            }
            int incl = part;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = dpct::experimental::shift_sub_group_right(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), incl,
                    o);
                if (lane >= o) incl += y;
            }
            const int excl = incl - part;
            const unsigned hit = sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(),
                (0xffffffffu &
                 (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_id())) &&
                        excl < need && incl >= need
                    ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())
                    : 0,
                sycl::ext::oneapi::plus<>());
            // no hit cannot happen (the cells total n_kv > width); the reference would then take digit 0
            const int who = hit ? dpct::ffs<int>(hit) - 1 : 31;
            if (lane == who) {
                int acc = excl, k = 0;
#pragma unroll
                for (; k < 7; ++k) {
                    if (acc + tot[k] >= need) break;
                    acc += tot[k];
                }
                s_prefix = prefix | ((uint32_t) (255 - 8 * lane - k) << shift);
                s_above = above + acc;
            }
        }
        item_ct1.barrier();
        prefix = s_prefix;
        above = s_above;
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;           // cells at thr that fit, lowest index first
    // each thread a contiguous run of the CTA's keys; an odd run length keeps the shared reads conflict-free
    int seg = (n + CL_T - 1) / CL_T;
    if ((seg & 1) == 0 && seg > 0) ++seg;
    const int s0 = t * seg < n ? t * seg : n, s1 = s0 + seg < n ? s0 + seg : n;
    uint32_t gt = 0, eq = 0;
    for (int i = s0; i < s1; ++i) {
        const uint32_t k = keys[i], w = i == tail_i ? (uint32_t) wtail : (uint32_t) R;
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    unsigned long long cta;
    const unsigned long long before =
        cl_excl_scan64((unsigned long long) gt | ((unsigned long long) eq << 32), s_warp, cta);
    if (t < CL_N) *cl_map(&cnt[rank], (unsigned) t) = cta;
    cl_arrive();
    cl_wait();
    int64_t gt_off = 0, eq_off = 0;
#pragma unroll
    for (unsigned r = 0; r < rank; ++r) {
        gt_off += (int64_t) (cnt[r] & 0xffffffffu);
        eq_off += (int64_t) (cnt[r] >> 32);
    }
    const int64_t eq_before = eq_off + (int64_t) (before >> 32);
    int64_t wpos = gt_off + (int64_t) (before & 0xffffffffu) + (eq_before < eq_budget ? eq_before : eq_budget);
    int64_t eq_left = eq_budget - eq_before;
    if (eq_left < 0) eq_left = 0;
    for (int i = s0; i < s1; ++i) {
        const uint32_t k = keys[i];
        const int w = i == tail_i ? wtail : R;
        const int64_t b = lo + i;
        if (k > thr) {
#pragma unroll
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t)(b * R + c);
        } else if (k == thr) {
#pragma unroll
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
#else
    (void) scores; (void) steps; (void) max_blocks; (void) cap; (void) ids;
    /* unreachable on SYCL: the launcher refuses this device */
#endif
}
#endif  // !__HIPCC__
}  // namespace

#include "qsa_select.kern.inc"

}  // namespace strata::kernels
