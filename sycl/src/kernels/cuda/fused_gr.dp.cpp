// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.
#include "strata/port.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "fused_gr_shared.hpp"

namespace strata::kernels {
namespace {

__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
    return v;
}
__dpct_inline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

// 8 bf16 packed in a uint4 against 8 floats.
__dpct_inline__ float dot8(const sycl::uint4 w, const float *x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x(), w.y(), w.z(), w.w()};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = sycl::fma(sycl::bit_cast<float>(v[j] << 16), (float)(x[2 * j]),
                        acc);
        acc = sycl::fma(sycl::bit_cast<float>(v[j] & 0xffff0000u),
                        (float)(x[2 * j + 1]), acc);
    }
    return acc;
}

__dpct_inline__ void gr_down_kernel(FusedGrArgs a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &xn = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[D]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &part =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS][HC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_rs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[HC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    // 1. R' * w_norm into shared memory, and the per-stream sums of squares of R'.
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), gw[c], r.x());
                r.y() = sycl::fma((float)(b.y()), gw[c], r.y());
            r.z() = sycl::fma((float)(b.z()), gw[c], r.z());
                r.w() = sycl::fma((float)(b.w()), gw[c], r.w());
        }
        const sycl::float4 g =
            *reinterpret_cast<const sycl::float4 *>(a.w_norm + i);
        float sq =
            r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<sycl::float4 *>(xn + i) = sycl::float4(
            r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    item_ct1.barrier();
    if (t < HC) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
        if (item_ct1.get_group(2) == 0) a.rs[t] = s_rs[t];
    }
    item_ct1.barrier();
#pragma unroll
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
    item_ct1.barrier();
    // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
    const bool inject_block = item_ct1.get_group(2) == DOWN_BLOCKS;
    const int row = inject_block ? warp : item_ct1.get_group(2) * WARPS + warp;
    if (inject_block && (a.w_inject == nullptr || warp >= HC)) return;
    const uint16_t* wrow = (inject_block ? a.w_inject : a.w_down) + (size_t) row * D;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(wrow);
    // SYCL port: the multi kernel's order - per-tile partial sums for the split rows (gr_parity: bit-identical),
    // the inject rows still one accumulation over the whole row (their block is not split there)
    float acc = 0.0f;
    if (inject_block) {
#pragma unroll 4
        for (int j = lane; j < D / 8; j += 32) acc += dot8(*(w4 + j), xn + j * 8);
        acc = warp_sum(acc);
    } else {
        for (int sp = 0; sp < DOWN_SPLIT; ++sp) {
            float part = 0.0f;
#pragma unroll 5
            for (int j = lane + sp * (TILE / 8); j < (sp + 1) * (TILE / 8); j += 32) part += dot8(*(w4 + j), xn + j * 8);
            acc += warp_sum(part);
        }
    }
    if (lane != 0) return;
    if (inject_block) {
        a.inject_out[row] = acc;
    } else {
        const float x = acc / (float) HC;
        a.lo[row] = x / (1.0f + sycl::native::exp(-x));
    }
}

__dpct_inline__ void gr_up_kernel(FusedGrArgs a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &lo = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[LR]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[HC][UP_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int d0 = item_ct1.get_group(2) * UP_COLS;
#pragma unroll
    for (int k = t; k < LR; k += THREADS) lo[k] = a.lo[k];
    item_ct1.barrier();
    // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
    for (int r = warp; r < HC * UP_COLS; r += WARPS) {
        const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
        const sycl::uint4 *w4 =
            reinterpret_cast<const sycl::uint4 *>(a.w_up + (size_t)i * LR);
        float acc = dot8(*(w4 + lane), lo + lane * 8);
        if (lane < LR / 8 - 32) acc +=
            dot8(*(w4 + 32 + lane), lo + (32 + lane) * 8);
        acc = warp_sum(acc);
        if (lane == 0) {
            float rv = a.R[i];
            if (a.apply) {
                rv = sycl::fma((float)(a.bo_prev[d0 + dd]),
                               2.0f * sigmoidf_(a.inj_prev[c] / (float)HC), rv);
                a.R_out[i] = rv;                       // this block owns column d0+dd of every stream
            }
            const float x = rv * a.w_norm[i] * a.rs[c];
            g[c][dd] = x * sigmoidf_(acc);
        }
    }
    item_ct1.barrier();
    if (t < UP_COLS) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[c][t];
        a.mixed[d0 + t] = s / (float) HC;
    }
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================

// Step 1 of `gr_down_kernel`, one block per token, same threads and reduction order: rs[t] and xn[t] to global.
__dpct_inline__ void gr_norm_multi_kernel(GrMulti m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS][HC]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_rs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[HC]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const FusedGrArgs &a = m.a[item_ct1.get_group(2)];
    float *xn = m.xn + (size_t)item_ct1.get_group(2) * D;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), gw[c], r.x());
                r.y() = sycl::fma((float)(b.y()), gw[c], r.y());
            r.z() = sycl::fma((float)(b.z()), gw[c], r.z());
                r.w() = sycl::fma((float)(b.w()), gw[c], r.w());
        }
        const sycl::float4 g =
            *reinterpret_cast<const sycl::float4 *>(a.w_norm + i);
        float sq =
            r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<sycl::float4 *>(xn + i) = sycl::float4(
            r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    item_ct1.barrier();
    if (t < HC) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
        a.rs[t] = s_rs[t];
    }
    item_ct1.barrier();
#pragma unroll
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
}


// SYCL port: gr_norm_multi_kernel with one work-group per (token, stream) instead of per token: T groups left the
// B70 nearly idle (16 us a call). rs is per stream, so a group owns its stream outright. Thread t visits the same
// elements of stream c in the same order as in the per-token kernel (i = 4t + 1024k), and the warp and cross-warp
// sums run in the same order, so rs and xn are bitwise the old kernel's. The scaled values stay in registers
// (at most 3 float4 per thread) instead of a second pass over global memory.
__dpct_inline__ void gr_norm_split_port_kernel(GrMulti m) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto& part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS]>(item.get_group());
    auto& s_rs = *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(item.get_group());
    const int tok = (int) item.get_group(2) / HC, c = (int) item.get_group(2) % HC;
    const FusedGrArgs& a = m.a[tok];
    float* xn = m.xn + (size_t) tok * D;
    const int t = (int) item.get_local_id(2), lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    constexpr int MAXK = (N + THREADS * 4 - 1) / (THREADS * 4) + 1;   // 3: float4s of one stream per thread
    sycl::float4 keep[MAXK];
    int kept = 0;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;
        const int d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4*>(a.R + i);
        if (a.apply) {
            const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(a.bo_prev + d);
            r.x() = sycl::fma((float) (b.x()), gw, r.x());
            r.y() = sycl::fma((float) (b.y()), gw, r.y());
            r.z() = sycl::fma((float) (b.z()), gw, r.z());
            r.w() = sycl::fma((float) (b.w()), gw, r.w());
        }
        const sycl::float4 g = *reinterpret_cast<const sycl::float4*>(a.w_norm + i);
        ss += r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
        if (kept < MAXK) keep[kept++] = sycl::float4(r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    item.barrier(sycl::access::fence_space::local_space);
    if (t == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) sum += part[w];
        s_rs = sycl::rsqrt(sum / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    item.barrier(sycl::access::fence_space::local_space);
    const float rs = s_rs;
    int k = 0;
    for (int i = t * 4; i < D && k < kept; i += THREADS * 4) {
        if (i / N != c) continue;
        const sycl::float4 x = keep[k++];
        *reinterpret_cast<sycl::float4*>(xn + i) = sycl::float4(x.x() * rs, x.y() * rs, x.z() * rs, x.w() * rs);
    }
}

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
//
// TILEV = xn floats per token staged at a time.  The tile only changes the staging granularity: the lane's chunk
// order (lane + 32*q within the tile, tiles ascending) is strictly increasing for either value, so the results are
// bitwise identical to `gr_down_kernel` for both.  2560 stages 320 chunks of 8 per tile (10 per lane); cards whose
// opt-in below 8 * 2560 * 4 B slices the tokens - sm_75 (64 KiB) carries 6 tokens of it - run TILEV 1280 instead,
// which fits all eight tokens in one 40 KiB launch and stages 160 chunks of 8 (5 per lane, half the registers held
// for the weight prefetch); smaller tiles raise how many blocks share an SM (the 41-block grid), e.g. three blocks
// of four tokens instead of one on sm_75.
template <int TILEV, int MAX_T = kFusedGrMaxT>
__dpct_inline__ void gr_down_multi_kernel(GrMulti m, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int TQ = TILEV / 8 / 32; // uint4 weight chunks per lane per tile
    auto tile = (float *)dpct_local;   // [T][TILEV]
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int gid = (int) item_ct1.get_group(2);
    const bool inject_block = gid == DOWN_BLOCKS * DOWN_SPLIT;
    const int split = inject_block ? 0 : gid % DOWN_SPLIT, rb = inject_block ? 0 : gid / DOWN_SPLIT;
    const int row = inject_block ? warp : rb * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(wrow);
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    for (int base = inject_block ? 0 : split * TILEV; base < D; base += inject_block ? TILEV : TILEV * DOWN_SPLIT) {
        sycl::uint4 wv[TQ];
        if (active) {
#pragma unroll
            for (int q = 0; q < TQ; ++q)
                wv[q] = *(w4 + base / 8 + lane + 32 * q);
        }
        item_ct1.barrier(); // the previous tile is consumed
        const sycl::float4 *src4 = reinterpret_cast<const sycl::float4 *>(m.xn);
        sycl::float4 *tile4 = reinterpret_cast<sycl::float4 *>(tile);
        for (int i = t; i < T * (TILEV / 4); i += THREADS) {
            const int k = i / (TILEV / 4), off = i - k * (TILEV / 4);
            tile4[i] = src4[((size_t) k * D + base) / 4 + off];
        }
        item_ct1.barrier();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < MAX_T; ++k)
                if (k < T) acc[k] += dot8(wv[q], tile + k * TILEV + j * 8);
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            m.part[((size_t) split * kFusedGrMaxT + k) * LR + row] = s[k];   // the up kernel sums the splits
        }
    }
}

// SYCL port: gr_down_multi_kernel without the shared-memory tile. The tiled kernel stages T x 2560 floats per
// work-group (60 KB at T = 6) for 8 rows (40 KB of weights): the copy outweighs the weights and the SLM caps the
// B70 at ~2 groups per Xe core (unitrace: 64 us per call, ~100 GB/s). Here each lane reads its 8 xn floats per
// token straight from global memory (T x 40 KB in all, L1/L2-resident, shared by every group). Same per-lane
// order of the sums, so the outputs are bitwise the tiled kernel's.
__dpct_inline__ void gr_down_multi_direct_kernel(GrMulti m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int gid = (int) item_ct1.get_group(2);
    const bool inject_block = gid == DOWN_BLOCKS * DOWN_SPLIT;
    const int split = inject_block ? 0 : gid % DOWN_SPLIT, rb = inject_block ? 0 : gid / DOWN_SPLIT;
    const int row = inject_block ? warp : rb * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    if (!active) return;
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) row * D;
    const sycl::uint4* w4 = reinterpret_cast<const sycl::uint4*>(wrow);
    float acc[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
    for (int base = inject_block ? 0 : split * TILE; base < D; base += inject_block ? TILE : TILE * DOWN_SPLIT) {
        sycl::uint4 wv[TQ];
#pragma unroll
        for (int q = 0; q < TQ; ++q) wv[q] = *(w4 + base / 8 + lane + 32 * q);
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k)
                if (k < T) {
                    const sycl::float4* x4 = reinterpret_cast<const sycl::float4*>(m.xn + (size_t) k * D + base + j * 8);
                    const sycl::float4 a0 = x4[0], a1 = x4[1];
                    const float xv[8] = {a0.x(), a0.y(), a0.z(), a0.w(), a1.x(), a1.y(), a1.z(), a1.w()};
                    acc[k] += dot8(wv[q], xv);
                }
        }
    }
    float s[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) m.a[k].inject_out[row] = s[k];
        else m.part[((size_t) split * kFusedGrMaxT + k) * LR + row] = s[k];
    }
}
// SYCL port: the down projection sliced by columns. The direct kernel above has each warp read T x 32 bytes of
// activations (L2) per 16 bytes of weights - 12x the weight traffic at 6 tokens, ~134 GB/s. Here work-group s owns
// columns [128 s, 128 s + 128) of all 324 rows (320 down + 4 inject): it stages that slice of xn for the T tokens
// once (<= 4 KB of SLM) and streams the rows through it, 4 lanes per row (one 64-byte line per step), reducing over
// those 4 lanes once per row. gr_down_reduce then sums the 80 slice partials in a fixed order.
__dpct_inline__ void gr_down_sliced_kernel(GrMulti m, float* tile) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = (int) item.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T, sl = (int) item.get_group(2), c0 = sl * GRS_COLS;
    for (int i = t; i < T * GRS_COLS; i += THREADS) {
        const int k = i / GRS_COLS, c = i - k * GRS_COLS;
        tile[i] = m.xn[(size_t) k * D + c0 + c];
    }
    item.barrier(sycl::access::fence_space::local_space);
    const int rsub = lane >> 2, q = lane & 3;
    const bool inj = m.a[0].w_inject != nullptr;
    for (int r0 = 0; r0 < GRS_ROWS; r0 += WARPS * 8) {
        const int r = r0 + warp * 8 + rsub;
        const bool ok = r < LR || (r < GRS_ROWS && inj);
        const uint16_t* wrow = r < LR ? m.a[0].w_down + (size_t) r * D : m.a[0].w_inject + (size_t) (ok ? r - LR : 0) * D;
        sycl::uint4 wv[4];
#pragma unroll
        for (int cc = 0; cc < 4; ++cc)
            wv[cc] = ok ? reinterpret_cast<const sycl::uint4*>(wrow + c0)[q + 4 * cc] : sycl::uint4(0, 0, 0, 0);
        float acc[kFusedGrMaxT];
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            acc[k] = 0.0f;
            if (k < T) {
#pragma unroll
                for (int cc = 0; cc < 4; ++cc) acc[k] += dot8(wv[cc], tile + k * GRS_COLS + (q + 4 * cc) * 8);
            }
        }
        auto sg = item.get_sub_group();
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float v = acc[k];
            v += sycl::permute_group_by_xor(sg, v, 1);
            v += sycl::permute_group_by_xor(sg, v, 2);
            if (q == 0 && ok) m.part2[((size_t) sl * kFusedGrMaxT + k) * GRS_ROWS + r] = v;
        }
    }
}
__dpct_inline__ void gr_down_reduce_kernel(GrMulti m) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = (int) item.get_global_id(2);
    if (i >= m.T * GRS_ROWS) return;
    const int k = i / GRS_ROWS, r = i - k * GRS_ROWS;
    if (r >= LR && m.a[0].w_inject == nullptr) return;
    float x = 0.0f;
    for (int sl = 0; sl < GRS_SLICES; ++sl) x += m.part2[((size_t) sl * kFusedGrMaxT + k) * GRS_ROWS + r];
    if (r < LR) m.part[(size_t) k * LR + r] = x;   // split 0
    else m.a[k].inject_out[r - LR] = x;
}

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
__dpct_inline__ void gr_up_multi_kernel(GrMulti m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &lo = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[kFusedGrMaxT][LR]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kFusedGrMaxT][HC][UPM_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int d0 = item_ct1.get_group(2) * UPM_COLS;
#pragma unroll
    for (int i = t; i < T * LR; i += THREADS) {   // the down kernel's partial sums, in split order, then silu
        const int k = i / LR, r = i % LR;
        float x = 0.0f;
#pragma unroll
        for (int sp = 0; sp < DOWN_SPLIT; ++sp) if (sp < m.nsplit) x += m.part[((size_t) sp * kFusedGrMaxT + k) * LR + r];
        x /= (float) HC;
        const float v = x / (1.0f + sycl::native::exp(-x));
        lo[k][r] = v;
        m.a[k].lo[r] = v;
    }
    item_ct1.barrier();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const sycl::uint4 *w4 =
            reinterpret_cast<const sycl::uint4 *>(m.a[0].w_up + (size_t)i * LR);
        const sycl::uint4 wa = *(w4 + lane);
        const sycl::uint4 wb =
            lane < LR / 8 - 32 ? *(w4 + 32 + lane) : sycl::uint4(0, 0, 0, 0);
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float acc = dot8(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float)HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    item_ct1.barrier();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
}


// ================================ hc read v3 (opt-in: STRATA_GR_V3=1) - two kernels, stream-split ================
// The norm kernel runs on only T blocks (~16 us of pure latency per call) and `down` on 41 blocks (~280 GB/s).
// v3: `down` is split over (row group, stream[, column half]) = 164 or 328 blocks; each block stages its slice of
// R' * w_norm for the T tokens, reduces that slice's sum of squares itself, and writes UNSCALED partial dots.  The
// rms scale is per stream, so  w_down . xn = sum_c rs[c] * (w_down[:, c] . (R'[c] * w_norm[c]))  - `up` applies it
// in its prologue (lo, inject, rs).  Same maths, ANOTHER SUMMATION ORDER: not bitwise the default kernels, hence
// opt-in.  Dynamic shared memory is T * (N / S) floats: S (1 or 2 column halves) is the smallest that fits the
// card's opt-in limit at kFusedGrMaxT tokens (Ampere 99 KB: S = 1; Turing / HIP 64 KB: S = 2); a card where
// neither fits keeps the default kernels.
constexpr int TQ3 = N / 8 / 32;                    // uint4 weight chunks per lane in one stream's slice (10)

template <int S>
__dpct_inline__ void gr_down_v3_kernel(GrMulti m, float *__restrict__ part,
                                       float *__restrict__ ssg,
                                       uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto xs = (float *)dpct_local;                 // [T][N / S]
    constexpr int R2 = 1;                          // down rows per warp
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS][kFusedGrMaxT]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    // S = 2: each stream's 2560 columns in two halves (blockIdx.y = stream * S + half): twice the blocks
    constexpr int SL = N / S, TQS = SL / 8 / 32;
    const int rg = item_ct1.get_group(2), c = item_ct1.get_group(1) / S,
              h = item_ct1.get_group(1) - (item_ct1.get_group(1) / S) * S;
    constexpr int NDB = LR / (WARPS * R2);          // down row blocks per stream; block NDB = the inject rows
    const bool inject_block = rg == NDB;
    // warp w owns rows row0 + w * R2 + r (r < R2); the inject block: warps 0-3, one row each
    const int row0 = inject_block ? warp : (rg * WARPS + warp) * R2;
    const int nrows = inject_block ? ((m.a[0].w_inject != nullptr && warp < HC) ? 1 : 0) : R2;
    const bool active = nrows > 0;
    const uint16_t* wbase = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    sycl::uint4 wv[R2][TQS];
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(
            wbase + (size_t)(row0 + r) * D + (size_t)c * N + (size_t)h * SL);
#pragma unroll
        for (int q = 0; q < TQS; ++q) wv[r][q] = *(w4 + lane + 32 * q);
    }
    float ssp[kFusedGrMaxT];
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        ssp[k] = 0.0f;
        if (k >= T) continue;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        const sycl::float4 *R4 = reinterpret_cast<const sycl::float4 *>(
            a.R + (size_t)c * N + (size_t)h * SL);
        const sycl::float4 *G4 = reinterpret_cast<const sycl::float4 *>(
            a.w_norm + (size_t)c * N + (size_t)h * SL);
        const sycl::float4 *B4 =
            reinterpret_cast<const sycl::float4 *>(a.bo_prev + (size_t)h * SL);
        sycl::float4 *X4 =
            reinterpret_cast<sycl::float4 *>(xs + (size_t)k * SL);
        for (int i = t; i < SL / 4; i += THREADS) {
            sycl::float4 r = R4[i];
            if (a.apply) {
                const sycl::float4 b = B4[i];
                r.x() = sycl::fma((float)(b.x()), (float)gw, r.x());
                    r.y() = sycl::fma((float)(b.y()), (float)gw, r.y());
                r.z() = sycl::fma((float)(b.z()), (float)gw, r.z());
                    r.w() = sycl::fma((float)(b.w()), (float)gw, r.w());
            }
            const sycl::float4 g = G4[i];
            ssp[k] +=
                r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
            X4[i] = sycl::float4(r.x() * g.x(), r.y() * g.y(), r.z() * g.z(),
                                 r.w() * g.w());
        }
    }
#pragma unroll
    for (int k = 0; k < kFusedGrMaxT; ++k) {
        if (k >= T) break;
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    item_ct1.barrier();
    if (rg == 0 && t < T) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) sum += red[w][t];
        ssg[(t * HC + c) * S + h] = sum;
    }
    if (!active) return;
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        float acc[kFusedGrMaxT];
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int q = 0; q < TQS; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k)
                if (k < T) acc[k] += dot8(wv[r][q], xs + (size_t) k * SL + j * 8);
        }
        const int prow = inject_block ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[(((size_t) k * HC + c) * S + h) * PR + prow] = v;
        }
    }
}

template <int S>
__dpct_inline__ void gr_up_v3_kernel(GrMulti m, const float *__restrict__ part,
                                     const float *__restrict__ ssg) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &lo = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[kFusedGrMaxT][LR]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rsS = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kFusedGrMaxT][HC]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kFusedGrMaxT][HC][UPM_COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int d0 = item_ct1.get_group(2) * UPM_COLS;
    constexpr int RPW = HC * UPM_COLS / WARPS;     // 8 rows per warp
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < S; ++h) ss += ssg[t * S + h];
        const float r = sycl::rsqrt(ss / (float)N + m.a[k].eps);
        rsS[k][c] = r;
        if (item_ct1.get_group(2) == 0) m.a[k].rs[c] = r;
    }
    item_ct1.barrier();
    for (int i = t; i < T * LR; i += THREADS) {
        const int k = i / LR, r = i - k * LR;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + r];
            sum = sycl::fma(rsS[k][c], p, sum);
        }
        const float x = sum / (float) HC;
        lo[k][r] = x / (1.0f + sycl::native::exp(-x));
    }
    if (item_ct1.get_group(2) == 0 && t < T * HC) {
        const int k = t / HC, cc = t - k * HC;
        if (m.a[k].w_inject != nullptr) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                float p = 0.0f;
#pragma unroll
                for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + LR + cc];
                sum = sycl::fma(rsS[k][c], p, sum);
            }
            m.a[k].inject_out[cc] = sum;
        }
    }
    item_ct1.barrier();
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int r = warp + q * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const sycl::uint4 *w4 =
            reinterpret_cast<const sycl::uint4 *>(m.a[0].w_up + (size_t)i * LR);
        const sycl::uint4 wa = *(w4 + lane);
        const sycl::uint4 wb =
            lane < LR / 8 - 32 ? *(w4 + 32 + lane) : sycl::uint4(0, 0, 0, 0);
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float acc = dot8(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float)HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    item_ct1.barrier();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
}
// ================================ #315: split / staged - the same arithmetic, split differently ==================
// The multi read's variants beside the plain read (0.1.31's default; not main's opt-in STRATA_GR_V3 read, which sums
// in another order).  Every variant computes each output with exactly the plain read's operations in its order (so
// bitwise the plain read and the single-token read); only who does what, and when, changes.  `fused_gr_check`
// compares them on the card before a verify window uses them (STRATA_HC_SPLIT below).
//  - the norm (split and staged): one block per token AND stream instead of one per token.  Each thread visits
//    exactly the elements, in the order, it visited for that stream in the plain read, so every sum of squares and rs
//    is the same; then it recomputes its R' * w_norm and writes it times rs (the plain read stored the product and
//    scaled it in place: the same two roundings).  On an RTX 4080 SUPER, the plain read's norm took 0.58 ms per
//    4-token round (96 launches of 1-4 blocks), this one 0.23 ms.
//  - the down projection, split: the plain read's kernel.
//  - the down projection, staged: the plain read's 8 rows per block and lane order, but the activations arrive by
//    cp.async in half-stream tiles, two in flight, so no thread waits on a chain of loads, and they sit in shared
//    memory as two planes (floats 0-3 and 4-7 of every 8), so a warp's reads hit no bank twice.  The next tile's
//    weights are loaded while the current one is used.  A tile is 160 = 5 x 32 chunks of 8, so a lane still
//    accumulates its chunks lane + 32 q in ascending order, as in the plain read with either tile.  Before sm_80
//    (and on HIP) the staging is a plain copy: the same bits, only not asynchronous.

__dpct_inline__ void gr_norm_split_kernel(GrMulti m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_rs = *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const FusedGrArgs &a = m.a[item_ct1.get_group(2)];
    const int c = item_ct1.get_group(1);
    float *xn = m.xn + (size_t)item_ct1.get_group(2) * D;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;                       // another stream's element: another block's
        const int d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), (float)gw, r.x());
                r.y() = sycl::fma((float)(b.y()), (float)gw, r.y());
            r.z() = sycl::fma((float)(b.z()), (float)gw, r.z());
                r.w() = sycl::fma((float)(b.w()), (float)gw, r.w());
        }
        const float sq =
            r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
        ss += sq;
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    item_ct1.barrier();
    if (t == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = sycl::rsqrt(s / (float)N + a.eps);
        a.rs[c] = s_rs;
    }
    item_ct1.barrier();
    const float rs = s_rs;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;
        const int d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4 *>(a.R + i);
        if (a.apply) {
            const sycl::float4 b =
                *reinterpret_cast<const sycl::float4 *>(a.bo_prev + d);
            r.x() = sycl::fma((float)(b.x()), (float)gw, r.x());
                r.y() = sycl::fma((float)(b.y()), (float)gw, r.y());
            r.z() = sycl::fma((float)(b.z()), (float)gw, r.z());
                r.w() = sycl::fma((float)(b.w()), (float)gw, r.w());
        }
        const sycl::float4 g =
            *reinterpret_cast<const sycl::float4 *>(a.w_norm + i);
        const float px = r.x() * g.x(), py = r.y() * g.y(), pz = r.z() * g.z(),
                    pw = r.w() * g.w();
        *reinterpret_cast<sycl::float4 *>(xn + i) =
            sycl::float4(px * rs, py * rs, pz * rs, pw * rs);
    }
}

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800 &&      \
    !defined(__HIPCC__)
#define STRATA_GR_CP_ASYNC 1
#endif
__dpct_inline__ void cp_async16(void *smem, const void *gmem) {
#if defined(STRATA_GR_CP_ASYNC)
    auto sa = smem;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa),
                 "l"(gmem)
                 : "memory");
#else
    *(((uint32_t *)(uintptr_t)sa)) = *(((uint32_t *)(uintptr_t)gmem));
    if (16 > 4)
        *(((uint32_t *)(uintptr_t)sa) + 1) =
            *(((uint32_t *)(uintptr_t)gmem) + 1);
    if (16 > 8)
        *(((uint32_t *)(uintptr_t)sa) + 2) =
            *(((uint32_t *)(uintptr_t)gmem) + 2);
    if (16 > 12)
        *(((uint32_t *)(uintptr_t)sa) + 3) =
            *(((uint32_t *)(uintptr_t)gmem) + 3);
#endif
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__dpct_inline__ void cp_async_commit() {
#if defined(STRATA_GR_CP_ASYNC)
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#else

#endif
#endif
}
__dpct_inline__ void cp_async_wait1() {
#if defined(STRATA_GR_CP_ASYNC)
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#else

#endif
#endif
}
__dpct_inline__ void cp_async_wait0() {
#if defined(STRATA_GR_CP_ASYNC)
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#else

#endif
#endif
}

// `dot8` with its 8 activations as two float4: the same eight fmaf in the same order
__dpct_inline__ float dot8v(const sycl::uint4 w, const sycl::float4 x0,
                            const sycl::float4 x1) {
    float acc = 0.0f;
    acc = sycl::fma(sycl::bit_cast<float>(w.x() << 16), (float)(x0.x()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.x() & 0xffff0000u), (float)(x0.y()),
                    acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.y() << 16), (float)(x0.z()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.y() & 0xffff0000u), (float)(x0.w()),
                    acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.z() << 16), (float)(x1.x()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.z() & 0xffff0000u), (float)(x1.y()),
                    acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.w() << 16), (float)(x1.z()), acc);
    acc = sycl::fma(sycl::bit_cast<float>(w.w() & 0xffff0000u), (float)(x1.w()),
                    acc);
    return acc;
}

// Stage tile `h` of every token into `buf`: [T][2 planes][160 chunks] float4, plane 0 = floats 0-3 of a chunk.
__dpct_inline__ void stage_htile(const GrMulti &m, int T, int h,
                                 sycl::float4 *buf, int t) {
    for (int i = t; i < T * (H_TILE / 4); i += THREADS) {
        const int k = i / (H_TILE / 4), s4 = i - k * (H_TILE / 4);   // s4: float4 of the tile, chunk s4/2, half s4&1
        const float* src = m.xn + (size_t) k * D + (size_t) h * H_TILE + (size_t) s4 * 4;
        cp_async16(buf + (size_t) k * (H_TILE / 4) + (s4 & 1) * (H_TILE / 8) + (s4 >> 1), src);
    }
}

template <int MAX_T = kFusedGrMaxT>
__dpct_inline__ void gr_down_staged_kernel(GrMulti m, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto hbuf = (sycl::float4 *)dpct_local; // 2 buffers x [T][2][160] float4
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = item_ct1.get_group(2) == DOWN_BLOCKS;
    const int row = inject_block ? warp : item_ct1.get_group(2) * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);    // float4 per buffer (the second one follows the first)
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    sycl::uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) wv[q] = *(w4 + lane + 32 * q);
    }
    stage_htile(m, T, 0, hbuf, t);
    cp_async_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t);
    cp_async_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q)
                wnext[q] = *(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES) cp_async_wait1();         // tile h has landed (h + 1 may still be on its way)
        else cp_async_wait0();
        item_ct1.barrier();
        const sycl::float4 *cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (k < T) {
                        const sycl::float4 *pk = cur + (size_t)k * (H_TILE / 4);
                        acc[k] += dot8v(wv[q], pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        item_ct1.barrier(); // every warp is done with buffer h & 1
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t);
        cp_async_commit();                              // an empty group at the end keeps the wait counts simple
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + sycl::native::exp(-x));
        }
    }
}

}  // namespace

#include "fused_gr.kern.inc"

}  // namespace strata::kernels
