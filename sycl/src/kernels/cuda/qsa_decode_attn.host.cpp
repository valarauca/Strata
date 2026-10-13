// src/kernels/cuda/qsa_decode_attn.host.cpp - the host side of qsa_decode_attn.dp.cpp: the launchers.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace strata::kernels {

#include "qsa_decode_attn.kern.inc"

namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = 12;            // query heads per KV head (24 / 2)
constexpr int CHUNK = 64;        // cells per block
constexpr int THREADS = 256;

#if defined(STRATA_EXPERIMENTAL_SM60)
// the current device is below sm_75 (per device: a layer split can mix cards); STRATA_ATTN_PRE75=0 turns PR #540's
// kernel off (A/B)
bool pre75_attn() {
    static const bool off = [] {
        const char* e = std::getenv("STRATA_ATTN_PRE75");
        return e != nullptr && e[0] == '0';
    }();
    static int cc[64] = {};
    int dev = 0;
    if (off) return false;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (cc[dev] == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cc[dev] = 10 * major + minor;
    }
    return cc[dev] < 75;
}
#define STRATA_ATTN_CHUNK(M) (pre75_attn() ? attn_chunk_kernel_pre75<M> : attn_chunk_kernel<M>)
#else
#define STRATA_ATTN_CHUNK(M) attn_chunk_kernel<M>
#endif

}  // namespace

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const ze::Dim3 grid{(unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q};
    ze::Stream* st = ze::stream_of(stream);
    // S25: STRATA_ATTN_LANECELL=1 - the score phase one cell per thread (bit-identical scores, no shuffle reductions)
    static const bool lane_cell = [] { const char* v = std::getenv("STRATA_ATTN_LANECELL"); return v && v[0] == '1'; }();
#define STRATA_ATTN_LC(M)                                                                                          \
    k_qsa_decode_attn_chunk_lc_##M(st, grid, {THREADS, 1, 1}, 0, q, pools, ids, steps, (int) s.n_head_kv,        \
                                   (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
    if (lane_cell) {
        if (kv_mode == 3) { STRATA_ATTN_LC(3); }
        else if (kv_mode == 2) { STRATA_ATTN_LC(2); }
        else if (kv_mode == 1) { STRATA_ATTN_LC(1); }
        else { STRATA_ATTN_LC(0); }
    } else
#undef STRATA_ATTN_LC
    if (kv_mode == 3)
    {
        k_qsa_decode_attn_chunk_3(st, grid, {THREADS, 1, 1}, 0, q, pools, ids, steps, (int) s.n_head_kv,
                                  (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    } else if (kv_mode == 2)
    {
        k_qsa_decode_attn_chunk_2(st, grid, {THREADS, 1, 1}, 0, q, pools, ids, steps, (int) s.n_head_kv,
                                  (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    } else if (kv_mode == 1)
    {
        k_qsa_decode_attn_chunk_1(st, grid, {THREADS, 1, 1}, 0, q, pools, ids, steps, (int) s.n_head_kv,
                                  (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    } else {
        k_qsa_decode_attn_chunk_0(st, grid, {THREADS, 1, 1}, 0, q, pools, ids, steps, (int) s.n_head_kv,
                                  (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    }
    k_qsa_decode_attn_merge(st, {(unsigned) s.n_head, (unsigned) n_q, 1}, {HD, 1, 1}, 0, part_acc, part_m, part_l,
                            n_chunks, attn, stride);
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const ze::Dim3 grid{(unsigned) n_chunks, (unsigned) s.n_head_kv, 1};
    ze::Stream* st = ze::stream_of(stream);
    if (kv_mode == 3)
    {
        k_qsa_decode_attn_chunk_3(st, grid, {THREADS, 1, 1}, 0, q, pools, ids, step, (int) s.n_head_kv,
                                  (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    } else if (kv_mode == 2)
    {
        k_qsa_decode_attn_chunk_2(st, grid, {THREADS, 1, 1}, 0, q, pools, ids, step, (int) s.n_head_kv,
                                  (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    } else if (kv_mode == 1)
    {
        k_qsa_decode_attn_chunk_1(st, grid, {THREADS, 1, 1}, 0, q, pools, ids, step, (int) s.n_head_kv,
                                  (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    } else {
        k_qsa_decode_attn_chunk_0(st, grid, {THREADS, 1, 1}, 0, q, pools, ids, step, (int) s.n_head_kv,
                                  (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    }
    k_qsa_decode_attn_merge(st, {(unsigned) s.n_head, 1, 1}, {HD, 1, 1}, 0, part_acc, part_m, part_l, n_chunks, attn, 0);
}

}  // namespace strata::kernels
