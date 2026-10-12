// src/kernels/cuda/qsa.host.cpp - the host side of qsa.dp.cpp: geometry checks, the step upload and the launchers.
#include "strata/core/emulate.hpp"
#include "strata/kernels/qsa.hpp"

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "qsa.kern.inc"

namespace {

constexpr int THREADS = 128;

void fail(const char* what) {
    std::fprintf(stderr, "qsa: %s\n", what);
    std::exit(1);
}

void check_launch(const char* what) {}

/// The SYNCHRONOUS entry points must check their own wait.  Without this the sticky error from a kernel that
/// faulted becomes visible only at the NEXT call's `cudaGetLastError()`, which then names the wrong kernel -
/// exactly the mis-attribution that made round 198's first `kv_append` failure look like a later one.
void check_sync(const char *what) {
    strata::ze::device_sync();
}

/// Geometry validation.  A silently wrong `head_dim % 4` would corrupt the gather's uint2 copy and a silently
/// wrong `n_head % n_head_kv` would produce a plausible attention with the wrong key, so both are refused.
void validate(const QsaShapes& s, const char* who) {
    if (s.n_head <= 0 || s.n_head_kv <= 0 || s.head_dim <= 0 || s.idx_dim <= 0 || s.idx_n_head <= 0 ||
        s.idx_block < 2 || s.page_size < 1) {
        std::fprintf(stderr, "qsa: %s: geometry is not set up\n", who);
        std::exit(1);
    }
    if (s.n_head % s.n_head_kv != 0) {
        std::fprintf(stderr, "qsa: %s: n_head %lld is not a multiple of n_head_kv %lld\n", who,
                     (long long) s.n_head, (long long) s.n_head_kv);
        std::exit(1);
    }
    if (s.head_dim % 4 != 0) {
        std::fprintf(stderr, "qsa: %s: head_dim %lld must be a multiple of 4 (the gather copies uint2)\n", who,
                     (long long) s.head_dim);
        std::exit(1);
    }
    if (s.n_rot <= 0 || s.n_rot % 2 != 0 || s.n_rot > s.head_dim || s.n_rot > s.idx_dim) {
        std::fprintf(stderr, "qsa: %s: n_rot %lld must be even and <= head_dim %lld and idx_dim %lld\n", who,
                     (long long) s.n_rot, (long long) s.head_dim, (long long) s.idx_dim);
        std::exit(1);
    }
    if (s.idx_n_head > 32) {
        std::fprintf(stderr, "qsa: %s: idx_n_head %lld > 32 (one warp per indexer head)\n", who,
                     (long long) s.idx_n_head);
        std::exit(1);
    }
}

constexpr int TOPK_THREADS = 256;

inline unsigned grid_for(long long n, int threads) {
    return (unsigned) ((n + threads - 1) / threads);
}

}  // namespace

/// THE STEP STATE, UPLOADED.  Every kernel below reads its per-token counts from here rather than from a kernel
/// argument, because a CUDA graph bakes arguments in at capture time - see the long note in `qsa.hpp`.  This
/// module-level buffer is what lets the OLD host-scalar signatures keep working unchanged while the kernels
/// already read the device values; the layer will pass its own buffer once it captures.
///
/// It is per-PROCESS and reused, which is safe because the upload and the kernel that reads it are on the same
/// stream and the host is the only writer.  `cudaMemcpyAsync` on the null stream is ordered against the null
/// stream's kernels, which is what every direct-launch caller here uses.
namespace {

int32_t *step_scratch() {
    static int32_t* d_step = nullptr;
    if (d_step == nullptr) {
        d_step = (int32_t *)strata::ze::malloc_device(qsa_step_bytes());
        if (d_step == nullptr) {
            std::fprintf(stderr, "qsa: step upload: cudaMalloc failed\n");
            std::exit(1);
        }
    }
    return d_step;
}

void step_upload_raw(const int32_t *h_step) {
    int32_t* d = step_scratch();
    strata::ze::default_stream().memcpy(d, h_step, qsa_step_bytes());
    strata::ze::default_stream().sync();
}

/// From a POSITION: fills all four entries.  This is the form the layer will use, and the only form that cannot
/// produce an inconsistent `n_kv`/`n_bid`/`width`.
const int32_t* step_upload(int64_t pos, int64_t n_kv_hint, const QsaShapes& s) {
    int32_t h[kStepCount];
    qsa_step_fill(h, pos, s);
    // A caller that also knows `n_kv` is CHECKED against it rather than believed: a disagreement is exactly the
    // inconsistency `qsa_step_fill` exists to make impossible, so it is worth an exit rather than a silent
    // preference for one of the two.
    if (n_kv_hint >= 0 && n_kv_hint != (int64_t) h[kStepNKv]) {
        std::fprintf(stderr, "qsa: step upload: n_kv %lld disagrees with pos+1 = %d\n", (long long) n_kv_hint,
                     h[kStepNKv]);
        std::exit(1);
    }
    step_upload_raw(h);
    return step_scratch();
}

/// From a WIDTH.  `kv_gather` and `qsa_attend` are handed the selection width directly and never see `pos`, so
/// only that entry is meaningful for them; the others are set consistently anyway rather than left stale,
/// because a stale entry is a value the NEXT kernel might read.
const int32_t* step_upload_width(int64_t width, const QsaShapes& s) {
    int32_t h[kStepCount];
    for (int i = 0; i < kStepCount; ++i) h[i] = 0;
    h[kStepWidth] = (int32_t) width;
    (void) s;
    step_upload_raw(h);
    return step_scratch();
}

}  // namespace

// ================= THE CAPTURABLE ENTRY POINTS =================
//
// These are the REAL implementations: every launch below is sized from a constant capacity and every per-token
// count comes from `step`.  The host-scalar entry points further down are wrappers that fill a module-level step
// buffer and forward this token's counts as the capacities.

void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const QsaShapes& s, void* stream, const KvHostPools* host) {
    validate(s, "kv_append");
    if (step == nullptr) fail("kv_append: step is null");
    // The grid is the head x dim count, which is a CONSTANT - `kv_append` writes one cell.
    const long long n = s.n_head_kv * s.head_dim;
    k_qsa_kv_append(strata::ze::stream_of(stream), {grid_for(n, THREADS), 1, 1}, {THREADS, 1, 1}, 0, k_pool, v_pool,
                    page_table, step, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size,
                    host ? *host : KvHostPools{});
    check_launch("kv_append");
    if (stream == nullptr) check_sync("kv_append");
}

void qsa_index_step(const float* pooled, const float* q_idx, const float* bias, const QsaShapes& s,
                    const int32_t* step, int64_t max_blocks, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    if (step == nullptr) fail("qsa_index: step is null");
    if (max_blocks <= 0) fail("qsa_index: max_blocks must be positive");
    // THE GRID IS `max_blocks`, A CONSTANT - not this token's `n_bid + 1`.  The kernel returns for
    // `b > n_bid`, so the surplus blocks cost a launch and nothing else.
    const int threads = 32 * (int) s.idx_n_head;
    k_qsa_index(strata::ze::stream_of(stream), {(unsigned) max_blocks, 1, 1}, {(unsigned) threads, 1, 1}, 0, pooled, q_idx,
                bias, (int) s.idx_n_head, (int) s.idx_dim, s.idx_block, step, cell_scores);
    check_launch("qsa_index");
    if (stream == nullptr) check_sync("qsa_index");
}

void topk_512_step(const float* cell_scores, const QsaShapes& s, int64_t cap, const int32_t* step,
                   int32_t* ids, void* stream) {
    validate(s, "topk_512");
    if (step == nullptr) fail("topk_512: step is null");
    // `cap` is checked against the GEOMETRY'S largest possible width rather than this token's, because this
    // token's is a device value.  The two are the same guarantee: `width = min(n_kv, top_k + r - 1)` and
    // `n_kv <= kTopkMaxCells`, so this is the most the kernel can ever write.
    const int64_t width_max = qsa_selection_width(kTopkMaxCells, s);
    if (cap < width_max) {
        std::fprintf(stderr, "qsa: topk_512: cap %lld < the largest possible selection width %lld\n",
                     (long long) cap, (long long) width_max);
        std::exit(1);
    }
    // One block, so the launch is already constant; the kernel reads `n_kv` and `width` from `step`.
    k_qsa_topk(strata::ze::stream_of(stream), {1, 1, 1}, {TOPK_THREADS, 1, 1}, 0, cell_scores, step, ids);
    check_launch("topk_512");
    if (stream == nullptr) check_sync("topk_512");
}

void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                    const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                    uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather");
    if (step == nullptr) fail("kv_gather: step is null");
    if (max_ids <= 0) return;
    // THE GRID IS THE CAPACITY.  The kernel reads the real `n_ids` from `step` and its `if (i >= total) return;`
    // guard leaves the surplus threads idle.
    const long long total = max_ids * s.n_head_kv * (s.head_dim / 4);
    k_qsa_kv_gather(strata::ze::stream_of(stream), {grid_for(total, 256), 1, 1}, {256, 1, 1}, 0, k_pool, v_pool,
                    page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
                    v_scratch);
    check_launch("kv_gather");
    if (stream == nullptr) check_sync("kv_gather");
}

void qsa_attend_step(const float *q, const uint16_t *k_scratch,
                     const uint16_t *v_scratch, const int32_t *step,
                     int64_t max_ids, const QsaShapes &s, float *attn,
                     float *weights, void *stream) {
    validate(s, "qsa_attend");
    if (step == nullptr) fail("qsa_attend: step is null");
    if (max_ids <= 0) fail("qsa_attend: max_ids must be positive");
    // THE SHARED SIZE IS THE CAPACITY.  Shared memory is part of the LAUNCH CONFIGURATION, so sizing it from
    // this token's `n_ids` bakes the sequence length in: a captured layer would allocate room for token 1 and
    // then index past it at token 500.  The kernel reads the real count from `step`, so a larger allocation is
    // exactly what it wants.  The opt-in therefore happens once, for the capacity.
    const size_t smem = (size_t)(max_ids + 32) * sizeof(float);
    // per DEVICE: the opt-in is a device setting (a layer split runs this on two cards)
    static size_t s_configured_dev[64] = {};
    int cur_dev = 0;
    cur_dev = strata::ze::current_device();
    size_t& s_configured = s_configured_dev[(cur_dev >= 0 && cur_dev < 64) ? cur_dev : 0];
    if (smem > s_configured) {
        int dev = 0, max_shared = 0;
        dev = strata::ze::current_device();
        max_shared = (int) strata::ze::device_info(dev).max_slm;
        max_shared = strata::smem_optin_of(max_shared);
        if ((int) smem > max_shared) {
            std::fprintf(stderr, "qsa: qsa_attend: max_ids %lld needs %zu B of shared, over the %d B limit\n",
                         (long long) max_ids, smem, max_shared);
            std::exit(1);
        }
        s_configured = smem;
    }
    k_qsa_attend(strata::ze::stream_of(stream), {(unsigned) s.n_head, 1, 1}, {(unsigned) s.head_dim, 1, 1}, smem, q,
                 k_scratch, v_scratch, step, (int) s.n_head, (int) s.n_head_kv, (int) s.head_dim, attn, weights);
    check_launch("qsa_attend");
    if (stream == nullptr) check_sync("qsa_attend");
}

// ================= host entry points =================

void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes& s) {
    if (host_step == nullptr) return;
    if (pos < 0) fail("qsa_step_fill: pos < 0");
    // ONE PLACE COMPUTES ALL FOUR, which is the whole reason the step state is a BUFFER rather than four
    // separate parameters.  `n_kv`, `n_bid` and `width` are DERIVED from `pos`, so a caller filling them in
    // itself could fill them inconsistently - and `qsa_index`'s host guard against exactly that inconsistency
    // now has a single writer to check against instead of two independent arguments.
    //
    // This is the form the LAYER will use: every value a QSA kernel needs derives from one position, so one
    // H2D per token updates all of them and no entry point takes a per-token scalar.
    const int64_t n_kv = pos + 1;
    host_step[kStepPos] = (int32_t) pos;
    host_step[kStepNKv] = (int32_t) n_kv;
    host_step[kStepNBid] = (int32_t) (n_kv / s.idx_block);
    host_step[kStepWidth] = (int32_t) qsa_selection_width(n_kv, s);
}

void kv_append(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, int64_t pos,
               const float* kcur, const float* vcur, const QsaShapes& s, void* stream) {
    if (pos < 0) { validate(s, "kv_append"); fail("kv_append: pos < 0"); }
    kv_append_step(k_pool, v_pool, page_table, step_upload(pos, -1, s), kcur, vcur, s, stream);
}

void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base, const float* w_k_norm,
                        float eps, const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab,
                        const float* sin_tab, void* stream) {
    validate(s, "indexer_key_append");
    if (pos_dev == nullptr) fail("indexer_key_append: pos_dev is null");
    if (b.tail == nullptr || b.dead == nullptr || b.pooled == nullptr || b.block_pos == nullptr)
        fail("indexer_key_append: the indexer buffers are not all set (tail/dead/pooled/block_pos)");
    const int threads = (int) s.idx_dim;
    const size_t smem = (size_t)s.idx_dim * sizeof(double);
    // **ONE LAUNCH, NO HOST BRANCH ON THE POSITION.**  The completion rotation is inside the kernel now.  The
    // arguments here are identical for every token, which is the property a CUDA graph needs: this function can
    // be captured and replayed, and the position arrives through `pos_dev` like every other kernel's data.
    k_qsa_indexer_key_append(strata::ze::stream_of(stream), {1, 1, 1}, {(unsigned) threads, 1, 1}, smem, raw, pos_dev,
                             (int) pos_base, w_k_norm, eps, b.tail, b.dead, b.pooled, b.block_pos, (int) s.idx_dim,
                             (int) s.idx_block, (int) s.n_rot, cos_tab, sin_tab, mrope_table());
    check_launch("indexer_key_append");
    if (stream == nullptr) check_sync("indexer_key_append");
}

void qsa_index(const float* pooled, int64_t n_bid, const float* q_idx, const float* bias, const QsaShapes& s,
               int64_t n_kv, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    if (n_bid < 0 || n_kv <= 0) fail("qsa_index: n_bid < 0 or n_kv <= 0");
    // `n_kv`, `n_bid` and `width` are DERIVED from one position, so this refuses a caller whose two arguments
    // disagree rather than launching a kernel that would score the wrong number of blocks.
    if ((n_kv / s.idx_block) != n_bid) {
        std::fprintf(stderr, "qsa: qsa_index: n_bid %lld is not n_kv %lld / r %lld\n", (long long) n_bid,
                     (long long) n_kv, (long long) s.idx_block);
        std::exit(1);
    }
    // THIS TOKEN'S counts as the CAPACITIES: correct for a direct launch, WRONG in a graph.  The layer passes a
    // constant `max_blocks` from its state instead - see the note in `qsa.hpp`.
    qsa_index_step(pooled, q_idx, bias, s, step_upload(n_kv - 1, n_kv, s), n_bid + 1, cell_scores, stream);
}

void topk_512(const float* cell_scores, int64_t n_kv, const QsaShapes& s, int64_t cap, int32_t* ids,
              void* stream) {
    validate(s, "topk_512");
    if (n_kv <= 0) return;
    if (n_kv > kTopkMaxCells) {
        std::fprintf(stderr, "qsa: topk_512: n_kv %lld > kTopkMaxCells %lld (one block; phase 3 replaces this)\n",
                     (long long) n_kv, (long long) kTopkMaxCells);
        std::exit(1);
    }
    topk_512_step(cell_scores, s, cap, step_upload(n_kv - 1, n_kv, s), ids, stream);
}

void kv_gather(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, const int32_t* ids,
               int64_t n_ids, const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather");
    if (n_ids <= 0) return;
    kv_gather_step(k_pool, v_pool, page_table, ids, step_upload_width(n_ids, s), n_ids, s, k_scratch,
                   v_scratch, stream);
}

void qsa_attend(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch, int64_t n_ids,
                const QsaShapes& s, float* attn, float* weights, void* stream) {
    validate(s, "qsa_attend");
    if (n_ids < 0) fail("qsa_attend: n_ids < 0");
    if (n_ids == 0) {
        // The empty selection goes through the kernel's own zero path; `max_ids` must still be positive, so the
        // wrapper gives it the geometry's capacity rather than 0.
        qsa_attend_step(q, k_scratch, v_scratch, step_upload_width(0, s),
                        qsa_selection_width(kTopkMaxCells, s), s, attn, weights, stream);
        return;
    }
    qsa_attend_step(q, k_scratch, v_scratch, step_upload_width(n_ids, s), n_ids, s, attn, weights, stream);
}

void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out, void* stream) {
    validate(s, "qsa_gate_apply_f32");
    const long long n = s.n_head * s.head_dim;
    k_qsa_gate_apply_f32(strata::ze::stream_of(stream), {grid_for(n, 256), 1, 1}, {256, 1, 1}, 0, attn, q_full,
                         (int) s.n_head, (int) s.head_dim, out);
    check_launch("qsa_gate_apply_f32");
    if (stream == nullptr) check_sync("qsa_gate_apply_f32");
}

void qsa_gate_apply(const float* attn, const float* q_full, const QsaShapes& s, uint16_t* out, void* stream) {
    validate(s, "qsa_gate_apply");
    const long long n = s.n_head * s.head_dim;
    k_qsa_gate_apply(strata::ze::stream_of(stream), {grid_for(n, 256), 1, 1}, {256, 1, 1}, 0, attn, q_full,
                     (int) s.n_head, (int) s.head_dim, out);
    check_launch("qsa_gate_apply");
    if (stream == nullptr) check_sync("qsa_gate_apply");
}

}  // namespace strata::kernels
