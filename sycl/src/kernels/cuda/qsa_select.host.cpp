// src/kernels/cuda/qsa_select.host.cpp - the host side of qsa_select.dp.cpp: kernel choice and launches.
#include "strata/core/emulate.hpp"
#include <cstdlib>
#include <cstring>
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>

namespace strata::kernels {

#include "qsa_select.kern.inc"

namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

constexpr int TC_QT = 16;                 // queries per CTA (one m16 tile per indexer head)
constexpr int TC_NB = 32;                 // blocks per tile (4 warps x n8)
constexpr int TC_ITER = 4;                // tiles per CTA (the query tile is loaded once)
constexpr int TC_QS = IDX_HEADS * IDX_DIM + 4;   // query row stride in floats
constexpr int TC_KS = IDX_DIM + 4;               // key row stride

#if defined(__HIPCC__)
constexpr int WQT = 16;                    // queries per CTA (the N of one WMMA)
constexpr int WITER = 4;                   // key tiles per warp (the CTA covers 4 warps * WITER * 16 blocks)

// the WMMA kernel needs gfx12 (gfx1200/gfx1201) or gfx11 (gfx110x/gfx115x: the other fragment layout) code objects
bool sel_gfx12_device() {
    static int ok[64] = {};   // per device: 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        cudaDeviceProp prop;
        ok[dev] = (cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                   (std::strncmp(prop.gcnArchName, "gfx1200", 7) == 0 || std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0 ||
                    strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName)))
                      ? 1 : 2;
        cudaGetLastError();
    }
    return ok[dev] == 1;
}
#endif  // __HIPCC__

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

constexpr int MQ = 8;

constexpr int CL_MAXQ = 16;    // larger calls (prefill sub-batches) fill the GPU with one CTA per query already

}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    static const bool multi = [] { const char* v = std::getenv("STRATA_SCORES_MULTI"); return v == nullptr || std::atoi(v) != 0; }();
    if (multi && nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
        // STRATA_QSA_EARLY_EXIT=0 keeps the old staging (every CTA loads the queries first); the scores are the same bits
        static const bool early = [] { const char* v = std::getenv("STRATA_QSA_EARLY_EXIT"); return v == nullptr || std::atoi(v) != 0; }();
        k_qsa_select_block_scores_multi(strata::ze::stream_of(stream), {256, 1, 1}, {SCORE_WARPS * 32, 1, 1}, 0, pooled,
                                        dead, q_idx, steps, (int) nq, max_blocks, scores, early);
        return;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const strata::ze::Dim3 grid{(unsigned)((reach + SCORE_WARPS - 1) / SCORE_WARPS),
                                (unsigned)nq};
    k_qsa_select_block_scores(strata::ze::stream_of(stream), grid, {SCORE_WARPS * 32, 1, 1}, 0, pooled, dead, q_idx, steps,
                              max_blocks, scores);
}

bool qsa_block_scores_tc(const float *pooled, const float *dead,
                         const float *q_idx, const int32_t *steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes &s, float *scores,
                         void *stream, int64_t active_blocks) {
    if (nq <= 0) return true;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535 * TC_QT) return false;
#if defined(__HIPCC__)
    // AMD: the gfx12 (RDNA4) WMMA scorer, opt-in (STRATA_SELECT_WMMA=1): it selects slightly differently from the warp
    // kernel (254/256 queries the same), so the default keeps the warp kernel; every other target keeps it too (false)
    static const bool wmma_on = [] {
        const char* v = std::getenv("STRATA_SELECT_WMMA");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (!wmma_on || !sel_gfx12_device()) return false;
    {
        const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
        const int64_t per = (int64_t) 4 * WITER * 16;
        const dim3 grid((unsigned) ((reach + per - 1) / per), (unsigned) ((nq + WQT - 1) / WQT));
        block_scores_wmma_kernel<<<grid, 128, 0, (cudaStream_t) stream>>>(pooled, q_idx, steps, nq, max_blocks, reach, scores);
        block_scores_tail_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(dead, q_idx, steps, max_blocks, scores);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_tc: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return true;
    }
#else
    {   // sm_80 or newer (TF32 MMA); an older card keeps the warp kernel
        return false;   // SYCL: the tensor-core kernel is not ported yet; the caller takes the older kernel
    }
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs it on several)
    int adev = 0;
    adev = strata::ze::current_device();
    const int bytes = (TC_QT * TC_QS + TC_NB * TC_KS) * (int)sizeof(float);
    if (!attr[adev]) {
        if (0 != 0) {
            return false;
        }
        attr[adev] = true;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t per = (int64_t) TC_NB * TC_ITER;
    const strata::ze::Dim3 grid{(unsigned)((reach + per - 1) / per),
                                (unsigned)((nq + TC_QT - 1) / TC_QT)};
    k_qsa_select_block_scores_tc(strata::ze::stream_of(stream), grid, {128, 1, 1}, bytes, pooled, q_idx, steps, nq,
                                 max_blocks, reach, scores);
    k_qsa_select_block_scores_tail(strata::ze::stream_of(stream), {(unsigned) nq, 1, 1}, {32, 1, 1}, 0, dead, q_idx, steps,
                                   max_blocks, scores);

    return true;
#endif
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    k_qsa_select_block_topk(strata::ze::stream_of(stream), {(unsigned) nq, 1, 1}, {TOPK_T, 1, 1}, 0, scores, steps,
                            max_blocks, cap, ids);
}

#if !defined(__HIPCC__)
// Only Turing has a retained model measurement for this CUDA dispatch. Other CUDA devices keep the capacity rule
// (the RTX 5070 regression below). An Intel GPU is never Turing.
static bool topk_active_turing_device() {
    return false;
}
#endif

// SYCL port: no thread-block clusters (sm_90): the caller takes the plain top-k
bool qsa_block_topk_cluster(const float *scores, const int32_t *steps,
                            int64_t nq, int64_t max_blocks, int64_t cap,
                            const QsaShapes &s, int32_t *ids,
                            void *stream) {
    (void) scores; (void) steps; (void) nq; (void) max_blocks; (void) cap; (void) s; (void) ids; (void) stream;
    return false;
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    // keys in registers when every query's blocks fit (contexts up to ~135K cells), else (CUDA) the same threads
    // reading them from memory; the same ids. STRATA_TOPK_OLD=1: the original kernel
    static const bool old = std::getenv("STRATA_TOPK_OLD") != nullptr;
    if (nq <= 0) return;
#if !defined(__HIPCC__)
    // sm_90+: a cluster of CL_N CTAs per query for the calls of a few queries (decode windows); the same ids.
    // STRATA_QSA_CLUSTER=0: the one-CTA kernels below
    static const bool cluster = [] {
        const char* v = std::getenv("STRATA_QSA_CLUSTER");
        return !v || std::atoi(v) != 0;
    }();
    if (cluster && !old && nq <= CL_MAXQ && qsa_block_topk_cluster(scores, steps, nq, max_blocks, cap, s, ids, stream))
        return;
#endif
    // the blocks a query can have: the call's active count when the caller knows it (the prompt path), else the capacity.
    // Decode (no count) keeps the capacity rule and the original register width: nothing changes there.
#if defined(__HIPCC__)
    const bool counted = active_blocks > 0;
#else
    // Turing: --max-context 262144 makes the stride 65538, even while a 131K prompt's active blocks fit in
    // TK_T * TK_PER registers. Use the prefill bound on sm_75, keeping max_blocks as the score-row stride.
    // Other CUDA devices keep 0.1.32's capacity rule: #337 was measured on RDNA4, and RTX 5070 64K prompts were
    // 1-3% slower. Decode/captured graphs omit the bound and never query the device here.
    static const bool capacity_guard = std::getenv("STRATA_TOPK_CAPACITY_GUARD") != nullptr;
    // STRATA_TOPK_ACTIVE_ANY=1 (tests): the Turing dispatch on any CUDA card, so qsa_topk_active_parity checks it
    // on whatever card runs the tests (the kernels are the same on every architecture)
    static const bool any_card = [] { const char* v = std::getenv("STRATA_TOPK_ACTIVE_ANY"); return v && v[0] == '1'; }();
    const bool counted = !capacity_guard && active_blocks > 0 && active_blocks <= max_blocks &&
                         (any_card || topk_active_turing_device());
#endif
    const int64_t reach = counted && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t fit = (int64_t) TK_T * (counted ? TK_PER_MAX : TK_PER);
#if defined(__HIPCC__)
    constexpr int64_t kRegMinBlocks = 7168;   // gfx1201: below ~28K cells the 1,024-thread kernel's fixed cost loses to the ref
    const bool too_small = counted && reach < kRegMinBlocks;
    if (old || too_small || reach > fit) {
#else
    constexpr int64_t kWideMinBlocks = 4608;   // RTX 3060: a prompt batch below ~18K cells is faster on the ref
    const bool too_small = reach > fit && active_blocks > 0 && active_blocks < kWideMinBlocks;
    if (old || too_small) {
#endif
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
#if !defined(__HIPCC__)
    // Turing prefill (the `counted` bound) above ~90K cells (22,528 blocks): the wide kernel, which reads the keys
    // coalesced, instead of the register kernel's uncoalesced per-thread runs (PR #743: 131K, 1.03 -> 0.75 ms); the
    // same ids.  STRATA_TOPK_STREAM=0 restores the register kernel there.
    static const bool turing_wide = [] {
        const char* v = std::getenv("STRATA_TOPK_STREAM");
        return v == nullptr || v[0] != '0';
    }();
    if (reach > fit || (turing_wide && counted && reach > 22528))   // past the register kernel's reach, or Turing's band
    {
        k_qsa_select_block_topk_wide(strata::ze::stream_of(stream), {(unsigned) nq, 1, 1}, {TK_T, 1, 1}, 0, scores, steps,
                                     max_blocks, cap, ids);
    } else
#endif
        if (reach <= (int64_t)TK_T * TK_PER)
    {
        k_qsa_select_block_topk_reg(strata::ze::stream_of(stream), {(unsigned) nq, 1, 1}, {TK_T, 1, 1}, 0, scores, steps,
                                    max_blocks, cap, ids);
    } else
    {
        k_qsa_select_block_topk_reg(strata::ze::stream_of(stream), {(unsigned) nq, 1, 1}, {TK_T, 1, 1}, 0, scores, steps,
                                    max_blocks, cap, ids);
    }
}

}  // namespace strata::kernels