// src/kernels/cuda/verify_kernels.host.cpp - the host side of verify_kernels.dp.cpp: the launchers.
#include "strata/sycl_doorbell.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/resident_plan_mirror.hpp"
#include "strata/ze/kernel_abi.hpp"
#include "verify_kernels_shared.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strata::kernels {

#include "verify_kernels.kern.inc"

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    k_verify_kernels_fetch_blobs(strata::ze::stream_of(stream), {48 * 8, 1, 1}, {256, 1, 1}, 0, src, n, dst,
                                 (long long) (blob_bytes / 16));
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    k_verify_kernels_rebase_ptrs(strata::ze::stream_of(stream), {1, 1, 1}, {128, 1, 1}, 0, ptr, n,
                                 (unsigned long long) base, (long long) blob_bytes);
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    k_verify_kernels_add_streams_broadcast(strata::ze::stream_of(stream),
                                           {(unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok, 1}, {256, 1, 1},
                                           0, h, e, R, n_embd, hc);
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    k_verify_kernels_ident_hits(strata::ze::stream_of(stream), {1, 1, 1}, {1024, 1, 1}, 0, ids, n, slot, dst, count);
}

void copy_row_to_first(const int32_t* row_dev, float* a, int64_t a_n, float* b, int64_t b_n, float* c, int64_t c_n,
                       void* stream) {
    k_verify_kernels_copy_row_to_first(strata::ze::stream_of(stream), {16, 1, 1}, {256, 1, 1}, 0, row_dev, a, a_n, b,
                                       b_n, c, c_n);
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    k_verify_kernels_mtp_select(strata::ze::stream_of(stream), {1, 1, 1}, {256, 1, 1}, 0, R_src, R_stride, ids, row_dev,
                                R_dst, tok_dst, out, j, probs, out_p);
}

void force_token(int32_t* tok, const int32_t* force, int j, void* stream) {
    k_verify_kernels_force_token(strata::ze::stream_of(stream), {1, 1, 1}, {1, 1, 1}, 0, tok, force, j);
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    strata::ze::Stream* s = strata::ze::stream_of(stream);
    if (row_bytes % 16 == 0)
        k_verify_kernels_gather_rows_16(s, {48 * 8, 1, 1}, {256, 1, 1}, 0, src, (long long) (row_bytes / 16), ids, n,
                                        dst);
    else if (row_bytes % 4 == 0)
        k_verify_kernels_gather_rows_4(s, {48 * 8, 1, 1}, {256, 1, 1}, 0, src, (long long) (row_bytes / 4), ids, n, dst);
    else
        k_verify_kernels_gather_rows_1(s, {48 * 8, 1, 1}, {256, 1, 1}, 0, src, row_bytes, ids, n, dst);
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    k_verify_kernels_map_ids(strata::ze::stream_of(stream), {1, 1, 1}, {64, 1, 1}, 0, ids, table, n);
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    k_verify_kernels_row_top_prob(strata::ze::stream_of(stream), {(unsigned) n_rows, 1, 1}, {1024, 1, 1}, 0, logits,
                                  n_vocab, ids, probs);
}

#if !defined(__HIPCC__)   // warp-32 kernels: an AMD card keeps the one-block row_top_prob and sampler_greedy_kernel
uint64_t argmax_rows_scratch_bytes(int n_rows) { return (uint64_t) n_rows * sizeof(ArgRow); }

void argmax_rows(const float* logits, int n_rows, int n, void* scratch, int32_t* out, void* stream) {
    if (n_rows <= 0) return;
    const int nb = std::min(kArgMaxBlocks, std::max(1, (n + 4095) / 4096));
    const int per_block = (n + nb - 1) / nb;
    k_verify_kernels_argmax_rows(strata::ze::stream_of(stream), {(unsigned) nb, (unsigned) n_rows, 1},
                                 {kArgThreads, 1, 1}, 0, logits, n, per_block, scratch, out);
}

uint64_t row_top_prob_scratch_bytes(int n_rows) { return (uint64_t) n_rows * sizeof(TopRow); }

void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* scratch,
                        void* stream) {
    if (n_rows <= 0) return;
    k_verify_kernels_row_top_prob_split(strata::ze::stream_of(stream), {kTopBlocks, (unsigned) n_rows, 1},
                                        {1024 / kTopBlocks, 1, 1}, 0, logits, n_vocab, ids, probs, scratch);
}
#else
uint64_t argmax_rows_scratch_bytes(int) { return 0; }
void argmax_rows(const float*, int, int, void*, int32_t*, void*) {
    std::fprintf(stderr, "argmax_rows: not built for this backend\n");
    std::exit(1);
}
uint64_t row_top_prob_scratch_bytes(int) { return 0; }
void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void*,
                        void* stream) {
    row_top_prob(logits, n_rows, n_vocab, ids, probs, stream);
}
#endif

bool multi_block_head_ops() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MULTI_BLOCK_ARGMAX");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
#endif
}

bool argmax_rows_wanted() {
    return false;
}

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    k_verify_kernels_window_ids(strata::ze::stream_of(stream), {8, (unsigned) n, 1}, {256, 1, 1}, 0, steps, window, ids,
                                (long long) ids_stride);
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    k_verify_kernels_dense_steps(strata::ze::stream_of(stream), {1, 1, 1}, {64, 1, 1}, 0, cells, n, steps);
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin, bool commit) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT ||
        (commit && (n_tok != 1 || t_begin != 0))) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    k_verify_kernels_gdn_conv_l2_multi(strata::ze::stream_of(stream), {(unsigned) (channels / S), (unsigned) n_tok, 1},
                                       {S, 1, 1}, 0, const_cast<float*>(history), qkv, conv_w, h, channels, qk_heads,
                                       eps, t_begin, commit);
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    k_verify_kernels_gdn_conv_commit(strata::ze::stream_of(stream), {(unsigned) ((channels + 255) / 256), 1, 1},
                                     {256, 1, 1}, 0, history, qkv, channels, n_keep);
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    const unsigned blocks = (unsigned) ((2 * h_v + 1) / 2);
    strata::ze::Stream* st = strata::ze::stream_of(stream);
    const strata::ze::Dim3 grid{blocks, 1, 1}, block{64, 1, 1};
    switch (n_tok) {
    case 1:
        k_verify_kernels_gdn_ab_multi_t1(st, grid, block, 0, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v,
                                         n_tok);
        break;
    case 2:
        k_verify_kernels_gdn_ab_multi_t2(st, grid, block, 0, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v,
                                         n_tok);
        break;
    case 3:
        k_verify_kernels_gdn_ab_multi_t3(st, grid, block, 0, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v,
                                         n_tok);
        break;
    case 4:
        k_verify_kernels_gdn_ab_multi_t4(st, grid, block, 0, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v,
                                         n_tok);
        break;
    case 5:
        k_verify_kernels_gdn_ab_multi_t5(st, grid, block, 0, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v,
                                         n_tok);
        break;
    case 6:
        k_verify_kernels_gdn_ab_multi_t6(st, grid, block, 0, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v,
                                         n_tok);
        break;
    default:
        k_verify_kernels_gdn_ab_multi_tmax(st, grid, block, 0, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v,
                                           n_tok);
        break;
    }
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin, void* xq_out) {
    void* const xq = xq_out;   // S26: non-null = also the outputs' q8_1 (STRATA_QFUSE)
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    strata::ze::Stream* st = strata::ze::stream_of(stream);
    static const bool split = [] { const char* v = std::getenv("STRATA_GDN_SPLIT"); return v && v[0] == '1'; }();
    // S26: the split runs where it is faster (gfx1151 harness, h_v 48: the commit 8-19% per call; a window of 2+ rows
    // 3-13%; one row: no gain, the old kernel stays) - bitwise equal in every case
    if (split && (t_out_begin >= n_tok || n_tok - t_out_begin >= 2)) {
        const strata::ze::Dim3 grid{(unsigned) h_v, S / GS_COLS, 1}, block{GS_COLS, RG, 1};
        if (t_out_begin >= n_tok) {   // no outputs (the commit): the state only
            k_verify_kernels_gdn_state_replay(st, grid, block, 0, state, h, conv_channels, gate, beta, h_k, h_v, n_tok,
                                              n_keep);
        } else {
            k_verify_kernels_gdn_step_split_out(st, grid, block, 0, state, h, conv_channels, gate, beta, y, h_k, h_v,
                                                n_tok, n_keep, t_out_begin);
            const strata::ze::Dim3 ng{(unsigned) h_v, (unsigned) (n_tok - t_out_begin), 1};
            if (xq)
                k_verify_kernels_gdn_out_norm_q(st, ng, {S, RG, 1}, 0, z, gamma, eps, y, h_v, n_tok, n_keep,
                                                t_out_begin, xq);
            else
                k_verify_kernels_gdn_out_norm(st, ng, {S, RG, 1}, 0, z, gamma, eps, y, h_v, n_tok, n_keep, t_out_begin,
                                              nullptr);
        }
        return;
    }
    static const bool commit_split = [] {
        const char* e = std::getenv("STRATA_GDN_COMMIT_SPLIT");
        return !e || e[0] != '0';
    }();
    if (commit_split && n_keep != nullptr && t_out_begin >= n_tok) {
        k_verify_kernels_gdn_step_commit(st, {(unsigned) h_v, 4u, 1}, {32, RG, 1}, 0, state, h, conv_channels, gate, beta,
                                         h_k, h_v, n_keep);
        return;
    }
    const strata::ze::Dim3 g1{(unsigned) h_v, 1, 1}, b1{S, RG, 1};
    const bool all_out = n_keep == nullptr && t_out_begin <= 0;
    if (xq && t_out_begin < n_tok) {
        if (all_out)
            k_verify_kernels_gdn_step_norm_multi_all_q(st, g1, b1, 0, state, h, conv_channels, gate, beta, z, gamma, eps,
                                                       y, h_k, h_v, n_tok, nullptr, 0, xq);
        else
            k_verify_kernels_gdn_step_norm_multi_keep_q(st, g1, b1, 0, state, h, conv_channels, gate, beta, z, gamma,
                                                        eps, y, h_k, h_v, n_tok, n_keep, t_out_begin, xq);
    } else if (all_out) {
        k_verify_kernels_gdn_step_norm_multi_all(st, g1, b1, 0, state, h, conv_channels, gate, beta, z, gamma, eps, y,
                                                 h_k, h_v, n_tok, nullptr, 0, nullptr);
    } else {
        k_verify_kernels_gdn_step_norm_multi_keep(st, g1, b1, 0, state, h, conv_channels, gate, beta, z, gamma, eps, y,
                                                  h_k, h_v, n_tok, n_keep, t_out_begin, nullptr);
    }
}

namespace {
struct MirrorReg {
    const int32_t* res = nullptr;
    int64_t n = 0;
    const unsigned long long* table = nullptr;
    int32_t* usage = nullptr;
};
MirrorReg g_mirror[16];
}
void resident_plan_set_mirror(const int32_t* d_res, int64_t entries, const unsigned long long* mirror_table,
                              int32_t* usage) {
    for (MirrorReg& r : g_mirror)
        if (r.res == d_res || r.res == nullptr) {
            r = MirrorReg{d_res, entries, mirror_table, usage};
            return;
        }
    std::fprintf(stderr, "resident_plan_set_mirror: more than 16 residency tables\n");
    std::exit(1);
}
void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* plan_err) {
    const unsigned long long* mir = nullptr;   // SYCL port: the layer's slice of the host-mirror table, if any
    int32_t* use = nullptr;                    // the layer's slice of the device routing counts, if on
    for (const MirrorReg& m : g_mirror) {
        if (m.res == nullptr) break;
        const uintptr_t r0 = (uintptr_t) m.res, r = (uintptr_t) res_layer;
        const int64_t at = r >= r0 ? (int64_t) ((r - r0) / sizeof(int32_t)) : -1;
        if (at < 0 || at + n_expert > m.n) continue;
        if (m.table != nullptr) mir = m.table + at;
        if (m.usage != nullptr) use = m.usage + at;
        break;
    }
    k_verify_kernels_resident_plan(strata::ze::stream_of(stream), {1, 1, 1}, {kResidentPlanMax, 1, 1}, 0, ids,
                                   n_entries, k, res_layer, n_expert, cache_base, slot_off, blob, plan, capx, skip, ring,
                                   mir, plan_err, use);
}
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    const uint32_t spin_max = strata::spin_max(strata::ze::stream_of(stream)->device());
    k_verify_kernels_wait_flag_ge_or(strata::ze::stream_of(stream), {1, 1, 1}, {1, 1, 1}, 0, flag, value, skip,
                                     spin_max);
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    k_verify_kernels_copy_i32_unless(strata::ze::stream_of(stream), {1, 1, 1}, {128, 1, 1}, 0, dst, src, (int) n, skip,
                                     value);
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const bool pair = strata::a770_fast();
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    k_verify_kernels_copy_or_zero(strata::ze::stream_of(stream), {(unsigned) blocks, 1, 1}, {256, 1, 1}, 0, dst, src, n4,
                                  skip, value, pair);
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    const uint32_t spin_max = strata::spin_max(strata::ze::stream_of(stream)->device());
    k_verify_kernels_wait_flag_ge(strata::ze::stream_of(stream), {1, 1, 1}, {1, 1, 1}, 0, flag, value, spin_max);
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    k_verify_kernels_embedding_gather_dev(strata::ze::stream_of(stream), {(unsigned) ((n + 255) / 256), (unsigned) n_tok, 1},
                                          {256, 1, 1}, 0, codes, scales, offsets, tokens, n, code_bits, code_bias,
                                          group_elems, row_codes, row_groups, out);
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    k_verify_kernels_broadcast_streams(strata::ze::stream_of(stream),
                                       {(unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok, 1}, {256, 1, 1}, 0, x,
                                       R, n_embd, hc);
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    k_verify_kernels_copy_indexed(strata::ze::stream_of(stream), {blocks, 1, 1}, {256, 1, 1}, 0, dst, src, stride, index,
                                  n);
}

// Opt-in GPU stage timestamps. DG2 has no comparable device-wide clock, so
// read a live host USM clock uncached (the fork's gpu_stamp implementation).
// The writer is joined before runtime teardown and exists only in profile mode.
namespace {
uint64_t profile_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct ProfileHostClock {
    uint64_t* word;
    std::atomic<bool> stop{false};
    std::thread writer;
    ProfileHostClock() : word((uint64_t*) strata::ze::malloc_host(8 * sizeof(uint64_t))) {
        if (!word) throw std::runtime_error("gpu_stamp: host clock allocation failed");
        __atomic_store_n(word, profile_now_ns(), __ATOMIC_RELEASE);
        writer = std::thread([this] {
            while (!stop.load(std::memory_order_relaxed)) {
                __atomic_store_n(word, profile_now_ns(), __ATOMIC_RELEASE);
                for (int i = 0; i < 8; ++i) __builtin_ia32_pause();
            }
        });
    }
    ~ProfileHostClock() {
        stop.store(true, std::memory_order_relaxed);
        if (writer.joinable()) writer.join();
        // Keep the word alive until context teardown, as in the fork: a graph
        // may still retain its pointer when process-level statics are destroyed.
    }
};
uint64_t* profile_clock_word() {
    static std::mutex mu;
    static std::unique_ptr<ProfileHostClock> clock;
    std::lock_guard<std::mutex> lock(mu);
    if (!clock) clock = std::make_unique<ProfileHostClock>();
    return clock->word;
}
}

void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    uint64_t* word = profile_clock_word();
    k_verify_kernels_gpu_stamp(strata::ze::stream_of(stream), {1, 1, 1}, {1, 1, 1}, 0, buf, i, word);
}

}  // namespace strata::kernels
