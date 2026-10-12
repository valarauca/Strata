// src/kernels/cuda/sampler.host.cpp - the host side of sampler.dp.cpp: path selection, the split scratch, the launchers.
#include "strata/kernels/sampler.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace strata::kernels {

#include "sampler.kern.inc"

namespace {

constexpr int kSelMax = 64;               // the widest top_k list, `sampler_kernel`'s KMAX

// top_k 1..64 as given; 0 ("off") and anything wider keep 64; never more than the vocabulary
inline int sampled_k(int top_k, int n_vocab) {
    int k = (top_k > 0 && top_k < kSelMax) ? top_k : kSelMax;
    return k > n_vocab ? n_vocab : k;
}

constexpr int kSplitPerLane = 32;                               // logits per lane, in registers
constexpr int kSplitWarpSpan = 32 * kSplitPerLane;              // 1,024 logits per warp
constexpr int kSplitWarps = 4;
constexpr int kSplitBlockSpan = kSplitWarps * kSplitWarpSpan;   // 4,096 logits per block
constexpr int kSplitMaxBlocks = 64;                             // lists the merge holds: n_vocab <= 262,144
constexpr int kSplitMaxRows = 64;                               // rows per split launch (the scratch's bound)

// Which sampled path runs, read once: `STRATA_OLD_SAMPLER=1` is `sampler_kernel` (engine 0.1.20),
// `STRATA_SAMPLER_ONE_BLOCK=1` the one-block kernel; by default the split top_k wherever it applies.
enum class SampledPath { Split, OneBlock, Old };

bool env_flag(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
}

SampledPath sampled_path() {
    static const SampledPath path = env_flag("STRATA_OLD_SAMPLER")         ? SampledPath::Old
                                    : env_flag("STRATA_SAMPLER_ONE_BLOCK") ? SampledPath::OneBlock
                                                                           : SampledPath::Split;
    return path;
}

// A stream being captured into a graph must not reach `split_scratch` (cudaMalloc): it
// gets the one-block kernel, which needs no memory of its own.  The legacy stream cannot be captured.
bool stream_capturing(void *stream) {
    if (stream == nullptr) return false;
    return ze::stream_of(stream)->capturing();
}

// The split's block lists, one buffer per (device, stream): launches on one stream run in order, so a stream reuses
// its buffer with no sync, and two streams never share one.  Grown on demand (at least doubling, up to the size of
// `kSplitMaxRows` rows at the widest vocabulary: 2 MB), never shrunk.  A grown-out buffer is retired, not freed: a
// pointer handed out earlier (to another host thread on the same stream, say) may still be waiting for its launch,
// and freeing it would need a sync that proves nothing about that thread.  Doubling keeps a slot's retired buffers
// smaller than its live one, so a slot holds about 4 MB at most (0.5 MB for the engine's <= 16 rows).  nullptr
// when the memory cannot be had, and the caller falls back to the one-block kernel; a failed grow is remembered, so
// later calls of that size do not retry cudaMalloc each time.
int32_t *split_scratch(void *stream, size_t entries) {
    struct Slot {
        int device;
        void* stream;
        int32_t *ptr;
        size_t entries;
        size_t failed;                       // the smallest size cudaMalloc refused (0: none)
    };
    static std::mutex mu;
    static std::vector<Slot> slots;
    static std::vector<int32_t *> retired;
    const int device = ze::current_device();
    std::lock_guard<std::mutex> lock(mu);
    Slot* slot = nullptr;
    for (Slot& s : slots)
        if (s.device == device && s.stream == stream) slot = &s;
    if (slot == nullptr) {
        slots.push_back({device, stream, nullptr, 0, 0});
        slot = &slots.back();
    }
    if (slot->entries >= entries) return slot->ptr;
    if (slot->failed != 0 && entries >= slot->failed) return nullptr;
    constexpr size_t kCap = (size_t) kSplitMaxRows * kSplitMaxBlocks * kSelMax;
    size_t want = 2 * slot->entries < kCap ? 2 * slot->entries : kCap;
    if (want < entries) want = entries;
    int32_t *ptr = (int32_t *) ze::malloc_device(want * 2 * sizeof(int32_t));
    if (ptr == nullptr) {
        want = entries;
        ptr = (int32_t *) ze::malloc_device(want * 2 * sizeof(int32_t));
        if (ptr == nullptr) {
            slot->failed = entries;
            return nullptr;
        }
    }
    if (slot->ptr != nullptr) retired.push_back(slot->ptr);
    slot->ptr = ptr;
    slot->entries = want;
    return ptr;
}

}  // namespace

// SYCL port: no thread-block clusters (sm_90): the caller takes the plain greedy kernel
bool sample_greedy_cluster(const float *logits, int n_tokens, int n_vocab,
                           int *out, void *stream) {
    (void) logits; (void) n_tokens; (void) n_vocab; (void) out; (void) stream;
    return false;
}

void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0)) {
        std::fprintf(stderr, "sample_tokens: penalty_last_n %d needs a history (got %p, len %d)\n",
                     p.penalty_last_n, (const void*) history, history_len);
        std::exit(1);
    }
    const unsigned shmem =
        (history != nullptr && history_len > 0 && p.penalty_last_n > 0)
            ? (unsigned)((n_vocab + 31) / 32) *
                  sizeof(unsigned) // the penalty bitmap
            : 0;
    ze::Stream* q = ze::stream_of(stream);
    if (p.greedy || p.temperature <= 0.0f) {
        // Without penalties (shmem == 0: no window) on sm_90+, a cluster of CTAs per token - the same token; see
        // `sampler_greedy_cluster_kernel`.  STRATA_ARGMAX_MULTI=0: always the one-block kernel.
        static const bool multi = [] {
            const char* v = std::getenv("STRATA_ARGMAX_MULTI");
            return !v || std::atoi(v) != 0;
        }();
        // One block per token, 1,024 threads over the vocabulary.  See `sampler_greedy_kernel`.
        const int gthreads = 1024;
        if (!(multi && shmem == 0 && sample_greedy_cluster(logits, n_tokens, n_vocab, out, stream)))
            k_sampler_greedy(q, {(unsigned) n_tokens, 1, 1}, {gthreads, 1, 1}, shmem, logits, n_vocab, history,
                             history_len, p, out);
    } else if (sampled_path() == SampledPath::Old) {
        // The same block-per-token shape: the selection's k argmax rounds reduce inside the block.  See
        // `sampler_kernel`'s header for what the old one-thread-per-token launch cost.
        k_sampler_old(q, {(unsigned) n_tokens, 1, 1}, {1024, 1, 1}, shmem, logits, n_vocab, n_tokens, history,
                      history_len, p, out);
    } else {
        // The split top_k by default: stage 1 over (61 blocks x rows) for 248,320 logits, stage 2 one warp per
        // row.  The one-block kernel when asked for, or when the split cannot run: a wider vocabulary than the merge
        // holds, more than `kSplitMaxRows` rows (the engine samples at most a verify window), a stream under capture,
        // no scratch.
        const int k = sampled_k(p.top_k, n_vocab);
        const int n_blocks = (n_vocab + kSplitBlockSpan - 1) / kSplitBlockSpan;
        int32_t *scratch = nullptr;
        if (sampled_path() == SampledPath::Split && n_blocks <= kSplitMaxBlocks && n_tokens <= kSplitMaxRows &&
            !stream_capturing(stream))
            // sized for 16 rows and 64 entries at least, so a verify window or a wider top_k does not regrow it
            scratch = split_scratch(stream, (size_t) (n_tokens > 16 ? n_tokens : 16) * n_blocks * kSelMax);
        if (scratch != nullptr) {
            k_sampler_split_part(q, {(unsigned) n_blocks, (unsigned) n_tokens, 1}, {kSplitWarps * 32, 1, 1}, 0, logits,
                                 n_vocab, history, history_len, p, k, n_blocks, scratch);
            k_sampler_split_merge(q, {(unsigned) n_tokens, 1, 1}, {32, 1, 1}, 0, scratch, n_blocks, n_vocab, p, k, out);
        } else {
            k_sampler_one_block(q, {(unsigned) n_tokens, 1, 1}, {1024, 1, 1}, shmem, logits, n_vocab, history,
                                history_len, p, out);
        }
    }

    if (stream == nullptr) ze::device_sync();
}

namespace {
int coupled_blocks(int nv) { return (nv + kSplitBlockSpan - 1) / kSplitBlockSpan; }
int coupled_kpart(int nv) { return nv < kSelMax ? nv : kSelMax; }
}  // namespace

size_t coupled_draft_scratch_bytes(int nv) {
    if (nv <= 0 || coupled_blocks(nv) > kSplitMaxBlocks) return 0;
    return (size_t)coupled_blocks(nv) * (size_t)kSelMax * (2 * sizeof(int32_t));
}

void coupled_draft_stage(const SamplerParams* mapped_params, const int32_t* mapped_hist, SamplerParams* params,
                         int32_t* ring, int cap, void* stream) {
    k_sampler_coupled_stage(ze::stream_of(stream), {1, 1, 1}, {256, 1, 1}, 0, mapped_params, mapped_hist, params, ring,
                            cap);
}

void coupled_draft_sample(float* logits, int nv, const int32_t* sub_to_id, const int32_t* id_to_sub, int id_vocab,
                          const SamplerParams* params, int32_t* ring, int cap, int j, const int32_t* step_rec,
                          void* scratch, int32_t* out_id, float* out_prob, void* stream) {
    ze::Stream* const s = ze::stream_of(stream);
    const int n_blocks = coupled_blocks(nv), kpart = coupled_kpart(nv);
    if (nv <= 0 || n_blocks > kSplitMaxBlocks || scratch == nullptr) {
        std::fprintf(stderr, "coupled_draft_sample: %d logits need scratch and at most %d blocks\n", nv, kSplitMaxBlocks);
        std::exit(1);
    }
    k_sampler_coupled_penalize(s, {1, 1, 1}, {1024, 1, 1}, (unsigned)((nv + 31) / 32) * sizeof(unsigned), logits, nv,
                               id_to_sub, id_vocab, params, ring, cap, j);
    // the selection of the split sampler, unchanged: every 4,096-logit block's first `kpart` (the widest list,
    // since the request's top_k is only known on the device), penalties already applied above
    k_sampler_coupled_split_part(s, {(unsigned) n_blocks, 1, 1}, {kSplitWarps * 32, 1, 1}, 0, logits, nv, kpart,
                                 n_blocks, scratch);
    k_sampler_coupled_merge(s, {1, 1, 1}, {32, 1, 1}, 0, scratch, n_blocks, nv, kpart, params, step_rec, sub_to_id,
                            ring, cap, j, out_id, out_prob);
}

}  // namespace strata::kernels
