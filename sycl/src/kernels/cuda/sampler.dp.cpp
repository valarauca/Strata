// src/kernels/cuda/sampler.cu - P2.S2: the sampler chain, in llama.cpp's order.
//
//     penalties -> top_k -> top_p -> min_p -> temperature -> pick
//
// THE ORDER IS THE WHOLE CONTENT OF THIS FILE.  llama.cpp builds its chain by walking `params.samplers`, whose
// default is { PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE } (`common/common.h`
// at 3cf03257) - ONE penalties stage, first, and TEMPERATURE AFTER THE TRUNCATION FILTERS.  (Issue #53: this file
// used to apply the penalties a second time after the temperature, and min_p before top_p - both taken from the
// order of the `case` labels in `common/sampling.cpp`, which is not the order the chain runs.)  Every order
// produces a valid token, so only a comparison at the distribution level can tell them apart; the parity test
// does that against an independently computed distribution.
//
// `sampler_greedy_kernel` is the plain argmax, one block per token over the vocabulary (on sm_90+ without penalties,
// `sampler_greedy_cluster_kernel`: the same token from a cluster of 8 CTAs per row).  The sampled chain has
// three implementations that pick the same token, bit for bit:
//   - the SPLIT top_k (default): `sampler_split_part_kernel` cuts each row into 4,096-logit blocks over the whole
//     GPU, each keeps its own top_k, and `sampler_split_merge_kernel` merges those lists and runs the tail;
//   - `sampler_one_block_kernel` (`STRATA_SAMPLER_ONE_BLOCK=1`, and the fallback when the split cannot run): one
//     block per token, `top_k` block-argmax rounds, each over the logits after the previous pick;
//   - `sampler_kernel` (`STRATA_OLD_SAMPLER=1`), the kernel of engine 0.1.20, kept as the reference.
// The two new ones share `sampled_tail_warp` (top_p / min_p / temperature / draw on one warp).
#include "strata/port.hpp"
#include "strata/sycl_math.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/core/coupled_draft.hpp"

namespace strata::kernels {
namespace {

// Philox 4x32-10, the counter-based generator the phase asks for.  Counter-based matters because it makes the
// stream a function of (seed, position) rather than of how many draws came before - so a batch can be sampled
// in any order and a run is reproducible.
__dpct_inline__ uint32_t philox4x32_round(uint32_t &c0, uint32_t &c1,
                                          uint32_t &c2, uint32_t &c3,
                                          uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = sycl::mul_hi<unsigned>(0x9E3779B9u, c0);
    const uint32_t hi1 = sycl::mul_hi<unsigned>(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    return 0;
}

__dpct_inline__ float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    }
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// `count_in_history` and the penalty application, transcribed from `llama_sampler_penalties_apply`.
// The repeat penalty MULTIPLIES for non-positive logits and DIVIDES for positive ones - dividing
// unconditionally is the natural reading of the source paper and it INVERTS the penalty on half the
// vocabulary.  The presence penalty is `float(count > 0)`, a boolean, not the count.
__dpct_inline__ int history_count(const int *__restrict__ h, int n, int v) {
    int c = 0;
#pragma unroll
    for (int i = 0; i < n; ++i) if (h[i] == v)++ c;
    return c;
}

__dpct_inline__ float apply_penalties(float logit, int count,
                                      const SamplerParams &p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

/// **THE GREEDY ARGMAX, ONE BLOCK PER TOKEN, COVERING THE VOCABULARY.**
///
/// **WHY THIS IS A SEPARATE KERNEL AND NOT A BRANCH.**  `sampler_kernel` is launched as a grid over TOKENS
/// with 64 threads and a `if (t >= n_tokens) return;` at the top.  The decode path has `n_tokens == 1`, so
/// that launch was `<<<1, 64>>>`, 63 threads exited on the first line, and ONE THREAD walked all 248,320
/// logits in a dependent loop on one SM of 48.  Measured in isolation (`bench/micro/sampler_cost.cu`):
/// **3.11 ms per token**, 5.7% of an ~54 ms token, and the whole of round 309's `sample` phase - the two
/// synchronisations around it are 0.03 ms each.
///
/// The obvious repair is to parallelise the scan inside `sampler_kernel`, and it is WRONG: with one thread
/// per token, a block reduction over the vocabulary has nothing to reduce, and the threads that returned
/// early are not there for `__syncthreads` or `__shfl_down_sync`.  The first attempt did exactly that and
/// produced the token `5120` thirty-two times.  The grid has to be over tokens with the BLOCK over the
/// vocabulary, which is a different launch configuration and therefore a different kernel.
///
/// **THE TIE RULE IS UNCHANGED AND THAT IS THE WHOLE CORRECTNESS ARGUMENT.**  The serial scan walked `v`
/// ascending with `if (s > bv)`, so the LOWEST index wins a tie.  Each thread keeps that rule over its own
/// strided subset and the reduction resolves two candidates by taking the larger value and, on equality, the
/// SMALLER index - the same total order, so `sampler_parity` and C1 see no change.
__dpct_inline__ void
sampler_greedy_kernel(const float *__restrict__ logits, int n_vocab,
                      const int *__restrict__ history, int history_len,
                      const SamplerParams p, int pmin, int plen,
                      int *__restrict__ out, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
    const float* l = logits + (size_t) t * n_vocab;
    (void) pmin;
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = plen < history_len ? plen : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }

    // PENALTY MEMBERSHIP AS A BITMAP.  The history touches at most `hlen` tokens of a quarter-million
    // vocabulary, but the naive `history_count` per candidate per argmax round costs O(k x n_vocab x hlen)
    // integer compares (~318 M per token at k=20, hlen=64 - measured 45 -> 31 tok/s on a real workload).
    // A shared bitmap gives an O(1) membership test, and only the (at most hlen) hits pay the count scan;
    // the counts - and therefore every sampled value - are exactly what the per-candidate scan produced.
    auto penal_bits = (unsigned int *)dpct_local;
    const int bits_words = (int) ((n_vocab + 31) / 32);
    // The gate needs a NON-EMPTY WINDOW (`hlen > 0`): the launch sizes the shared bitmap only when penalties
    // are on, so a caller handing over a history buffer with `penalty_last_n == 0` must not touch it.
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
#pragma unroll
        for (int w = item_ct1.get_local_id(2); w < bits_words;
             w += item_ct1.get_local_range(2)) penal_bits[w] = 0u;
        item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < hlen;
             i += item_ct1.get_local_range(2))
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                dpct::atomic_fetch_or<
                    sycl::access::address_space::generic_space>(
                    &penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    auto hit_count = [&](int v, uint8_t *dpct_local) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    // `n_vocab` is the "no candidate" index: it loses every comparison to a real one, so a thread with no
    // elements contributes nothing rather than contributing a bogus zero.
    float bv = sycl::bit_cast<float, int>(0xff800000); // -inf
    int best = n_vocab;
    for (int v = item_ct1.get_local_id(2); v < n_vocab;
         v += item_ct1.get_local_range(2)) {
        const float s = apply_penalties(l[v], hit_count(v, dpct_local), p);
        if (s > bv) { bv = s; best = v; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = dpct::experimental::shift_sub_group_left(
            0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), bv,
            off);
        const int oi = dpct::experimental::shift_sub_group_left(
            0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            best, off);
        if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
    }
    auto &sv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &si = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int warp = (int)(item_ct1.get_local_id(2) >> 5),
              lane = (int)(item_ct1.get_local_id(2) & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = best; }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (warp == 0) {
        const int nw = (int)((item_ct1.get_local_range(2) + 31) >> 5);
        float wv =
            lane < nw ? sv[lane] : sycl::bit_cast<float, int>(0xff800000);
        int wi = lane < nw ? si[lane] : n_vocab;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = dpct::experimental::shift_sub_group_left(
                0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                wv, off);
            const int oi = dpct::experimental::shift_sub_group_left(
                0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                wi, off);
            if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
        }
        // A tie between two `-inf` candidates leaves `wi == n_vocab`, and the serial version answered 0.
        if (lane == 0) out[t] = (wi < n_vocab) ? wi : 0;
    }
}

#if !defined(__HIPCC__)
/// **THE SAME ARGMAX ON A THREAD-BLOCK CLUSTER (sm_90+, S19), FOR THE CALLS WITHOUT PENALTIES.**  One block per
/// token reads its 1 MB row (248,320 logits) on one SM with one load in flight per thread: latency-bound, 40 us per
/// call on an RTX 5070, for every verify window and every draft step.  Here a cluster of `kAmCtas` CTAs shares the
/// row, each thread keeps four loads in flight, and the CTAs' results meet in CTA 0's shared memory (distributed
/// shared memory): 6 us (decode_cluster_parity --bench).
///
/// The answer is a function of the row alone, so it is the one-block kernel's bit for bit: the LOWEST index whose value
/// is the largest non-NaN value above -inf (each thread walks its elements in ascending order with a strict `>`, and
/// every merge takes the larger value or, on equality, the smaller index - an order-free rule), and 0 when there is
/// none (all -inf / NaN), as there.  NaN never wins a `>`.
///
/// Barriers: a relaxed cluster arrive at entry, waited before the remote store (CTA 0 must be running); each CTA's
/// result goes to slot `rank` of CTA 0, released by the next arrive; CTAs 1.. then exit (a cluster wait counts the
/// threads that have not exited) and CTA 0 waits, reads its own slots, and writes the token.
constexpr int kAmCtas = 8;      // CTAs per token (the portable cluster size)
constexpr int kAmThreads = 1024;
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 900
#define STRATA_AM_CLUSTER 1
#else
#define STRATA_AM_CLUSTER 0     // older targets: a trap, never launched (sample_greedy_cluster checks)
#endif
#if STRATA_AM_CLUSTER
__dpct_inline__ void am_take(float s, int v, float &bv, int &best) {
    if (s > bv) { bv = s; best = v; }
}
__dpct_inline__ void am_merge(float ov, int oi, float &bv, int &best) {
    if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
}
#endif
// grid (kAmCtas, n_tokens), cluster (kAmCtas, 1, 1), kAmThreads threads
void sampler_greedy_cluster_kernel(const float* __restrict__ logits,
                                                                            int n_vocab, int* __restrict__ out) {
#if STRATA_AM_CLUSTER
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &si = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &cv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kAmCtas]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // on CTA 0: each CTA's result
    auto &ci =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[kAmCtas]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
    const float *l = logits + (size_t)item_ct1.get_group(1) * n_vocab;
    float bv = sycl::bit_cast<float, int>(
        0xff800000); // -inf; `n_vocab` is "no candidate", as in
                     // sampler_greedy_kernel
    int best = n_vocab;
    constexpr int S = kAmCtas * kAmThreads;
    int v = (int)rank * kAmThreads + (int)item_ct1.get_local_id(2);
    for (; v + 3 * S < n_vocab; v += 4 * S) {   // four independent loads, then taken in ascending order
        const float x0 = l[v], x1 = l[v + S], x2 = l[v + 2 * S], x3 = l[v + 3 * S];
        am_take(x0, v, bv, best);
        am_take(x1, v + S, bv, best);
        am_take(x2, v + 2 * S, bv, best);
        am_take(x3, v + 3 * S, bv, best);
    }
#pragma unroll
    for (; v < n_vocab; v += S) am_take(l[v], v, bv, best);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        am_merge(
            dpct::experimental::shift_sub_group_left(
                0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bv, off),
            dpct::experimental::shift_sub_group_left(
                0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                best, off),
            bv, best);
    const int warp = (int)(item_ct1.get_local_id(2) >> 5),
              lane = (int)(item_ct1.get_local_id(2) & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = best; }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (warp == 0) {
        bv = sv[lane];
        best = si[lane];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            am_merge(dpct::experimental::shift_sub_group_left(
                         0xFFFFFFFFu,
                         sycl::ext::oneapi::this_work_item::get_sub_group(), bv,
                         off),
                     dpct::experimental::shift_sub_group_left(
                         0xFFFFFFFFu,
                         sycl::ext::oneapi::this_work_item::get_sub_group(),
                         best, off),
                     bv, best);
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::
                     : "memory"); // CTA 0 runs
    if (item_ct1.get_local_id(2) == 0) {
        uint64_t a;
        asm volatile("mapa.u64 %0, %1, %2;\n"
                     : "=l"(a)
                     : "l"((uint64_t)&cv[rank]), "r"(0u));
        *reinterpret_cast<float*>(a) = bv;
        asm volatile("mapa.u64 %0, %1, %2;\n"
                     : "=l"(a)
                     : "l"((uint64_t)&ci[rank]), "r"(0u));
        *reinterpret_cast<int*>(a) = best;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    if (rank != 0) return;
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    if (warp == 0) {
        bv = lane < kAmCtas ? cv[lane] : sycl::bit_cast<float, int>(0xff800000);
        best = lane < kAmCtas ? ci[lane] : n_vocab;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            am_merge(dpct::experimental::shift_sub_group_left(
                         0xFFFFFFFFu,
                         sycl::ext::oneapi::this_work_item::get_sub_group(), bv,
                         off),
                     dpct::experimental::shift_sub_group_left(
                         0xFFFFFFFFu,
                         sycl::ext::oneapi::this_work_item::get_sub_group(),
                         best, off),
                     bv, best);
        if (lane == 0) out[item_ct1.get_group(1)] = (best < n_vocab) ? best : 0;
    }
#else
    (void) logits; (void) n_vocab; (void) out;
    __builtin_trap();
#endif
}
#endif  // !__HIPCC__

/// **THE SAMPLED PATH, ONE BLOCK PER TOKEN.**  The kernel below replaced a version that ran the whole chain
/// in ONE THREAD per token (`<<<ceil(T/64), 64>>>`, so a 4-token window fielded four threads): `top_k` alone
/// was `k` sequential scans of the vocabulary with an inner sweep over the already-taken list - 20 x 248,320
/// iterations of dependent work on one SM - and a verify window measured **1.6 s in the sampler**, which made
/// every temperature-bearing request ~30x slower than a greedy one.  The selection is `k` argmax rounds, and
/// an argmax over the vocabulary parallelises exactly like `sampler_greedy_kernel` (block over the vocab), so
/// the rounds run back to back inside a block-per-token launch: the per-token cost falls to
/// `k x n_vocab / 1024` plus `k` block reductions.
///
/// THE SEMANTICS ARE THE SERIAL ONES, EXACTLY.  Each round's argmax resolves ties to the LOWEST index (the
/// serial scan's strict `>` keeps the first maximum it meets), so the kept sequence - both its set and its
/// order - is unchanged; `top_p`'s cut reads that order in double arithmetic as before; temperature and the
/// Philox draw apply after the cut.  `sampler_parity` pins all of it against the host reference.
///
/// **KEPT AS THE REFERENCE, BEHIND `STRATA_OLD_SAMPLER=1`.**  Two costs remain in it: the `taken`
/// sweep is O(k) per logit per round, O(k^2 x n_vocab) per row (47 M shared-memory compares at k = 20, 500 M at
/// 64), and the double-precision tail runs on all 1,024 threads where one warp suffices - GeForce issues FP64 at
/// 1/64 of FP32.  The kernels after this one remove both and select the same list in the same order.
__dpct_inline__ void sampler_kernel(const float *__restrict__ logits,
                                    int n_vocab, int n_tokens,
                                    const int *__restrict__ history,
                                    int history_len, const SamplerParams p,
                                    int *__restrict__ out,
                                    uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
    if (t >= n_tokens) return;
    const float* l = logits + (size_t) t * n_vocab;

    // Temperature is needed by BOTH stages below, so it is computed here; the chain still APPLIES it after
    // the truncation filters - the survivors are chosen on the raw logits and only then scaled.
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;

    // The penalty window is the last `penalty_last_n` entries of this row's history (disabled at this
    // launch: `sample_tokens` refuses a non-zero `penalty_last_n` without a history buffer).
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }

    // the membership bitmap, as in `sampler_greedy_kernel` - see the cost note there.  The gate needs an
    // NON-EMPTY WINDOW too: the launch sizes the bitmap only when penalties are on, so a caller that hands over
    // a stale history buffer with `penalty_last_n == 0` must not touch it.
    auto penal_bits = (unsigned int *)dpct_local;
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
#pragma unroll
        for (int w = item_ct1.get_local_id(2); w < bits_words;
             w += item_ct1.get_local_range(2)) penal_bits[w] = 0u;
        item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < hlen;
             i += item_ct1.get_local_range(2))
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                dpct::atomic_fetch_or<
                    sycl::access::address_space::generic_space>(
                    &penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    auto hit_count = [&](int v, uint8_t *dpct_local) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    // top_k in 1..64 is taken as given; 0 ("off") and anything wider mean the widest shortlist the kernel
    // keeps, 64.  Every row writes out[t]: a verify window reads all of them.
    const int KMAX = 64;
    int k = (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX;
    if (k > n_vocab) k = n_vocab;

    // ---- top_k: k rounds of a block argmax over the not-yet-taken.  `sel_*` holds the kept ids and their
    // raw logits in selection order: descending by value, ties to the lower index, which is the order the
    // top_p cut below is defined over.
    auto &sel_ids =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[KMAX]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sel_logit =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[KMAX]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &si = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int i = 0; i < k; ++i) {
        // `n_vocab` is the "no candidate" index: it loses every comparison to a real one (same convention as
        // the greedy kernel, whose tie rule this reduction shares).
        float bv = sycl::bit_cast<float, int>(0xff800000); // -inf
        int best = n_vocab;
        for (int v = item_ct1.get_local_id(2); v < n_vocab;
             v += item_ct1.get_local_range(2)) {
            bool taken = false;
#pragma unroll
            for (int j = 0; j < i; ++j) if (sel_ids[j] == v) {
                taken = true; break;
            }
            if (taken) continue;
            const float s = apply_penalties(l[v], hit_count(v, dpct_local), p);
            if (s > bv) { bv = s; best = v; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = dpct::experimental::shift_sub_group_left(
                0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bv, off);
            const int oi = dpct::experimental::shift_sub_group_left(
                0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                best, off);
            if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
        }
        const int warp = (int)(item_ct1.get_local_id(2) >> 5),
                  lane = (int)(item_ct1.get_local_id(2) & 31);
        if (lane == 0) { sv[warp] = bv; si[warp] = best; }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (warp == 0) {
            const int nw = (int)((item_ct1.get_local_range(2) + 31) >> 5);
            float wv =
                lane < nw ? sv[lane] : sycl::bit_cast<float, int>(0xff800000);
            int wi = lane < nw ? si[lane] : n_vocab;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = dpct::experimental::shift_sub_group_left(
                    0xFFFFFFFFu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), wv,
                    off);
                const int oi = dpct::experimental::shift_sub_group_left(
                    0xFFFFFFFFu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), wi,
                    off);
                if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
            }
            if (lane == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // ---- top_p over the top_k list (penalised logits, descending as the selection produced them), then min_p,
    // then temperature and one Philox draw - llama.cpp's order (issue #53).  Every thread computes the same chain
    // redundantly over `sel_*` - the arithmetic is the serial kernel's, instruction for instruction - so they
    // agree on `pick` and thread 0 writes it.
    int n_keep = k;
    float mx = sel_logit[0];
#pragma unroll
    for (int i = 1; i < k; ++i) mx = sycl::fmax(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        strata::samp_acc_t sum = 0.0;
#pragma unroll
        for (int i = 0; i < k; ++i) sum +=
            sycl::exp((strata::samp_acc_t)sel_logit[i] - (strata::samp_acc_t)mx);
        strata::samp_acc_t cum = 0.0;
        int cut = k;
#pragma unroll
        for (int i = 0; i < k; ++i) {
            cum += sycl::exp((strata::samp_acc_t)sel_logit[i] - (strata::samp_acc_t)mx) / sum;
            if (cum >= (strata::samp_acc_t) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    // ---- min_p on top_p's survivors: the descending prefix whose probability is at least `min_p` of the top
    // token's.  In logit space the threshold is `sel_logit[0] + logf(min_p)` - equivalent to `p >= min_p * p_max`
    // without the overflow an exp of raw logits risks.  0 disables, and the head itself always survives
    // (`expf(0) == 1 >= min_p` for min_p in 0..1), so the count never reaches zero.
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + sycl::log((float)(p.min_p));
#pragma unroll
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature only: the penalties were applied once, before the selection (issue #53: they were applied a
    // second time here, after the temperature scaling - llama.cpp's chain has one penalties stage)
    auto scaled = [&](int i) { return sel_logit[i] * inv_t; };
    float smx = scaled(0);
#pragma unroll
    for (int i = 1; i < n_keep; ++i) smx = sycl::fmax(smx, scaled(i));
    strata::samp_acc_t sum = 0.0;
#pragma unroll
    for (int i = 0; i < n_keep; ++i) sum +=
        sycl::exp((strata::samp_acc_t)scaled(i) - (strata::samp_acc_t)smx);
    const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
    strata::samp_acc_t cum = 0.0;
    int pick = sel_ids[n_keep - 1];
#pragma unroll
    for (int i = 0; i < n_keep; ++i) {
        cum += sycl::exp((strata::samp_acc_t)scaled(i) - (strata::samp_acc_t)smx) / sum;
        if ((strata::samp_acc_t) u < cum) { pick = sel_ids[i]; break; }
    }
    if (item_ct1.get_local_id(2) == 0) out[t] = pick;
}

// ---- the same list, in the same order, without the `taken` sweep ----
//
// THE SELECTION ORDER.  `sampler_kernel`'s rounds rank a candidate (value, id) before (value', id') when value >
// value', or value == value' and id < id' - each thread's ascending scan keeps the first maximum it meets (strict
// `>`), and the reductions resolve a tie to the lower id.  That is a strict total order (-0 and +0 compare equal and
// fall to the id, as there), so round i picks the i-th logit of the order.  NaN and -inf are never picked (`s > bv`
// from -inf fails); a round that finds nothing yields the SENTINEL (-inf, n_vocab), stored as id 0.

constexpr int kSelMax = 64;               // the widest top_k list, `sampler_kernel`'s KMAX
constexpr unsigned kFullMask = 0xFFFFFFFFu;

// top_k 1..64 as given; 0 ("off") and anything wider keep 64; never more than the vocabulary
__dpct_inline__ int sampled_k(int top_k, int n_vocab) {
    int k = (top_k > 0 && top_k < kSelMax) ? top_k : kSelMax;
    return k > n_vocab ? n_vocab : k;
}

// (bv, bi) <- the first of (bv, bi) and (ov, oi) in the selection order
__dpct_inline__ void take_first(float &bv, int &bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// The first of the warp's 32 candidates, left in EVERY lane.  An XOR butterfly is exact here: "the first of two" is
// associative and commutative in a strict total order (two lanes never hold different candidates that compare
// equal - an id is in one lane at most, and two sentinels are the same pair), so every lane ends with the same pair.
__dpct_inline__ void warp_first(float &bv, int &bi) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = dpct::experimental::permute_sub_group_by_xor(
            kFullMask, sycl::ext::oneapi::this_work_item::get_sub_group(), bv,
            off);
        const int oi = dpct::experimental::permute_sub_group_by_xor(
            kFullMask, sycl::ext::oneapi::this_work_item::get_sub_group(), bi,
            off);
        take_first(bv, bi, ov, oi);
    }
}

/// **THE TAIL ON ONE WARP, WITH `sampler_kernel`'S ARITHMETIC.**  `sel_ids` / `sel_logit` (shared, `k` entries) are
/// the top_k list in selection order; the 32 lanes of one warp call this and lane 0 writes `out[t]`.  The old tail
/// ran on all 1,024 threads, each computing the same ~4k double `exp`s - FP64 issues at 1/64 of FP32 on GeForce.
/// Here the lanes share the `exp`s (one per entry, into `ex`) and then the quotients, and lane 0 alone runs the two
/// ORDERED sums and the two cumulative scans.  Every double is the one the old chain computed: the same `exp` of
/// the same argument, the sums in the same order, `cum += e / sum` with the same correctly rounded quotient - so
/// the cut, the survivors and the pick are the same.  (`n_keep == 0`, reachable only with min_p > 1, which the
/// callers clamp, read `sel_ids[-1]` in the old tail; it reads `sel_ids[0]` here.)
/// kProb (the coupled draft only): lane 0 also writes the pick's probability under the final distribution to
/// `*prob_out`.  The sampler's own calls take kProb = false, whose code is the tail above, unchanged.
template <bool kProb = false>
inline void sampled_tail_warp(const int *sel_ids, const float *sel_logit, int k,
                              const SamplerParams &p, int t,
                              int *__restrict__ out, strata::samp_acc_t *ex,
                              float *prob_out = nullptr) {
    const int lane =
        (int)(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
                  2) &
              31);
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    int n_keep = k;
    float mx = sel_logit[0];
#pragma unroll
    for (int i = 1; i < k; ++i) mx = sycl::fmax(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
#pragma unroll
        for (int i = lane; i < k; i += 32)
            ex[i] = sycl::exp((strata::samp_acc_t)sel_logit[i] - (strata::samp_acc_t)mx);
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        strata::samp_acc_t sum = 0.0;
        if (lane == 0)
#pragma unroll
            for (int i = 0; i < k; ++i) sum += ex[i];
        sum = dpct::experimental::select_from_sub_group(
            kFullMask, sycl::ext::oneapi::this_work_item::get_sub_group(), sum,
            0);
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::
                                get_sub_group()); // lane 0 has read every `ex`
                                                  // before it is overwritten
#pragma unroll
        for (int i = lane; i < k; i += 32) ex[i] = ex[i] / sum;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        int cut = k;
        if (lane == 0) {
            strata::samp_acc_t cum = 0.0;
#pragma unroll
            for (int i = 0; i < k; ++i) {
                cum += ex[i];
                if (cum >= (strata::samp_acc_t) p.top_p) { cut = i + 1; break; }
            }
        }
        cut = dpct::experimental::select_from_sub_group(
            kFullMask, sycl::ext::oneapi::this_work_item::get_sub_group(), cut,
            0);
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::
                                get_sub_group()); // `ex` is written again below
    }
    // min_p on top_p's survivors, as in `sampler_kernel` (every lane, the same float arithmetic)
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + sycl::log((float)(p.min_p));
#pragma unroll
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature, then one Philox draw
    float smx = sel_logit[0] * inv_t;
#pragma unroll
    for (int i = 1; i < n_keep; ++i)
        smx = sycl::fmax(smx, sel_logit[i] * inv_t);
#pragma unroll
    for (int i = lane; i < n_keep; i += 32)
        ex[i] = sycl::exp((strata::samp_acc_t)(sel_logit[i] * inv_t) - (strata::samp_acc_t)smx);
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    strata::samp_acc_t sum = 0.0;
    if (lane == 0)
#pragma unroll
        for (int i = 0; i < n_keep; ++i) sum += ex[i];
    sum = dpct::experimental::select_from_sub_group(
        kFullMask, sycl::ext::oneapi::this_work_item::get_sub_group(), sum, 0);
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
    for (int i = lane; i < n_keep; i += 32) ex[i] = ex[i] / sum;
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    if (lane == 0) {
        const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
        strata::samp_acc_t cum = 0.0;
        int pi = n_keep > 0 ? n_keep - 1 : 0;
        int pick = sel_ids[pi];
#pragma unroll
        for (int i = 0; i < n_keep; ++i) {
            cum += ex[i];
            if ((strata::samp_acc_t) u < cum) { pick = sel_ids[i]; pi = i; break; }
        }
        out[t] = pick;
        if constexpr (kProb) *prob_out = n_keep > 0 ? (float) ex[pi] : 1.0f;
    }
}

/// **THE ONE-BLOCK SAMPLED PATH: `sampler_kernel` WITHOUT THE `taken` SWEEP.**  Round i's candidates are the logits
/// strictly AFTER round i-1's pick in the selection order, `s < prev_v || (s == prev_v && v > prev_i)`, which is
/// exactly the set `taken` left: the picks so far are the first i of the order, so what comes after the last one is
/// what has not been picked.  The round is then the same block argmax with the same tie rule, so the list is the
/// same list in the same order.  An empty round leaves (-inf, id 0) as before, and every round after it is empty
/// in both versions (nothing after -inf beats -inf).  O(k x n_vocab) per row instead of O(k^2 x n_vocab), then warp
/// 0 runs the tail.  `STRATA_SAMPLER_ONE_BLOCK=1`, and the fallback when the split path cannot run.
__dpct_inline__ void
sampler_one_block_kernel(const float *__restrict__ logits, int n_vocab,
                         const int *__restrict__ history, int history_len,
                         const SamplerParams p, int *__restrict__ out,
                         uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
    const float* l = logits + (size_t) t * n_vocab;

    // the penalty window and its membership bitmap, exactly as in `sampler_kernel`
    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    auto penal_bits = (unsigned int *)dpct_local;
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
#pragma unroll
        for (int w = item_ct1.get_local_id(2); w < bits_words;
             w += item_ct1.get_local_range(2)) penal_bits[w] = 0u;
        item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < hlen;
             i += item_ct1.get_local_range(2))
            if (hrow[i] >= 0 && hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                dpct::atomic_fetch_or<
                    sycl::access::address_space::generic_space>(
                    &penal_bits[hrow[i] >> 5], 1u << (hrow[i] & 31));
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    auto hit_count = [&](int v, uint8_t *dpct_local) -> int {
        if (!use_bits || !(penal_bits[v >> 5] & (1u << (v & 31)))) return 0;
        return history_count(hrow, hlen, v);
    };

    const int k = sampled_k(p.top_k, n_vocab);
    auto &sel_ids =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[kSelMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sel_logit =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kSelMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &ex =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<strata::samp_acc_t[kSelMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &si = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int warp = (int)(item_ct1.get_local_id(2) >> 5),
              lane = (int)(item_ct1.get_local_id(2) & 31);
    float prev_v = sycl::bit_cast<float>(
        0x7f800000); // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    for (int i = 0; i < k; ++i) {
        float bv = sycl::bit_cast<float, int>(0xff800000); // -inf
        int best = n_vocab;
        for (int v = item_ct1.get_local_id(2); v < n_vocab;
             v += item_ct1.get_local_range(2)) {
            const float s = apply_penalties(l[v], hit_count(v, dpct_local), p);
            if ((s < prev_v || (s == prev_v && v > prev_i)) && s > bv) { bv = s; best = v; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = dpct::experimental::shift_sub_group_left(
                0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bv, off);
            const int oi = dpct::experimental::shift_sub_group_left(
                0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                best, off);
            if (ov > bv || (ov == bv && oi < best)) { bv = ov; best = oi; }
        }
        if (lane == 0) { sv[warp] = bv; si[warp] = best; }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (warp == 0) {
            const int nw = (int)((item_ct1.get_local_range(2) + 31) >> 5);
            float wv =
                lane < nw ? sv[lane] : sycl::bit_cast<float, int>(0xff800000);
            int wi = lane < nw ? si[lane] : n_vocab;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = dpct::experimental::shift_sub_group_left(
                    0xFFFFFFFFu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), wv,
                    off);
                const int oi = dpct::experimental::shift_sub_group_left(
                    0xFFFFFFFFu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), wi,
                    off);
                if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
            }
            if (lane == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        // An empty round leaves (-inf, 0): nothing comes after it, as nothing was left untaken.
        prev_v = sel_logit[i];
        prev_i = sel_ids[i];
    }
    if (warp != 0) return;
    sampled_tail_warp(sel_ids, sel_logit, k, p, t, out, ex);
}

// ---- THE SPLIT top_k.  `sampler_kernel` and the one-block kernel keep a row on ONE SM of 170 (5090) and walk its
// 248,320 logits k times.  Here a warp keeps the top_k of 1,024 logits held in registers, four warps make a block of
// 4,096 logits that merges their lists, and a row is 61 such blocks over the whole GPU; one warp per row then
// merges the 61 lists and runs the tail.  Exact, because the first k of a union are within the first k of each part
// and a merge of ordered lists is ordered - the selection order being a strict total order (above).
constexpr int kSplitPerLane = 32;                               // logits per lane, in registers
constexpr int kSplitWarpSpan = 32 * kSplitPerLane;              // 1,024 logits per warp
constexpr int kSplitWarps = 4;
constexpr int kSplitBlockSpan = kSplitWarps * kSplitWarpSpan;   // 4,096 logits per block
constexpr int kSplitMaxBlocks = 64;                             // lists the merge holds: n_vocab <= 262,144
constexpr int kSplitMaxRows = 64;                               // rows per split launch (the scratch's bound)

/// Merge `nl` (<= 64) lists of `k` candidates - list L at `lists[L * stride]`, each in the selection order and
/// padded with sentinels - into their first `k`: `sink(i, value, id)` runs in every lane for i = 0..k-1 with the
/// same pair.  Lane owns lists `lane` and `lane + 32`; a round takes the first of all heads and advances the list
/// it came from.  An id is in one list at most (the lists cover disjoint logits), so exactly one head matches.
template <typename Sink>
__dpct_inline__ void warp_merge_lists(const sycl::int2 *lists, int nl,
                                      int stride, int k, int n_vocab,
                                      Sink &&sink) {
    const int lane =
        (int)(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
                  2) &
              31);
    float hv[2];
    int hi[2], pos[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int L = lane + 32 * m;
        pos[m] = 0;
        hv[m] = sycl::bit_cast<float, int>(0xff800000);
        hi[m] = n_vocab;
        if (L < nl) {
            const sycl::int2 c = lists[(size_t)L * stride];
            hi[m] = c.x();
            hv[m] = sycl::bit_cast<float>(c.y());
        }
    }
    for (int i = 0; i < k; ++i) {
        float bv = hv[0];
        int bi = hi[0];
        take_first(bv, bi, hv[1], hi[1]);
        warp_first(bv, bi);
        sink(i, bv, bi);
        if (bi < n_vocab) {
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                if (hi[m] != bi) continue;
                if (++pos[m] < k) {
                    const sycl::int2 c =
                        lists[(size_t)(lane + 32 * m) * stride + pos[m]];
                    hi[m] = c.x();
                    hv[m] = sycl::bit_cast<float>(c.y());
                } else {
                    hi[m] = n_vocab;
                    hv[m] = sycl::bit_cast<float, int>(0xff800000);
                }
            }
        }
    }
}

/// **SPLIT STAGE 1: THE top_k OF EACH 4,096-LOGIT BLOCK.**  Grid (blocks per row, rows), 128 threads.  Each warp
/// loads its 1,024 penalised logits once into registers (lane + 32 j: every load is one coalesced 128-byte line)
/// and runs `k` warp-argmax rounds over them with the threshold of `sampler_one_block_kernel` - no shared memory
/// and no block barrier per round.  Warp 0 then merges the four warp lists into the block's list in `cand`
/// (row-major: row t, block b, entry i at `(t * n_blocks + b) * k + i`, as (id, value bits)).
__dpct_inline__ void
sampler_split_part_kernel(const float *__restrict__ logits, int n_vocab,
                          const int *__restrict__ history, int history_len,
                          const SamplerParams p, int k, int n_blocks,
                          sycl::int2 *__restrict__ cand) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1);
    const float* l = logits + (size_t) t * n_vocab;
    const int warp = (int)(item_ct1.get_local_id(2) >> 5),
              lane = (int)(item_ct1.get_local_id(2) & 31);
    const int blo = (int)item_ct1.get_group(2) * kSplitBlockSpan;

    const int* hrow = history ? history + (size_t) t * history_len : nullptr;
    int hlen = 0;
    if (hrow) {
        hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;          // the window is the TAIL
    }
    // The membership bitmap of THIS BLOCK'S 4,096 logits (512 bytes, not the vocabulary's 31 KB): the same test,
    // and a hit pays the same exact count over the whole window.
    auto &bits = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        unsigned int[kSplitBlockSpan / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const bool use_bits = hrow != nullptr && hlen > 0;
    if (use_bits) {
#pragma unroll
        for (int w = item_ct1.get_local_id(2); w < kSplitBlockSpan / 32;
             w += item_ct1.get_local_range(2)) bits[w] = 0u;
        item_ct1.barrier(sycl::access::fence_space::local_space);
        for (int i = item_ct1.get_local_id(2); i < hlen;
             i += item_ct1.get_local_range(2)) {
            const int h = hrow[i];
            if (h >= 0 && h < n_vocab && h >= blo && h - blo < kSplitBlockSpan)
                dpct::atomic_fetch_or<
                    sycl::access::address_space::generic_space>(
                    &bits[(h - blo) >> 5], 1u << ((h - blo) & 31));
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // This warp's logits, penalised: `apply_penalties` with a zero count returns the logit unchanged, so only the
    // bitmap's hits go through it.  Past the vocabulary: -inf, which no round picks.
    const int lo = blo + warp * kSplitWarpSpan;
    float s[kSplitPerLane];
#pragma unroll
    for (int j = 0; j < kSplitPerLane; ++j) {
        const int v = lo + 32 * j + lane;
        s[j] = v < n_vocab ? l[v] : sycl::bit_cast<float, int>(0xff800000);
    }
    if (use_bits) {
#pragma unroll
        for (int j = 0; j < kSplitPerLane; ++j) {
            const int v = lo + 32 * j + lane, b = v - blo;
            if (v < n_vocab && (bits[b >> 5] & (1u << (b & 31))))
                s[j] = apply_penalties(s[j], history_count(hrow, hlen, v), p);
        }
    }

    auto &wl = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::int2[kSplitWarps][kSelMax]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    float prev_v = sycl::bit_cast<float>(
        0x7f800000); // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    int i = 0;
    for (; i < k; ++i) {
        // two chains (even and odd j), each walked in ascending id with a strict `>`, so each keeps its first in
        // the order; `take_first` then orders the two
        float b0 = sycl::bit_cast<float, int>(0xff800000),
              b1 = sycl::bit_cast<float, int>(0xff800000);
        int i0 = n_vocab, i1 = n_vocab;
#pragma unroll
        for (int j = 0; j < kSplitPerLane; j += 2) {
            const int v0 = lo + 32 * j + lane, v1 = v0 + 32;
            const float x0 = s[j], x1 = s[j + 1];
            if ((x0 < prev_v || (x0 == prev_v && v0 > prev_i)) && x0 > b0) { b0 = x0; i0 = v0; }
            if ((x1 < prev_v || (x1 == prev_v && v1 > prev_i)) && x1 > b1) { b1 = x1; i1 = v1; }
        }
        take_first(b0, i0, b1, i1);
        warp_first(b0, i0);
        if (i0 >= n_vocab) break;                // the same in every lane: nothing left in these 1,024 logits
        if (lane == 0) wl[warp][i] = sycl::int2(i0, sycl::bit_cast<int>(b0));
        prev_v = b0;
        prev_i = i0;
    }
#pragma unroll
    for (int r = i + lane; r < k; r += 32)
        wl[warp][r] = sycl::int2(n_vocab, (int)0xff800000u); // sentinels
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (warp == 0) {
        sycl::int2 *dst =
            cand + ((size_t)t * n_blocks + item_ct1.get_group(2)) * k;
        warp_merge_lists(&wl[0][0], kSplitWarps, kSelMax, k, n_vocab, [&](int r, float v, int id) {
            if (lane == 0) dst[r] = sycl::int2(id, sycl::bit_cast<int>(v));
        });
    }
}

/// **SPLIT STAGE 2: ONE WARP PER ROW MERGES THE BLOCK LISTS, THEN RUNS THE TAIL.**  The row's lists (at most 64 x
/// 64 entries, 32 KB) are copied to shared memory, merged into the row's top_k list, and `sampled_tail_warp` picks.
__dpct_inline__ void
sampler_split_merge_kernel(const sycl::int2 *__restrict__ cand, int n_blocks,
                           int n_vocab, const SamplerParams p, int k,
                           int *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
    const int lane = (int)item_ct1.get_local_id(2);
    auto &lists = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::int2[kSplitMaxBlocks * kSelMax]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sel_ids =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[kSelMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sel_logit =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kSelMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &ex =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<strata::samp_acc_t[kSelMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const sycl::int2 *src = cand + (size_t)t * n_blocks * k;
#pragma unroll
    for (int e = lane; e < n_blocks * k; e += 32) lists[e] = src[e];
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    warp_merge_lists(lists, n_blocks, k, k, n_vocab, [&](int i, float v, int id) {
        if (lane == 0) { sel_ids[i] = id < n_vocab ? id : 0; sel_logit[i] = v; }
    });
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    sampled_tail_warp(sel_ids, sel_logit, k, p, t, out, ex);
}

// ---- COUPLED DRAFT SAMPLING (include/strata/core/coupled_draft.hpp): the MTP draft layer samples its draft with the
// target's chain and the target's Philox draw.  Everything that varies per request or per round - the chain's
// parameters, the seed, the counter (from the cell's step record), the penalty history - is read from DEVICE memory:
// these kernels are captured into the drafter's round/step graphs.  One row, `nv` logits: the draft head's
// vocabulary subset (rt/draft_vocab.bin) or the whole vocabulary; `sub_to_id` maps a subset index to its token id.

/// The round's inputs: the request's SamplerParams and the history base (the last h slots before `cap`), from
/// mapped host memory into the device copies the chain's kernels read.
__dpct_inline__ void coupled_stage_kernel(const SamplerParams *__restrict__ mp,
                                          const int *__restrict__ mh,
                                          SamplerParams *__restrict__ dp,
                                          int *__restrict__ ring, int cap) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const volatile int *s = (const volatile int *)mp;
    int* d = (int*) dp;
#pragma unroll
    for (int i = item_ct1.get_local_id(2);
         i < (int)(sizeof(SamplerParams) / sizeof(int));
         i += item_ct1.get_local_range(2)) d[i] = s[i];
    const int h = strata::core::coupled_hist_len(((const volatile SamplerParams*) mp)->penalty_last_n, cap);
    const volatile int* vh = (const volatile int*) mh;
#pragma unroll
    for (int i = cap - h + (int)item_ct1.get_local_id(2); i < cap;
         i += item_ct1.get_local_range(2)) ring[i] = vh[i];
}

/// The penalties, applied in place to the draft logits before the selection - the target applies the same
/// `apply_penalties` to the same token with the same count over the same window before its own.  Draft j's window
/// is the ring's [cap + j - h, cap + j): the base plus drafts 0 .. j-1.  A history token outside the draft head's
/// subset has no logit here.  Each distinct token is penalised once: the entry whose `atomicOr` sets its bit does it.
__dpct_inline__ void coupled_penalize_kernel(
    float *__restrict__ logits, int nv, const int *__restrict__ id_to_sub,
    int id_vocab, const SamplerParams *__restrict__ dp,
    const int *__restrict__ ring, int cap, int j, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const SamplerParams p = *dp;
    const int h = strata::core::coupled_hist_len(p.penalty_last_n, cap);
    if (h <= 0) return;
    const int* hrow = ring + strata::core::coupled_hist_start(cap, j, h);
    auto seen = (unsigned int *)dpct_local;
    const int words = (nv + 31) / 32;
#pragma unroll
    for (int w = item_ct1.get_local_id(2); w < words;
         w += item_ct1.get_local_range(2)) seen[w] = 0u;
    item_ct1.barrier();
    for (int i = item_ct1.get_local_id(2); i < h;
         i += item_ct1.get_local_range(2)) {
        const int v = hrow[i];
        if (v < 0 || v >= id_vocab) continue;
        const int s = id_to_sub != nullptr ? id_to_sub[v] : v;
        if (s < 0 || s >= nv) continue;
        const unsigned bit = 1u << (s & 31);
        if (dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                &seen[s >> 5], bit) &
            bit) continue;
        logits[s] = apply_penalties(logits[s], history_count(hrow, h, v), p);
    }
}

/// The merge of `sampler_split_merge_kernel` (lists of `kpart` entries, the request's top_k taken from them), then
/// `sampled_tail_warp` with the counter of the row that will verify this draft.  Lane 0 maps the pick to its token
/// id, writes it and its probability, and appends it to the ring for the next draft's penalty window.
__dpct_inline__ void coupled_merge_kernel(
    const sycl::int2 *__restrict__ cand, int n_blocks, int nv, int kpart,
    const SamplerParams *__restrict__ dp, const int *__restrict__ step_rec,
    const int *__restrict__ sub_to_id, int *__restrict__ ring, int cap, int j,
    int *__restrict__ out_id, float *__restrict__ out_prob) {
    const int lane =
        (int)sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
            2);
    SamplerParams p = *dp;
    p.counter = strata::core::coupled_draft_counter((int64_t) step_rec[0]);
    const int k = sampled_k(p.top_k, nv);    // <= kpart: the first k of a union lie in the first k of each list
    auto &lists = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        sycl::int2[kSplitMaxBlocks * kSelMax]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sel_ids =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[kSelMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sel_logit =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kSelMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &ex =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<strata::samp_acc_t[kSelMax]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &pick = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[1]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &prob = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
#pragma unroll
    for (int e = lane; e < n_blocks * kpart; e += 32) lists[e] = cand[e];
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    warp_merge_lists(lists, n_blocks, kpart, k, nv, [&](int i, float v, int id) {
        if (lane == 0) { sel_ids[i] = id < nv ? id : 0; sel_logit[i] = v; }
    });
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    if (p.greedy || p.temperature <= 0.0f) {   // never launched for greedy requests; the argmax, defensively
        if (lane == 0) { pick[0] = sel_ids[0]; prob[0] = 1.0f; }
    } else {
        sampled_tail_warp<true>(sel_ids, sel_logit, k, p, 0, pick, ex, prob);
    }
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    if (lane == 0) {
        const int s = pick[0];
        const int id = sub_to_id != nullptr ? sub_to_id[s] : s;
        *out_id = id;
        *out_prob = prob[0];
        ring[cap + j] = id;
    }
}

}  // namespace

#include "sampler.kern.inc"

}  // namespace strata::kernels
