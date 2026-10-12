// src/kernels/cuda/s2_expert_grouped.host.cpp - the host side of s2_expert_grouped.dp.cpp: path selection, the launchers.
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {

#include "s2_expert_grouped.kern.inc"

namespace {

constexpr int H = 2560;
constexpr int FF = 640;
constexpr int GU_ROWS = 32;                      // gate/up rows per block: 4 per warp
constexpr int D_ROWS = 64;                       // down rows per block: 8 per warp

constexpr int THREADS = 256;

void check(const char* who, void* stream) {
    // Deliberately NOT synchronising for a non-null stream: this is called once per layer from a captured
    // graph's worth of work, and `finish()`'s null-stream sync in `s_gemv.cu` is the pattern that made a whole
    // round of measurements the driver's cost instead of the kernel's (Memory/ERRORS.md RC-7).
    (void) stream;
}

// The previous kernels stay selectable for A/B - `STRATA_OLD_GROUPED=1` in the environment, or
// `moe_grouped_select_old` (the parity test runs both in one process).  The environment is read once, on first use;
// the choice is made at each launch, so a captured graph keeps the kernels it was captured with.
std::atomic<int> g_select_old{-1};
// Which kernels the last launch of an entry point used: 1 = new, 0 = previous, -1 = none since the last query.
std::atomic<int> g_last_path{-1};

bool old_kernels() {
    const int s = g_select_old.load(std::memory_order_relaxed);
    if (s >= 0) return s != 0;
    static const bool env = [] {
        const char* e = std::getenv("STRATA_OLD_GROUPED");
        return e != nullptr && e[0] == '1';
    }();
    return env;
}

// `STRATA_GROUPED_PAIR_MIN_HITS=N`: the per-hit path keeps the previous one-warp-per-row kernels below N hits of
// capacity.  The new per-hit kernels launch half the warps (two rows each), so one hit's gate/up is 80 blocks - fewer
// than an RTX 5090's SMs - and whether that costs time at one to three hits is what `--bench` measures.  Bitwise
// the same either way; default 0 (always the new kernels).  Ignored while `moe_grouped_select_old` forces a choice.
long long pair_min_hits() {
    if (g_select_old.load(std::memory_order_relaxed) >= 0) return 0;
    static const long long n = [] {
        const char* e = std::getenv("STRATA_GROUPED_PAIR_MIN_HITS");
        return e != nullptr ? std::atoll(e) : 0LL;
    }();
    return n;
}

// The new kernels read the activations as aligned words (`load_x_chunk`), so they need 4-byte aligned activation
// rows - the input's and the intermediate's in `scratch`; the per-hit ones also read a blob's codes as uint2, which
// needs an 8-byte aligned arena and slot size (the grouped kernels above already did).  Anything else keeps the
// previous kernels rather than issuing a misaligned load.
bool new_grouped(const void* x_q8_0, const void* scratch) {
    const bool fast = !old_kernels() && ((uintptr_t) x_q8_0 & 3) == 0 && ((uintptr_t) scratch & 3) == 0;
    g_last_path.store(fast ? 1 : 0, std::memory_order_relaxed);
    return fast;
}

bool new_hit(const void* blob_base, long long blob_bytes, const void* x_q8_0, const void* scratch, long long cap) {
    const bool fast = new_grouped(x_q8_0, scratch) && ((uintptr_t) blob_base & 7) == 0 && (blob_bytes & 7) == 0 &&
                      cap >= pair_min_hits();
    g_last_path.store(fast ? 1 : 0, std::memory_order_relaxed);
    return fast;
}

// The per-hit path's two projections, previous or new kernels (one warp per row, or per pair of rows).
void launch_hit_gu(bool fast, const uint8_t *blob_base,
                   const int32_t *slot_index, long long blob_bytes,
                   const uint8_t *x_q8_0, const float *x_scales, float *gate_up,
                   long long cap, const int32_t *d_count,
                   const int32_t *dst_index, int tok_div, ze::Stream* cs) {
    const int warps = THREADS / 32;
    if (fast) {
        const long long pairs = cap * (long long) FF;
        k_s2_expert_grouped_gu_pair(cs, {(unsigned)((pairs + warps - 1) / warps), 1, 1}, {THREADS, 1, 1}, 0, blob_base,
                                    slot_index, blob_bytes, x_q8_0, x_scales, gate_up, (int)cap, d_count, dst_index,
                                    tok_div);
    } else {
        const long long rows = cap * 2LL * FF;
        k_s2_expert_grouped_gu(cs, {(unsigned)((rows + warps - 1) / warps), 1, 1}, {THREADS, 1, 1}, 0, blob_base,
                               slot_index, blob_bytes, x_q8_0, x_scales, gate_up, (int)cap, d_count, dst_index,
                               tok_div);
    }
}

void launch_hit_down(bool fast, const uint8_t *blob_base,
                     const int32_t *slot_index, const int32_t *dst_index,
                     long long blob_bytes, const uint8_t *h_q8_0,
                     const float *h_scales, float *out, long long cap,
                     const int32_t *d_count, ze::Stream* cs) {
    const int warps = THREADS / 32;
    if (fast) {
        const long long pairs = cap * (long long) (H / 2);
        k_s2_expert_grouped_down_pair(cs, {(unsigned)((pairs + warps - 1) / warps), 1, 1}, {THREADS, 1, 1}, 0,
                                      blob_base, slot_index, dst_index, blob_bytes, h_q8_0, h_scales, out, (int)cap,
                                      d_count);
    } else {
        const long long rows = cap * (long long) H;
        k_s2_expert_grouped_down(cs, {(unsigned)((rows + warps - 1) / warps), 1, 1}, {THREADS, 1, 1}, 0, blob_base,
                                 slot_index, dst_index, blob_bytes, h_q8_0, h_scales, out, (int)cap, d_count);
    }
}

}  // namespace

void moe_grouped_select_old(int old) { g_select_old.store(old < 0 ? -1 : (old != 0 ? 1 : 0)); }

int moe_grouped_last_path() { return g_last_path.exchange(-1, std::memory_order_relaxed); }

uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    // R4.2h: the fp32 scales for the INTERMEDIATE's own quantization, one per 32-element chunk per hit.
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) +
           ((xh + 15) & ~15ull);
}

void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out,
                        void* stream, const float* x_scales) {
    if (n_hits <= 0) return;
    ze::Stream* cs = ze::stream_of(stream);
    const bool fast = new_hit(blob_base, blob_bytes, x_q8_0, scratch, n_hits);

    const uint64_t gu_bytes = ((uint64_t) n_hits * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);

    // 1. gate + up, one launch for every row of every hit.
    {
        launch_hit_gu(fast, blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, n_hits, nullptr, nullptr,
                      0, cs);
        check("moe_hit_grouped_s2/gu", stream);
    }
    // 2. silu(gate) * up.
    {
        const long long pairs = n_hits * (long long) FF;
        const unsigned blocks = (unsigned) ((pairs + THREADS - 1) / THREADS);
        k_s2_expert_grouped_swiglu(cs, {blocks, 1, 1}, {THREADS, 1, 1}, 0, gate_up, pairs);
        check("moe_hit_grouped_s2/swiglu", stream);
    }
    // 3. the intermediate's own contract, which is `ggml_mul_mat`'s rule and NOT a choice: the down weight is
    //    Q2_0, whose `vec_dot_type` is Q8_0.  `gate_up` holds the pairs; the product went into the first half.
    //    **R4.2h: the CPU quantizes this intermediate into `a2` with fp32 scales too** (`expert.cpp:232`), so
    //    when the caller supplies `x_scales` the intermediate gets the CPU's contract as well - otherwise the
    //    down projection would keep the very disagreement the gate/up projection just had removed.
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, n_hits * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, n_hits * (int64_t) FF, stream);
    // 4. down.
    {
        launch_hit_down(fast, blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                        x_scales != nullptr ? h_scales : nullptr, out, n_hits, nullptr, cs);
        check("moe_hit_grouped_s2/down", stream);
    }
}

void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst,
                    int32_t* count, void* stream) {
    if (k < 1 || k > 32) { std::fprintf(stderr, "moe_hit_select: k must be 1..32\n"); std::exit(1); }
    k_s2_expert_grouped_hit_select(ze::stream_of(stream), {1, 1, 1}, {32, 1, 1}, 0, ids, res_row, k, n_expert, slot,
                                   dst, count);
    check("moe_hit_select", stream);
}

void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                            const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                            void* scratch, float* out, void* stream, const float* x_scales) {
    if (cap <= 0) return;
    ze::Stream* cs = ze::stream_of(stream);
    const bool fast = new_hit(blob_base, blob_bytes, x_q8_0, scratch, cap);
    const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    {
        launch_hit_gu(fast, blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, cap, d_count, nullptr, 0,
                      cs);
        check("moe_hit_grouped_s2_dev/gu", stream);
    }
    {
        const long long pairs = cap * (long long) FF;
        k_s2_expert_grouped_swiglu(cs, {(unsigned)((pairs + THREADS - 1) / THREADS), 1, 1}, {THREADS, 1, 1}, 0, gate_up,
                                   pairs);
        check("moe_hit_grouped_s2_dev/swiglu", stream);
    }
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, cap * (int64_t) FF, stream);
    {
        launch_hit_down(fast, blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                        x_scales != nullptr ? h_scales : nullptr, out, cap, d_count, cs);
        check("moe_hit_grouped_s2_dev/down", stream);
    }
}

void moe_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot, int32_t* dst,
                          int32_t* count, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_hit_select_multi: n must be 1..128\n"); std::exit(1); }
    k_s2_expert_grouped_hit_select_multi(ze::stream_of(stream), {1, 1, 1}, {128, 1, 1}, 0, ids, res_row, n, n_expert,
                                         slot, dst, count);
    check("moe_hit_select_multi", stream);
}

void moe_hit_grouped_s2_multi(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                              const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                              const float* x_scales, int k_per_token, void* scratch, float* out, void* stream) {
    if (cap <= 0) return;
    ze::Stream* cs = ze::stream_of(stream);
    const bool fast = new_hit(blob_base, blob_bytes, x_q8_0, scratch, cap);
    const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    {
        launch_hit_gu(fast, blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, cap, d_count, dst_index,
                      k_per_token, cs);
        check("moe_hit_grouped_s2_multi/gu", stream);
    }
    {
        const long long pairs = cap * (long long) FF;
        k_s2_expert_grouped_swiglu(cs, {(unsigned)((pairs + THREADS - 1) / THREADS), 1, 1}, {THREADS, 1, 1}, 0, gate_up,
                                   pairs);
        check("moe_hit_grouped_s2_multi/swiglu", stream);
    }
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, cap * (int64_t) FF, stream);
    {
        launch_hit_down(fast, blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                        x_scales != nullptr ? h_scales : nullptr, out, cap, d_count, cs);
        check("moe_hit_grouped_s2_multi/down", stream);
    }
}

void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                        int32_t* ent_tok, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_group_resident: n must be 1..128\n"); std::exit(1); }
    k_s2_expert_grouped_group_resident(ze::stream_of(stream), {1, 1, 1}, {128, 1, 1}, 0, ids, n, k_per_tok, base,
                                       (long long)blob, grp_ptr, grp_start, counts, ent_dst, ent_tok);
    check("moe_group_resident", stream);
}

void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                    const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    ze::Stream* cs = ze::stream_of(stream);
    const uint64_t gu_bytes = ((uint64_t) cap_entries * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap_entries * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    const bool fast = new_grouped(x_q8_0, scratch);
    {
        const ze::Dim3 grid{(unsigned)(2 * FF / GU_ROWS), (unsigned)cap_groups, 1};
        if (fast)
            k_s2_expert_grouped_gu_grouped_t(cs, grid, {256, 1, 1}, 0, grp_ptr, grp_start, n_groups, ent_tok, x_q8_0,
                                             x_scales, gate_up, (int)cap_entries);
        else
            k_s2_expert_grouped_gu_grouped(cs, grid, {256, 1, 1}, 0, grp_ptr, grp_start, n_groups, ent_tok, x_q8_0,
                                           x_scales, gate_up, (int)cap_entries);
        check("moe_grouped_s2/gu", stream);
    }
    // #783 PR-k (stuchapin909): SwiGLU and the scaled Q8_0 quantize of its result in one kernel (gate_up still gets the
    // activation written back, so the parity checks read the same buffer); STRATA_S2_SWIGLU_Q8=0 keeps the two launches
    static const bool swiglu_q8 = [] {
        const char* e = std::getenv("STRATA_S2_SWIGLU_Q8");
        return e == nullptr || e[0] != '0';
    }();
    static_assert(FF % 32 == 0, "the fused SwiGLU + Q8_0 kernel works in whole 32-value blocks");
    if (fast && x_scales != nullptr && swiglu_q8) {
        const long long pairs = cap_entries * (long long) FF;
        const long long n_blocks = pairs >> 5;
        const int warps = THREADS / 32;
        k_s2_expert_grouped_swiglu_quantize_q8_0_scaled(cs, {(unsigned)((n_blocks + warps - 1) / warps), 1, 1},
                                                        {THREADS, 1, 1}, 0, gate_up, pairs, h_q8_0, h_scales);
        check("moe_grouped_s2/swiglu_q8", stream);
    } else {
        const long long pairs = cap_entries * (long long) FF;
        k_s2_expert_grouped_swiglu(cs, {(unsigned)((pairs + THREADS - 1) / THREADS), 1, 1}, {THREADS, 1, 1}, 0, gate_up,
                                   pairs);
        check("moe_grouped_s2/swiglu", stream);
        if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap_entries * (int64_t) FF, stream);
        else quantize_q8_0(gate_up, h_q8_0, cap_entries * (int64_t) FF, stream);
    }
    {
        const ze::Dim3 grid{(unsigned)(H / D_ROWS), (unsigned)cap_groups, 1};
        const float* hs = x_scales != nullptr ? h_scales : nullptr;
        if (fast)
            k_s2_expert_grouped_down_grouped_t(cs, grid, {256, 1, 1}, 0, grp_ptr, grp_start, n_groups, ent_dst, h_q8_0,
                                               hs, out);
        else
            k_s2_expert_grouped_down_grouped(cs, grid, {256, 1, 1}, 0, grp_ptr, grp_start, n_groups, ent_dst, h_q8_0,
                                             hs, out);
        check("moe_grouped_s2/down", stream);
    }
}

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    const ze::Dim3 grid{
        (unsigned)((n_embd + 255) / 256 < 8 ? (n_embd + 255) / 256 : 8),
        (unsigned)cap, 1};
    k_s2_expert_grouped_add_hits(ze::stream_of(stream), grid, {256, 1, 1}, 0, parts, hit_out, dst, count, (int)n_embd);
    check("moe_hit_add", stream);
}

void moe_hit_grouped_s2_cpu_order(const uint8_t *blob_base,
                                  const int32_t *slot_index,
                                  const int32_t *dst_index, int64_t n_hits,
                                  int64_t blob_bytes, const uint8_t *x_q8_0,
                                  void *scratch, float *out, void *stream,
                                  const float *x_scales,
                                  float *gate_up_trace) {
    if (n_hits <= 0) return;
    if (x_scales == nullptr) {
        std::fprintf(stderr, "moe_hit_grouped_s2_cpu_order requires fp32 activation scales\n");
        std::exit(1);
    }
    ze::Stream* cs = ze::stream_of(stream);
    const uint64_t gu_bytes = ((uint64_t) n_hits * 2 * FF * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (FF / 32) * 34 + 15) & ~15ull;
    const uint64_t scale_bytes = ((uint64_t) n_hits * (FF / 32) * 4 + 15) & ~15ull;
    float* gu = (float*) scratch;
    uint8_t* hq = (uint8_t*) scratch + gu_bytes;
    float* hs = (float*) (hq + q8_bytes);
    float* hh = (float*) ((uint8_t*) hs + scale_bytes);
    float* xh = (float*) ((uint8_t*) hh + scale_bytes);
    k_s2_expert_grouped_activation_correction(cs, {(H / 32 + THREADS - 1) / THREADS, 1, 1}, {THREADS, 1, 1}, 0, x_q8_0,
                                              x_scales, xh, H / 32);
    check("cpu_order/input_correction", stream);
    const int rows_per_block = THREADS / 8;
    k_s2_expert_grouped_cpu_order_gu(cs, {(unsigned)((n_hits * 2 * FF + rows_per_block - 1) / rows_per_block), 1, 1},
                                     {THREADS, 1, 1}, 0, blob_base, slot_index, dst_index, blob_bytes, x_q8_0,
                                     x_scales, xh, gu, (int)n_hits);
    check("cpu_order/gate_up", stream);
    if (gate_up_trace != nullptr) cs->memcpy(gate_up_trace, gu, (size_t)n_hits * 2 * FF * sizeof(float));
    k_s2_expert_grouped_cpu_order_swiglu(cs, {(unsigned)((n_hits * FF + THREADS - 1) / THREADS), 1, 1},
                                         {THREADS, 1, 1}, 0, gu, (int)n_hits * FF);
    check("cpu_order/swiglu", stream);
    k_s2_expert_grouped_cpu_order_quantize(cs, {(unsigned)((n_hits * (FF / 32) + THREADS - 1) / THREADS), 1, 1},
                                           {THREADS, 1, 1}, 0, gu, hq, hs, hh, (int)n_hits * (FF / 32));
    check("cpu_order/intermediate_quantize", stream);
    k_s2_expert_grouped_cpu_order_down(cs, {(unsigned)((n_hits * H + rows_per_block - 1) / rows_per_block), 1, 1},
                                       {THREADS, 1, 1}, 0, blob_base, slot_index, dst_index, blob_bytes, hq, hs, hh,
                                       out, (int)n_hits);
    check("cpu_order/down", stream);
}

}  // namespace strata::kernels
