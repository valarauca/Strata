// src/kernels/cuda/fused_gr.host.cpp - the host side of fused_gr.dp.cpp: the launchers and the variants' self-test.
#include <mutex>
#include <map>
#include "strata/core/emulate.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/ze/kernel_abi.hpp"
#include "fused_gr_shared.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace strata::kernels {

#include "fused_gr.kern.inc"

namespace {

// GrMulti's pointers for the STRATA_ZE_CHECK_DEVICES launch check: each token's FusedGrArgs, then the scratch.
void launch_ptrs(const GrMulti& m, const ze::LaunchPtrs& c) {
    for (int t = 0; t < m.T && t < kFusedGrMaxT; ++t) launch_ptrs(m.a[t], c);
    c(m.xn);
    c(m.part);
    c(m.part2);
}

bool gr_norm_split() {   // default; STRATA_GR_NORM_SPLIT=0: one work-group per token
    static const bool v = std::getenv("STRATA_GR_NORM_SPLIT") == nullptr || std::atoi(std::getenv("STRATA_GR_NORM_SPLIT")) != 0;
    return v;
}

bool gr_down_sliced() {   // default (gr_bench: GR read 108.6 -> 76.5 us at 6 tokens); STRATA_GR_DOWN_SLICED=0: direct
    static const bool v = std::getenv("STRATA_GR_DOWN_SLICED") == nullptr || std::atoi(std::getenv("STRATA_GR_DOWN_SLICED")) != 0;
    return v;
}
// The scratch of stream `q`, keyed by its GPU as well as its address: a deleted capture stream's address can come
// back as a stream of another GPU, which must not inherit the first GPU's buffer.
float* stream_scratch(std::map<std::pair<int, strata::ze::Stream*>, float*>& bufs, std::mutex& mu, strata::ze::Stream* q,
                      size_t floats) {
    std::lock_guard<std::mutex> lk(mu);
    float*& p = bufs[{q->device(), q}];
    if (p == nullptr) p = (float*) strata::ze::malloc_device(floats * sizeof(float), q->device());
    return p;
}
float* slice_partials(strata::ze::Stream* q) {
    static std::mutex mu;
    static std::map<std::pair<int, strata::ze::Stream*>, float*> bufs;
    return stream_scratch(bufs, mu, q, (size_t) GRS_SLICES * kFusedGrMaxT * GRS_ROWS);
}
bool gr_down_direct() {   // default; STRATA_GR_DOWN_DIRECT=0: the tiled kernel (gr_bench: 13-19% slower at 3-6 tokens)
    static const bool v = std::getenv("STRATA_GR_DOWN_DIRECT") == nullptr || std::atoi(std::getenv("STRATA_GR_DOWN_DIRECT")) != 0;
    return v;
}

// The tokens a down kernel may carry in one launch on the current card, and the plain read's tile: the plain read
// stages n_tok * TILEV floats (1280 on sm_75, 2560 elsewhere), staged two tiles of n_tok * H_TILE.  The shared-memory
// opt-in is set here once per device (a per-DEVICE setting: a layer split runs these kernels on two cards).
int down_chunk(bool staged, int* tile_out) {
    static bool attr[64] = {};
    static int chunk[64] = {};   // tokens the plain down kernel may carry in one launch on this card
    static int chunk_staged[64] = {};  // the same for staged
    static int tile[64] = {};    // the plain down kernel's TILEV on this card (1280 on sm_75, 2560 elsewhere)
    int dev = 0;
    dev = strata::ze::current_device();
    if (dev < 0 || dev >= 64) {
        *tile_out = 2560;
        return kFusedGrMaxT;
    }
    if (!attr[dev]) {
        int optin = 0;
        optin = (int) strata::ze::device_info(dev).max_slm;
        optin = strata::smem_optin_of(optin);   // STRATA_EMULATE_CC (tests only)
        // the down kernel stages n_tok*TILEV floats of dynamic shared memory - 80 KB at the full 8 tokens of the
        // CUDA tile.  sm_75 gets the smaller tile: all eight tokens fit one 40 KiB launch there (no more slicing),
        // the TQ-5 prefetch holds half the registers, and the smaller blocks raise how many of the 41-block grid
        // share an SM.  Cards whose opt-in is still below that (or that report no opt-in at all) slice the tokens;
        // the down kernel's outputs (lo, inject_out) are strictly per-token, so the chunk boundaries are safe, and
        // the up kernel below still sees every token of the batch in one launch.
#if defined(__HIPCC__)
        const bool small_tile = true;   // all eight tokens fit gfx1100's 64 KiB LDS at this tile
#else
        int cc_maj = 0, cc_min = 0;
        const bool small_tile = strata::cc_major_of(cc_maj) * 10 + strata::cc_minor_of(cc_min) == 75;
#endif
        const int tv = small_tile ? 1280 : 2560;
        tile[dev] = tv;
        // The opt-in is a promise a pre-Volta card does not keep: an sm_60 answers
        // 65536 and accepts the cudaFuncSetAttribute for 61440 B, then fails the
        // LAUNCH with "invalid argument".  What such a card will launch is its
        // per-block limit, so the capacity comes from that below sm_70 - the same
        // tokens the "no opt-in" branch assumes, but taken from the attribute that is
        // actually enforced.
#if defined(__HIPCC__)
        const int usable = optin > 0 ? optin : 48 * 1024;
#else
        int cc = 0, per_block = 0;
        per_block = (int) strata::ze::device_info(dev).max_slm;   // SYCL: the work-group local memory
        cc = strata::cc_major_of(cc);
        const int usable = (cc >= 7 && optin > 0) ? optin : per_block;
#endif
        const int capacity = usable / (int) (tv * sizeof(float));
        chunk[dev] = capacity < 1 ? 1 : (capacity > kFusedGrMaxT ? kFusedGrMaxT : capacity);
        const int capacity_staged = usable / (int) (2 * H_TILE * sizeof(float));
        chunk_staged[dev] = capacity_staged < 1 ? 1 : (capacity_staged > kFusedGrMaxT ? kFusedGrMaxT : capacity_staged);
        attr[dev] = true;
    }
    *tile_out = tile[dev] ? tile[dev] : 2560;
    return staged ? chunk_staged[dev] : chunk[dev];
}

// The multi read as `variant` (kHcPlain, kHcSplit or kHcStaged): the norm, the down projection in launches of as
// many tokens as fit the card, the up projection; the profile's stamps after the norm and after the down projection.
// kHcPlain is the default read exactly as before (never main's opt-in STRATA_GR_V3 path, which fused_gr_read_multi
// takes first).
void launch_multi(const GrMulti &m, int variant, strata::ze::Stream* st,
                  unsigned long long *stamp_buf, int stamp_i0) {
    const int n_tok = m.T;
    if (variant >= kHcSplit)
        k_fused_gr_norm_split(st, {(unsigned) n_tok, HC, 1}, {THREADS, 1, 1}, 0, m);
    else
        k_fused_gr_norm_multi(st, {(unsigned) n_tok, 1, 1}, {THREADS, 1, 1}, 0, m);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, (void*) st);
    const bool staged = variant >= kHcStaged;
    int tv = 2560;
    const int chunk_tok = down_chunk(staged, &tv);
    const size_t per_tok =
        (staged ? (size_t)2 * H_TILE : (size_t)tv) * sizeof(float);
    for (int c0 = 0; c0 < n_tok; c0 += chunk_tok) {
        const int ct = n_tok - c0 < chunk_tok ? n_tok - c0 : chunk_tok;
        GrMulti c{};
        if (ct == n_tok) {
            c = m;
        } else {
            c.xn = m.xn + (size_t) c0 * D;
            c.T = ct;
            for (int k = 0; k < ct; ++k) c.a[k] = m.a[c0 + k];
        }
        const size_t smem = (size_t) ct * per_tok;
        // #443, opt-in STRATA_GR_DOWN_MAX4=1: launches of up to 4 tokens hold 4 tokens' sums per thread instead of
        // kFusedGrMaxT - the same tile, block size, accumulation order and plain/split/staged path, so the same bits
        // (read once: this runs per layer when decode is not captured)
        static const bool max4_on = [] { const char* v = std::getenv("STRATA_GR_DOWN_MAX4"); return v && std::atoi(v) != 0; }();
        const bool max4 = max4_on && ct <= 4;
        const strata::ze::Dim3 grid{DOWN_BLOCKS + 1, 1, 1}, block{THREADS, 1, 1};
        if (staged) {
            if (max4)
                k_fused_gr_down_staged_max4(st, grid, block, smem, c);
            else
                k_fused_gr_down_staged(st, grid, block, smem, c);
        } else if (tv == 1280) {
            if (max4)
                k_fused_gr_down_multi_1280_max4(st, grid, block, smem, c);
            else
                k_fused_gr_down_multi_1280(st, grid, block, smem, c);
        } else {
            if (max4)
                k_fused_gr_down_multi_2560_max4(st, grid, block, smem, c);
            else
                k_fused_gr_down_multi_2560(st, grid, block, smem, c);
        }
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, (void*) st);
    k_fused_gr_up_multi(st, {UPM_BLOCKS, 1, 1}, {THREADS, 1, 1}, 0, m);
}

/// STRATA_HC_SPLIT: unset or 2 = the newest the check accepts (staged), 1 = at most split, 0 = the plain read
int env_variant() {
    const char* e = std::getenv("STRATA_HC_SPLIT");
    if (e == nullptr || e[0] == '\0') return kHcStaged;
    if (e[0] == '0') return kHcPlain;
    if (e[0] == '1') return kHcSplit;
    return kHcStaged;
}

// per device: the variant `fused_gr_check` chose (0 = not checked yet)
std::atomic<int> g_variant[64];

}  // namespace

// one partials buffer per queue (the verifier's and the drafter's launches never share a queue; within a queue the
// launches are ordered, and a graph capture records the pointer)
static float* down_partials(strata::ze::Stream* q) {
    static std::mutex mu;
    static std::map<std::pair<int, strata::ze::Stream*>, float*> bufs;
    return stream_scratch(bufs, mu, q, (size_t) DOWN_SPLIT * kFusedGrMaxT * LR);
}

bool fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream, unsigned long long* stamp_buf,
                         int stamp_i0) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed || (x.w_inject && !x.inject_out) ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    m.xn = xn_scratch;
    m.T = n_tok;
    strata::ze::Stream* st = strata::ze::stream_of(stream);
    // STRATA_GR_V3=1: the two-kernel read above (another summation order - opt-in)
    static const bool v3 = [] { const char* v = std::getenv("STRATA_GR_V3"); return v != nullptr && std::atoi(v) != 0; }();
    static int split3[64] = {};   // per device: 0 = not decided yet, 1 / 2 = column halves S, -1 = does not fit
    int dev3 = 0;
    if (v3) {
        dev3 = strata::ze::current_device();
        if (dev3 >= 0 && dev3 < 64 && split3[dev3] == 0) {
            int optin = 0;
            // SYCL port: the device's local memory is the limit (no per-kernel opt-in to raise)
            optin = (int) strata::ze::device_info(dev3).max_slm;
            const int limit = optin > 0 ? optin : 48 * 1024;
            const int need1 = (int) (kFusedGrMaxT * N * sizeof(float)), need2 = need1 / 2;
            int split = -1;
            if (need1 <= limit)
                split = 1;
            else if (need2 <= limit)
                // #375 (kenh0u): the S = 2 split (a 64 KB opt-in card: Turing) disagrees with itself in gr_parity
                // (graph replay vs direct call) - such a card keeps the default read until that split is fixed
                std::fprintf(stderr, "strata: STRATA_GR_V3=1 needs the two-half split on this card, which fails its "
                                     "checks (#375): the default read is used\n");
            split3[dev3] = split;
        }
    }
    const int split = v3 && dev3 >= 0 && dev3 < 64 ? split3[dev3] : -1;
    if (split > 0) {   // 2 kernels; scratch = partials + sums of squares
        float* part = xn_scratch;
        float* ssg = xn_scratch + (size_t) n_tok * HC * 2 * PR;   // room for S = 2
        const size_t sm = (size_t)n_tok * (N / split) * sizeof(float);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
        if (split == 2)
            k_fused_gr_down_v3_s2(st, {LR / WARPS + 1, HC * 2, 1}, {THREADS, 1, 1}, sm, m, part, ssg);
        else
            k_fused_gr_down_v3_s1(st, {LR / WARPS + 1, HC, 1}, {THREADS, 1, 1}, sm, m, part, ssg);
        if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
        if (split == 2)
            k_fused_gr_up_v3_s2(st, {UPM_BLOCKS, 1, 1}, {THREADS, 1, 1}, 0, m, part, ssg);
        else
            k_fused_gr_up_v3_s1(st, {UPM_BLOCKS, 1, 1}, {THREADS, 1, 1}, 0, m, part, ssg);
        return false;   // SYCL port: never writes the q8_1 images (STRATA_QFUSE is not supported here)
    }
    m.part = down_partials(st);
    if (gr_norm_split())
        k_fused_gr_norm_split_port(st, {(unsigned) (n_tok * HC), 1, 1}, {THREADS, 1, 1}, 0, m);
    else
        k_fused_gr_norm_multi(st, {(unsigned) n_tok, 1, 1}, {THREADS, 1, 1}, 0, m);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
    // the local-memory capacity is a per-DEVICE property: once per device (a layer split runs this on two cards).
    // SYCL port: upstream picks TILEV 1280 on sm_75 and takes the per-block limit below sm_70 (CUDA compute
    // capabilities, cudaFuncSetAttribute opt-ins); here the tile is the port's TILE and the device's local memory is
    // the only limit. The tiled kernel is the fallback (STRATA_GR_DOWN_SLICED=0 STRATA_GR_DOWN_DIRECT=0); the
    // default is the sliced kernel below.
    static bool attr[64] = {};
    static int chunk[64] = {};   // tokens the tiled down kernel may carry in one launch on this card
    int dev = 0;
    dev = strata::ze::current_device();
    if (dev >= 0 && dev < 64 && !attr[dev]) {
        const int optin = (int) strata::ze::device_info(dev).max_slm;
        const int capacity = (optin > 0 ? optin : 48 * 1024) / (int) (TILE * sizeof(float));
        chunk[dev] = capacity < 1 ? 1 : (capacity > kFusedGrMaxT ? kFusedGrMaxT : capacity);
        attr[dev] = true;
    }
    const int chunk_tok = (dev >= 0 && dev < 64 && chunk[dev]) ? chunk[dev] : kFusedGrMaxT;
    const auto& down_tiled = TILE == 1280 ? k_fused_gr_down_multi_1280 : k_fused_gr_down_multi_2560;
    if (gr_down_sliced()) {
        m.part2 = slice_partials(st);
        m.nsplit = 1;
        k_fused_gr_down_sliced(st, {GRS_SLICES, 1, 1}, {THREADS, 1, 1}, (size_t) kFusedGrMaxT * GRS_COLS * sizeof(float), m);
        const unsigned nred = unsigned((n_tok * GRS_ROWS + 127) / 128);
        k_fused_gr_down_reduce(st, {nred, 1, 1}, {128, 1, 1}, 0, m);
    } else if (gr_down_direct()) {
        k_fused_gr_down_multi_direct(st, {DOWN_BLOCKS * DOWN_SPLIT + 1, 1, 1}, {THREADS, 1, 1}, 0, m);
    } else if (chunk_tok >= n_tok) {
        down_tiled(st, {DOWN_BLOCKS * DOWN_SPLIT + 1, 1, 1}, {THREADS, 1, 1}, (size_t) n_tok * TILE * sizeof(float), m);
    } else {
        for (int c0 = 0; c0 < n_tok; c0 += chunk_tok) {
            const int ct = n_tok - c0 < chunk_tok ? n_tok - c0 : chunk_tok;
            GrMulti c{};
            c.xn = xn_scratch + (size_t) c0 * D;
            c.T = ct;
            for (int k = 0; k < ct; ++k) c.a[k] = a[c0 + k];
            c.part = m.part;
            down_tiled(st, {DOWN_BLOCKS * DOWN_SPLIT + 1, 1, 1}, {THREADS, 1, 1}, (size_t) ct * TILE * sizeof(float), c);
        }
    }
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
    k_fused_gr_up_multi(st, {UPM_BLOCKS, 1, 1}, {THREADS, 1, 1}, 0, m);
    return false;   // SYCL port: the q8_1 images (S26 STRATA_QFUSE) and the Q8_0 reads (S23 STRATA_HC_Q8) are not built; the caller quantizes
}

// the bench only (AMD latency-hidden kernels): nothing to switch on the SYCL port
void fused_gr_set_fast(int) {}

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    strata::ze::Stream* st = strata::ze::stream_of(stream);
    // SYCL port: the multi-token read's sliced down kernel and per-(token, stream) norm sum in their own orders, so a
    // window's token would differ in its last bits from this kernel's. The header promises "every token's outputs
    // are bitwise fused_gr_read(a[t])" (gr_parity checks it): with those paths on, the single read IS the multi read
    // of one token (each token's sums run in the same order whatever the window size).
    if (gr_down_sliced() || gr_norm_split() || gr_down_direct()) {
        static std::mutex mu;
        static std::map<std::pair<int, strata::ze::Stream*>, float*> xn;
        fused_gr_read_multi(&a, 1, stream_scratch(xn, mu, st, (size_t) D), stream);
        return;
    }
    k_fused_gr_down(st, {DOWN_BLOCKS + 1, 1, 1}, {THREADS, 1, 1}, 0, a);
    k_fused_gr_up(st, {UP_BLOCKS, 1, 1}, {THREADS, 1, 1}, 0, a);
}


namespace {

/// The plain read, split, staged and (for one token) the single-token read on random bf16 weights and random inputs,
/// 1..8 tokens, with and without the pending write; every output of split and staged compared with the plain read's
/// bit for bit (and the plain read's with the single-token read's).  `why[v]` gets the first difference of variant
/// v; false if the check itself could not run.
bool fused_gr_selftest(bool ok_variant[4], std::string why[4]) {
    constexpr int TM = kFusedGrMaxT;
    constexpr int NV = 4;                             // sets: 0 = plain, 1 = split, 2 = staged, 3 = single-token
    std::mt19937 rng(20260930u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> h_down((size_t) LR * D), h_up((size_t) D * LR), h_inj((size_t) HC * D);
    std::vector<float> h_norm(D), h_R((size_t) TM * D), h_bo((size_t) TM * N), h_ip((size_t) TM * HC);
    for (auto& w : h_down) w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_up) w = bf16_from_f32(0.05f * nd(rng));
    for (auto& w : h_inj) w = bf16_from_f32(0.02f * nd(rng));
    for (auto& w : h_norm) w = 1.0f + 0.1f * nd(rng);
    for (auto& x : h_R) x = nd(rng);
    for (auto& x : h_bo) x = 0.5f * nd(rng);
    for (auto& x : h_ip) x = 2.0f * nd(rng);
    for (int v = 0; v < 4; ++v) { ok_variant[v] = v == 1; why[v].clear(); }

    const size_t n_set = (size_t) TM * (D + D + LR + HC + HC + N);   // R_out, xn, lo, rs, inject, mixed (floats)
    const size_t bytes = h_down.size() * 2 + h_up.size() * 2 + h_inj.size() * 2 +
                         (h_norm.size() + h_R.size() + h_bo.size() + h_ip.size() + NV * n_set) * 4 + 32 * 256;
    uint8_t* base = (uint8_t*) strata::ze::malloc_device(bytes);
    if (base == nullptr) {
        why[0] = "no room for the check (" + std::to_string(bytes >> 20) + " MiB)";
        return false;
    }
    size_t off = 0;
    auto take = [&](size_t b) { void* p = base + off; off += (b + 255) / 256 * 256; return p; };
    uint16_t* d_down = (uint16_t*) take(h_down.size() * 2);
    uint16_t* d_up = (uint16_t*) take(h_up.size() * 2);
    uint16_t* d_inj = (uint16_t*) take(h_inj.size() * 2);
    float* d_norm = (float*) take(h_norm.size() * 4);
    float* d_R = (float*) take(h_R.size() * 4);
    float* d_bo = (float*) take(h_bo.size() * 4);
    float* d_ip = (float*) take(h_ip.size() * 4);
    struct Set { float *R_out, *xn, *lo, *rs, *inj, *mixed; } set[NV];
    for (Set& x : set) {
        x.R_out = (float*) take((size_t) TM * D * 4); x.xn = (float*) take((size_t) TM * D * 4);
        x.lo = (float*) take((size_t) TM * LR * 4); x.rs = (float*) take((size_t) TM * HC * 4);
        x.inj = (float*) take((size_t) TM * HC * 4); x.mixed = (float*) take((size_t) TM * N * 4);
    }
    strata::ze::Stream* st = &strata::ze::default_stream();
    bool ok = off <= bytes;
    if (ok) {
        st = new strata::ze::Stream();
        st->memcpy(d_down, h_down.data(), h_down.size() * 2);
        st->memcpy(d_up, h_up.data(), h_up.size() * 2);
        st->memcpy(d_inj, h_inj.data(), h_inj.size() * 2);
        st->memcpy(d_norm, h_norm.data(), h_norm.size() * 4);
        st->memcpy(d_R, h_R.data(), h_R.size() * 4);
        st->memcpy(d_bo, h_bo.data(), h_bo.size() * 4);
        st->memcpy(d_ip, h_ip.data(), h_ip.size() * 4);
    }
    if (!ok) why[0] = "setting up the check failed";
    std::vector<float> h1, h2;
    // true when equal; false with the first difference in `w`
    auto same = [&](const float* d1, const float* d2, size_t n, const char* what, int T, int apply, std::string& w) {
        h1.resize(n);
        h2.resize(n);
        st->memcpy(h1.data(), d1, n * 4);
        st->memcpy(h2.data(), d2, n * 4);
        st->sync();
        for (size_t i = 0; i < n; ++i) {
            uint32_t u1, u2;
            std::memcpy(&u1, &h1[i], 4);
            std::memcpy(&u2, &h2[i], 4);
            if (u1 != u2) {
                char buf[192];
                std::snprintf(buf, sizeof buf, "%s differs for %d token(s)%s at %zu (%08x, not %08x)", what, T,
                              apply ? " with the pending write" : "", i, u2, u1);
                w = buf;
                return false;
            }
        }
        return true;
    };
    auto all_same = [&](const Set& s1, const Set& s2, int T, int apply, std::string& w) {
        return same(s1.lo, s2.lo, (size_t) T * LR, "lo", T, apply, w) &&
               same(s1.rs, s2.rs, (size_t) T * HC, "rs", T, apply, w) &&
               same(s1.inj, s2.inj, (size_t) T * HC, "inject", T, apply, w) &&
               same(s1.mixed, s2.mixed, (size_t) T * N, "mixed", T, apply, w) &&
               same(s1.R_out, s2.R_out, (size_t) T * D, "R", T, apply, w);
    };
    ok_variant[2] = ok_variant[3] = ok;
    bool single_ok = ok;
    for (int apply = 0; apply < 2 && ok; ++apply) {
        for (int T = 1; T <= TM && ok; ++T) {
            FusedGrArgs a[NV][TM];
            for (int v = 0; v < NV; ++v) {
                const size_t span = (size_t) ((uint8_t*) (set[v].mixed + (size_t) TM * N) - (uint8_t*) set[v].R_out);
                st->memset(set[v].R_out, 0xFF, span);   // the whole set
                for (int k = 0; k < T; ++k) {
                    FusedGrArgs& x = a[v][k];
                    x.R = d_R + (size_t) k * D; x.R_out = set[v].R_out + (size_t) k * D; x.apply = apply != 0;
                    x.bo_prev = d_bo + (size_t) k * N; x.inj_prev = d_ip + (size_t) k * HC;
                    x.w_norm = d_norm; x.w_down = d_down; x.w_up = d_up; x.w_inject = d_inj; x.eps = 1e-6f;
                    x.lo = set[v].lo + (size_t) k * LR; x.rs = set[v].rs + (size_t) k * HC;
                    x.inject_out = set[v].inj + (size_t) k * HC; x.mixed = set[v].mixed + (size_t) k * N;
                }
            }
            for (int v = 0; v < 3; ++v) {
                GrMulti m;
                for (int k = 0; k < T; ++k) m.a[k] = a[v][k];
                m.xn = set[v].xn;
                m.T = T;
                launch_multi(m, v + 1, st, nullptr, 0);
            }
            if (T == 1) fused_gr_read(a[3][0], st);
            st->sync();
            if (ok_variant[2] && !all_same(set[0], set[1], T, apply, why[2])) ok_variant[2] = false;
            if (ok && ok_variant[3] && !all_same(set[0], set[2], T, apply, why[3])) ok_variant[3] = false;
            if (ok && T == 1 && single_ok && !all_same(set[3], set[0], T, apply, why[1])) single_ok = false;
        }
    }
    if (!single_ok && ok) {                            // the plain read itself disagrees with the single-token read
        why[2] = why[3] = "the plain read differs from the single-token read: " + why[1];
        ok_variant[2] = ok_variant[3] = false;
    }
    if (st != &strata::ze::default_stream()) {
        st->sync();
        delete st;
    }
    strata::ze::free(base);
    if (!ok) ok_variant[2] = ok_variant[3] = false;
    return ok;
}

}  // namespace

int fused_gr_variant() {
    int dev = 0;
    dev = strata::ze::current_device();
    const int v = dev >= 0 && dev < 64 ? g_variant[dev].load() : 0;
    if (v > 0) return v;
    // not checked on this card: the plain read, unless STRATA_HC_SPLIT names a variant (a test such as gr_parity)
    const char* e = std::getenv("STRATA_HC_SPLIT");
    return e != nullptr && (e[0] == '1' || e[0] == '2') ? env_variant() : kHcPlain;
}

void fused_gr_check() {
    int dev = 0;
    dev = strata::ze::current_device();
    if (dev < 0 || dev >= 64 || g_variant[dev].load() > 0) return;
    // SYCL port: the port's own read (sliced down / split norm, STRATA_GR_DOWN_SLICED etc.) does not use upstream's
    // variants, and their self-test crashes on the B70 (0.1.32 merge, under investigation): opt-in only.
    if (std::getenv("STRATA_HC_CHECK") == nullptr) {
        g_variant[dev].store(kHcPlain);
        return;
    }
    const int want = env_variant();
    if (want == kHcPlain) {
        g_variant[dev].store(kHcPlain);
        std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as the plain one (STRATA_HC_SPLIT=0)\n",
                     dev);
        return;
    }
    bool okv[4];
    std::string why[4];
    const bool ran = fused_gr_selftest(okv, why);
    int use = kHcPlain;
    if (want >= kHcStaged && okv[kHcStaged]) use = kHcStaged;
    else if (okv[kHcSplit]) use = kHcSplit;
    g_variant[dev].store(use);
    if (!ran)
        std::fprintf(stderr, "strata hc: CUDA%d: the check of split/staged could not run (%s)\n", dev, why[0].c_str());
    static const char* const name[4] = {"", "plain", "split", "staged"};
    for (int v = kHcStaged; v >= kHcSplit; --v)
        if (v <= want && !okv[v] && ran)
            std::fprintf(stderr, "strata hc: CUDA%d: the %s read differs from the plain read on this card - not used: "
                                 "%s\n", dev, name[v], why[v].c_str());
    static const char* const what[4] = {
        "", "the plain read (the norm per token, the down projection on 41 blocks)",
        "split (the norm per token and stream, then the plain read's down projection)",
        "staged (the norm per token and stream, the down projection's activations staged ahead by cp.async)"};
    std::fprintf(stderr, "strata hc: CUDA%d: the hyper-connection read runs as %s%s\n", dev, what[use],
                 use >= kHcSplit ? "; checked bit for bit against the plain read on this card (STRATA_HC_SPLIT=1 or 0 "
                                   "for the earlier ones)" : "");
}


}  // namespace strata::kernels
