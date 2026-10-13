// src/kernels/cuda/ple.host.cpp - the host side of ple.dp.cpp: the PLE block's launchers.
#include "strata/kernels/ple.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {

#include "ple.kern.inc"

namespace {

constexpr int THREADS = 256;
bool native_bf16 = false;
bool native_postops = false;

bool overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    if (a == nullptr || b == nullptr || a_bytes == 0 || b_bytes == 0) return false;
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    return aa < bb ? bb - aa < a_bytes : aa - bb < b_bytes;
}

void ck(int e, const char *what) {
}

}  // namespace

void ple_set_native_bf16(bool enabled) { native_bf16 = enabled; }
bool ple_native_bf16_enabled() { return native_bf16; }
void ple_set_native_postops(bool enabled) { native_postops = enabled; }
bool ple_native_postops_enabled() { return native_postops; }

void ple_history_advance(float* hist, const float* normalized, void* stream) {
    if (hist == nullptr || normalized == nullptr)
        throw std::invalid_argument("ple_history_advance: null history or normalized input");
    if (overlap(hist, (size_t) NG_HIST * NG_HC_DIM * sizeof(float),
                normalized, (size_t) NG_HC_DIM * sizeof(float)))
        throw std::invalid_argument("ple_history_advance: history and normalized input overlap");
    k_ple_history_advance(ze::stream_of(stream), {(NG_HC_DIM + THREADS - 1) / THREADS, 1, 1}, {THREADS, 1, 1}, 0, hist,
                          normalized);
    ck(0, "history advance launch");
}

bool ple_block_available() {
    int n = 0;
    n = ze::device_count();
    return n > 0;
}

uint64_t ple_block_scratch_bytes() {
    // five hc_dim floats + n_embd + hc, then the Q8 activation image, then the BF16 embedding copy, each
    // 16-byte aligned because `d_scratch` is cast to `float*` and `d_act` to `uint8_t*` at those offsets.
    const size_t f = (size_t) (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * sizeof(float);
    const size_t q = (size_t) (NG_N_EMBD / 32) * 34;
    const size_t e = (size_t) NG_N_EMBD * sizeof(uint16_t);
    return ((f + 15) & ~(size_t) 15) + ((q + 15) & ~(size_t) 15) + e + 256;
}

void ple_block(const float* emb, const float* hidden, const float* hist_rows, const PleWeights& w,
               PleOut& out, void* scratch, void* stream) {
    const bool native_key = w.key_native_data != nullptr && w.key_bf16 == nullptr;
    if (native_key && (!emb || !hidden || !hist_rows || !out.result || !scratch || !stream ||
                       !w.key_native_q8_1 ||
                       (w.key_native_type != 42 && w.key_native_type != 18 && w.key_native_type != 23 &&
                        w.key_native_type != 8)))
        throw std::invalid_argument("ple_block: native key requires Q2_0, IQ3_XXS, IQ4_XS or Q8_0 weights, input/output, private scratch and explicit stream");
    if (emb == nullptr || hidden == nullptr || hist_rows == nullptr || out.result == nullptr) return;
    const int n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM;
    static_assert(NG_N_EMBD == 2560 && NG_HC_DIM == 10240, "native PLE key geometry changed");
    const size_t float_bytes = (size_t) (5 * hc_dim + n_embd + hc) * sizeof(float);
    ze::Stream* st = ze::stream_of(stream);

    // One allocation for every intermediate.
    //
    // THE ENGINE WILL NOT DO THIS.  P2.T10 requires zero token-path allocations, and the fix is the same one
    // `gr_read` already uses: a caller-owned workspace carved once at startup.  It is done this way here
    // because this function's job in P2.S4 is to be CORRECT and comparable against the oracle, and matching
    // `shared_expert.cu`'s existing shape keeps the diff small.  Recorded rather than hidden.
    //
    // FIVE SEPARATE hc_dim BUFFERS, deliberately.  `key` and `normalized` are both needed at the end for the
    // oracle comparison, so aliasing one onto the other - which the first version did - silently returns the
    // WRONG `key`.  Two buffers of the same size look like an obvious saving and the only thing it saves is
    // 40 KB.
    // **THE WORKSPACE IS CARVED FROM THE CALLER'S, NOT ALLOCATED.**  Three `cudaMalloc`s and a
    // `cudaStreamSynchronize` used to live here, and inside a stream capture both are errors: the engine
    // reported `ple_block: scratch: operation not permitted when stream is capturing`, and a token-path
    // allocation is a P2.T10 violation besides.  The comment this replaces already said the right shape was
    // "a caller-owned workspace carved once at startup"; this is that.
    const size_t q8_bytes = (size_t) (n_embd / 32) * 34;
    if (scratch == nullptr) {
        std::fprintf(stderr, "ple_block: scratch is null; the caller owns it (see ple_block_scratch_bytes)\n");
        std::exit(1);
    }
    const struct Export { const float* pointer; size_t count; const char* name; } exports[] = {
        {out.key, (size_t) hc_dim, "key"}, {out.value, (size_t) n_embd, "value"},
        {out.gate, (size_t) hc, "gate"}, {out.gated, (size_t) hc_dim, "gated"},
        {out.normalized, (size_t) hc_dim, "normalized"}, {out.conv, (size_t) hc_dim, "conv"},
        {out.result, (size_t) hc_dim, "result"}
    };
    for (const auto& item : exports) {
        if (overlap(item.pointer, item.count * sizeof(float), scratch, (size_t) ple_block_scratch_bytes()))
            throw std::invalid_argument(std::string("ple_block: output ") + item.name + " overlaps scratch");
    }
    if (native_key) {
        const size_t native_bytes = native_q8_1_bytes(n_embd);
        const size_t weight_bytes = native_mmvq_weight_bytes(w.key_native_type, n_embd, hc_dim);
        if ((reinterpret_cast<uintptr_t>(w.key_native_data) & 3u) ||
            (reinterpret_cast<uintptr_t>(w.key_native_q8_1) & 3u))
            throw std::invalid_argument("ple_block: native key buffers require four-byte alignment");
        const struct Region { const void* pointer; size_t bytes; } regions[] = {
            {scratch, (size_t) ple_block_scratch_bytes()}, {emb, (size_t) n_embd * 4},
            {hidden, (size_t) hc_dim * 4}, {hist_rows, (size_t) NG_HIST * hc_dim * 4},
            {w.key_native_data, weight_bytes}, {w.value_bf16, (size_t) n_embd * n_embd * 2},
            {w.norm_key, (size_t) hc_dim * 4}, {w.norm_query, (size_t) hc_dim * 4},
            {w.norm_conv, (size_t) hc_dim * 4}, {w.conv1d_f16, (size_t) PLE_CONV_KERNEL * hc_dim * 2}
        };
        for (const auto& region : regions)
            if (overlap(w.key_native_q8_1, native_bytes, region.pointer, region.bytes))
                throw std::invalid_argument("ple_block: native key scratch overlaps workspace, input or weight");
        for (const auto& item : exports)
            if (overlap(w.key_native_q8_1, native_bytes, item.pointer, item.count * sizeof(float)))
                throw std::invalid_argument(std::string("ple_block: native key scratch overlaps output ") + item.name);
    }
    uint8_t* base = (uint8_t*) scratch;
    float* d_scratch = (float*) base;
    uint8_t* d_act = base + ((float_bytes + 15) & ~(size_t) 15);
    uint16_t* d_emb16 = (uint16_t*) (d_act + ((q8_bytes + 15) & ~(size_t) 15));
    float* d_key = d_scratch;
    float* d_query = d_key + hc_dim;
    float* d_norm = d_query + hc_dim;
    float* d_gated = d_norm + hc_dim;
    float* d_conv = d_gated + hc_dim;
    float* d_value = d_conv + hc_dim;
    float* d_gate = d_value + n_embd;

    // ---- key = grouped_norm(ple_key @ emb). The optional native projection
    // follows pinned CUDA Q8_1 MMVQ; the default retains its canonical Q8_0 path.
    if (w.pre_key != nullptr) {
        st->memcpy(d_key, w.pre_key, (size_t)hc_dim * sizeof(float));
    } else if (w.key_bf16 != nullptr) {
        bf16_gemv_fp32_mmvf(emb, w.key_bf16, d_key, n_embd, hc_dim, stream);
    } else if (native_key) {
        native_quantize_q8_1(emb, w.key_native_q8_1, n_embd, 1, stream);
        native_mmvq(w.key_native_type, w.key_native_data, w.key_native_q8_1,
                    d_key, n_embd, hc_dim, 1, stream);
    } else {
        quantize_q8_0(emb, d_act, n_embd, st);
        s2_gemv_q8(d_act, w.key_codes, w.key_scales, d_key, n_embd, hc_dim, 8, st);
    }
    if (!native_postops) {
        k_ple_gnorm(st, {(unsigned) hc, 1, 1}, {THREADS, 1, 1}, 0, d_key, w.norm_key, d_key, n_embd, NG_RMS_EPS);
        k_ple_gnorm(st, {(unsigned) hc, 1, 1}, {THREADS, 1, 1}, 0, hidden, w.norm_query, d_query, n_embd, NG_RMS_EPS);
    }

    // The value projection's independent option leaves the nonlinear PLE operations unchanged.
    if (w.pre_value != nullptr) {
        st->memcpy(d_value, w.pre_value, (size_t)n_embd * sizeof(float));
    } else if (native_bf16) {
        bf16_gemv_fp32_mmvf(emb, w.value_bf16, d_value, n_embd, n_embd, stream);
    } else {
        k_ple_to_bf16(st, {(unsigned) ((n_embd + THREADS - 1) / THREADS), 1, 1}, {THREADS, 1, 1}, 0, emb, d_emb16, n_embd);
        k_ple_bf16_gemv(st, {(unsigned) ((n_embd + THREADS - 1) / THREADS), 1, 1}, {THREADS, 1, 1}, 0, d_emb16,
                        w.value_bf16, d_value, n_embd, n_embd);
    }

    const float* normalized_key = d_key;
    if (native_postops) {
        // The key norm has a distinct destination; temporary query storage can
        // be reused for normalized gated values after the gate consumes it.
        NativePlePostopsBuffers buffers{d_query,d_norm,d_gate,d_gated,d_norm,d_conv,out.result};
        native_ple_postops(d_key,hidden,d_value,hist_rows,w,buffers,stream);
        normalized_key = d_query;
    } else {
        k_ple_gate(st, {(unsigned) hc, 1, 1}, {THREADS, 1, 1}, 0, d_key, d_query, d_gate, n_embd,
                   1.0f / sqrtf((float)n_embd));
        k_ple_bcast(st, {(unsigned) ((hc_dim + THREADS - 1) / THREADS), 1, 1}, {THREADS, 1, 1}, 0, d_value, d_gate, d_gated,
                    n_embd, hc);
        k_ple_gnorm(st, {(unsigned) hc, 1, 1}, {THREADS, 1, 1}, 0, d_gated, w.norm_conv, d_norm, n_embd, NG_RMS_EPS);
        k_ple_conv(st, {(unsigned) ((hc_dim + THREADS - 1) / THREADS), 1, 1}, {THREADS, 1, 1}, 0, hist_rows, d_norm,
                   w.conv1d_f16, d_conv, hc_dim, PLE_CONV_KERNEL, NGRAM_SIZE, NG_HIST);
        k_ple_add3(st, {(unsigned) ((hc_dim + THREADS - 1) / THREADS), 1, 1}, {THREADS, 1, 1}, 0, hidden, d_gated, d_conv,
                   out.result, hc_dim);
    }

    // ---- the intermediates the oracle comparison needs.  `key` is the NORMALISED key, because the source's
    //      `cb(key, ...)` capture is after `gnorm`; `value` is the projection before the gate.  Each stage the
    //      oracle records is reproduced here so a mismatch can be attributed instead of guessed at.
    if (out.key) st->memcpy(out.key, normalized_key, hc_dim * sizeof(float));
    if (out.value)
        st->memcpy(out.value, d_value, n_embd * sizeof(float));
    if (out.gate)
        st->memcpy(out.gate, d_gate, hc * sizeof(float));
    if (out.gated)
        st->memcpy(out.gated, d_gated, hc_dim * sizeof(float));
    if (out.normalized)
        st->memcpy(out.normalized, d_norm, hc_dim * sizeof(float));
    if (out.conv)
        st->memcpy(out.conv, d_conv, hc_dim * sizeof(float));

    ck(0, "launch");
    // **NO `cudaStreamSynchronize` HERE.**  It was there to make the function self-contained for the parity
    // test, and inside a capture it is an error - a caller that wants the result immediately synchronises
    // itself, and the engine's caller does not want that at all.
}

void ple_block_projected(const float* projected_key, const float* projected_value, const float* hidden,
                         const float* hist_rows, const PleWeights& w, PleOut& out, void* scratch, void* stream) {
    if (!projected_key || !projected_value || !hidden || !hist_rows || !out.result || !scratch) return;
    const int n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM;
    float* d_scratch = (float*) scratch;
    float* d_key = d_scratch;
    float* d_query = d_key + hc_dim;
    float* d_norm = d_query + hc_dim;
    float* d_gated = d_norm + hc_dim;
    float* d_conv = d_gated + hc_dim;
    float* d_value = d_conv + hc_dim;
    float* d_gate = d_value + n_embd;
    float* norm_dst = out.normalized ? out.normalized : d_norm;
    NativePlePostopsBuffers buffers{d_query, norm_dst, d_gate, d_gated, norm_dst, d_conv, out.result};
    native_ple_postops(projected_key, hidden, projected_value, hist_rows, w, buffers, stream);
}

}  // namespace strata::kernels
