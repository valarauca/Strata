// src/kernels/cuda/gr.host.cpp - the host side of gr.dp.cpp: the workspace carve and the launchers.
#include "strata/kernels/gr.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {

#include "gr.kern.inc"

namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
bool fp32_activations = false;
bool native_mmvf = false;

}  // namespace

void gr_set_fp32_activations(bool enabled) { fp32_activations = enabled; }
void gr_set_native_mmvf(bool enabled) { native_mmvf = enabled; }

size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    // ONE table, indexed by the same `k` that assigns the pointers below.  The first version of this function
    // sized three locals named `xn`, `xq`, `lq` in a DIFFERENT order from the pointer assignments - `xq` got
    // hc_lr*2 while `lq` got hc_dim*4 - so for the real geometry xq was given 640 bytes where it needed
    // 20,480 and every region overlapped its neighbour.
    //
    // What that looked like is worth recording: `inject` came out ~20% wrong, and rewriting its reduction
    // from block-wide to warp-wide produced BYTE-IDENTICAL wrong answers.  Two different reductions agreeing
    // exactly is not a coincidence to investigate - it means the reduction is not where the bug is, and the
    // inputs are.  Chasing the reduction instead cost most of a round.
    const size_t sz[5] = {
        hc_dim * sizeof(float),                // 0: xn
        hc_dim * sizeof(uint16_t),             // 1: xq
        (size_t) s.hc_lr * sizeof(uint16_t),   // 2: lq
        hc_dim * sizeof(float),                // 3: gated
        (size_t) s.hc_lr * sizeof(float),        // 4: lo (FP32 activation experiment)
    };
    size_t al[5], bytes = 0;
    for (int k = 0; k < 5; ++k) {
        al[k] = (sz[k] + 15) & ~(size_t) 15;
        bytes += al[k];
    }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*) base;
        void* ptr[5];
        for (int k = 0; k < 5; ++k) {
            ptr[k] = p;
            p += al[k];
        }
        out.xn = (float*) ptr[0];
        out.xq = (uint16_t*) ptr[1];
        out.lq = (uint16_t*) ptr[2];
        out.gated = (float*) ptr[3];
        out.lo = (float*) ptr[4];
    }
    return bytes;
}

void gr_read(const float *R, const float *w_norm, const uint16_t *w_down,
             const uint16_t *w_up, const uint16_t *w_inject, float eps,
             const GrShapes &s, const GrWorkspace &ws, float *mixed,
             float *inject, void *stream) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0) return;
    if (ws.xn == nullptr || ws.xq == nullptr || ws.lq == nullptr || ws.gated == nullptr || ws.lo == nullptr) {
        std::fprintf(stderr, "gr_read: GrWorkspace is not initialised (see gr_workspace_init)\n");
        std::exit(1);
    }
    if (ws.bytes < gr_workspace_bytes(s)) {
        std::fprintf(stderr, "gr_read: GrWorkspace is %zu bytes but this geometry needs %zu\n",
                     ws.bytes, gr_workspace_bytes(s));
        std::exit(1);
    }
    const int n_embd = (int) s.n_embd, hc = (int) s.hc, hc_lr = (int) s.hc_lr;
    const int hc_dim = (int) (s.hc * s.n_embd);
    strata::ze::Stream* st = strata::ze::stream_of(stream);

    // One setting selects every projection in this call. Captured graphs retain these kernel variants.
    const bool use_native = native_mmvf;
    const bool use_fp32 = fp32_activations || use_native;
    if (use_native) {
        if ((hc_dim & 1) != 0 || (hc_lr & 1) != 0)
            throw std::invalid_argument("gr_read native MMVF requires even hc*n_embd and hc_lr");
        native_gr_rms_norm_weighted(R, w_norm, ws.xn, n_embd, hc, eps, stream);
        bf16_gemv_fp32_mmvf(ws.xn, w_down, ws.lo, hc_dim, hc_lr, stream);
        native_gr_down_silu(ws.lo, hc_lr, hc, stream);
        bf16_gemv_fp32_mmvf(ws.lo, w_up, ws.gated, hc_lr, hc_dim, stream);
        // The existing null-injection contract identifies the final mixer.
        // Pinned qwen4exp uses fused HC pre for layers and the unfused graph for il=-1.
        native_gr_pre_gated(ws.xn, ws.gated, mixed, n_embd, hc, w_inject != nullptr, stream);
    } else if (use_fp32) {
        k_gr_norm_f32(st, {(uint32_t) hc, 1, 1}, {THREADS, 1, 1}, 0, R, w_norm, eps, n_embd, ws.xn, ws.xq);
        k_gr_down_f32(st, {(uint32_t) hc_lr, 1, 1}, {THREADS, 1, 1}, 0, ws.xn, w_down, hc_dim, hc_lr, hc, ws.lo);
        k_gr_gate_f32(st, {(uint32_t) ((hc_dim + WARPS - 1) / WARPS), 1, 1}, {THREADS, 1, 1}, 0, ws.lo, w_up, ws.xn,
                      hc_dim, hc_lr, ws.gated);
    } else {
        k_gr_norm_bf16(st, {(uint32_t) hc, 1, 1}, {THREADS, 1, 1}, 0, R, w_norm, eps, n_embd, ws.xn, ws.xq);
        k_gr_down_bf16(st, {(uint32_t) hc_lr, 1, 1}, {THREADS, 1, 1}, 0, ws.xq, w_down, hc_dim, hc_lr, hc, ws.lq);
        k_gr_gate_bf16(st, {(uint32_t) ((hc_dim + WARPS - 1) / WARPS), 1, 1}, {THREADS, 1, 1}, 0, ws.lq, w_up, ws.xn,
                       hc_dim, hc_lr, ws.gated);
    }
    if (!use_native)
        k_gr_mean(st, {(uint32_t) ((n_embd + THREADS - 1) / THREADS), 1, 1}, {THREADS, 1, 1}, 0, ws.gated, n_embd, hc,
                  mixed);
    // Absent for the final mixer, and then nothing is written - computing an injection nobody reads would be
    // a claim, not a convenience.
    if (w_inject != nullptr) {
        const int nthreads = 32 * hc;
        if (use_native)
            bf16_gemv_fp32_mmvf(ws.xn, w_inject, inject, hc_dim, hc, stream);
        else if (use_fp32)
            k_gr_inject_f32(st, {1, 1, 1}, {(uint32_t) nthreads, 1, 1}, 0, ws.xn, w_inject, hc_dim, hc, inject);
        else
            k_gr_inject_bf16(st, {1, 1, 1}, {(uint32_t) nthreads, 1, 1}, 0, ws.xq, w_inject, hc_dim, hc, inject);
    }

    if (stream == nullptr) strata::ze::device_sync();
}

static bool no_multi_gr() {
    static const bool off = [] {
        const char* v = std::getenv("STRATA_NO_MULTI_GR");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return off;
}

void gr_read_multi(const float *R, const float *w_norm, const uint16_t *w_down,
                   const uint16_t *w_up, const uint16_t *w_inject, float eps,
                   const GrShapes &s, const GrWorkspace &ws, float *xn_multi,
                   float *lo_multi, float *gated_multi, float *mixed,
                   float *inject, int n_tok, void *stream) {
    if (n_tok <= 0 || s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0) return;
    if (!no_multi_gr() && native_mmvf && n_tok > 1 && xn_multi != nullptr && lo_multi != nullptr && gated_multi != nullptr) {
        const int n_embd = (int) s.n_embd, hc = (int) s.hc, hc_lr = (int) s.hc_lr;
        const int hc_dim = (int) (s.hc * s.n_embd);
        if ((hc_dim & 1) != 0 || (hc_lr & 1) != 0)
            throw std::invalid_argument("gr_read native MMVF requires even hc*n_embd and hc_lr");
        native_gr_rms_norm_weighted_multi(R, w_norm, xn_multi, n_embd, hc, n_tok, eps, stream);
        bf16_gemv_fp32_mmvf_multi(xn_multi, hc_dim, w_down, lo_multi, hc_lr, hc_dim, hc_lr, n_tok, stream);
        native_gr_down_silu(lo_multi, hc_lr * n_tok, hc, stream);
        bf16_gemv_fp32_mmvf_multi(lo_multi, hc_lr, w_up, gated_multi, hc_dim, hc_lr, hc_dim, n_tok, stream);
        native_gr_pre_gated_multi(xn_multi, gated_multi, mixed, n_embd, hc, n_tok, w_inject != nullptr, stream);
        if (w_inject != nullptr)
            bf16_gemv_fp32_mmvf_multi(xn_multi, hc_dim, w_inject, inject, hc, hc_dim, hc, n_tok, stream);

        if (stream == nullptr) strata::ze::device_sync();
        return;
    }
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    for (int t = 0; t < n_tok; ++t) {
        gr_read(R + (size_t) t * hc_dim, w_norm, w_down, w_up, w_inject, eps, s, ws,
                mixed + (size_t) t * s.n_embd, inject ? inject + (size_t) t * s.hc : nullptr, stream);
    }
}

void gr_write(const float *R, const float *block_out, const float *inject,
              const GrShapes &s, float *R_out, void *stream) {
    if (s.n_embd <= 0 || s.hc <= 0) return;
    const long long n = (long long) s.hc * s.n_embd;
    const int blocks = (int) ((n + THREADS - 1) / THREADS);
    if (native_mmvf)
        native_gr_post(R, block_out, inject, R_out, (int) s.n_embd, (int) s.hc, stream);
    else
        k_gr_write(strata::ze::stream_of(stream), {(uint32_t) blocks, 1, 1}, {THREADS, 1, 1},
                   (size_t) s.hc * sizeof(float), R, block_out, inject, (int) s.n_embd, (int) s.hc, R_out);
    if (stream == nullptr) strata::ze::device_sync();
}

void gr_write_multi(const float *R, const float *block_out, const float *inject,
                    const GrShapes &s, float *R_out, int n_tok,
                    void *stream) {
    if (n_tok <= 0 || s.n_embd <= 0 || s.hc <= 0) return;
    if (!no_multi_gr() && native_mmvf && n_tok > 1) {
        native_gr_post_multi(R, block_out, inject, R_out, (int) s.n_embd, (int) s.hc, n_tok, stream);
        if (stream == nullptr) strata::ze::device_sync();
        return;
    }
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    for (int t = 0; t < n_tok; ++t) {
        gr_write(R + (size_t) t * hc_dim, block_out + (size_t) t * s.n_embd, inject + (size_t) t * s.hc, s,
                 R_out + (size_t) t * hc_dim, stream);
    }
}

}  // namespace strata::kernels
