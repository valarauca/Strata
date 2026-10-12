// src/prefill/kernels.host.cpp - the host side of kernels.dp.cpp: the prompt path's launchers.
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::prefill {

#include "kernels.kern.inc"

namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;
constexpr int CONV_TILE = 64;
constexpr int RG = 4, RPG = S / RG;
constexpr int CB = 32, NCB = S / CB;
unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }
#if !defined(__HIPCC__)
// gdn_rec_kh_kernel where it pays: a CUDA card with cp.async (sm_80+) that holds all 64 of its blocks at once (each
// walks the whole chunk, so blocks left for a second wave would double the time), and not with 48 to 63 SMs.  The
// busiest SM sets the pace: from 64 SMs up this kernel has one block per SM, below that two on some SMs (1.5 times as
// long), while the kernel before has ceil(192 / SMs); two against at most four is a draw.  With the engine's grids on
// a 4080 SUPER held to fewer SMs (gdn_rec_parity --bench): 1.40-1.42x at 64 to 80 SMs, 1.02-1.04x at 48 to 63,
// 1.28-1.31x at 39 to 47, 1.53-1.57x at 32 to 38; an Ampere card gained less at 82 SMs (1.28x on a 3090 against
// 1.41x here), so 48 to 63 SMs keep the kernel before.  Per call, from the current device (a layer split can mix
// cards).
bool gdn_keyhead_ok() {
    // SYCL port: upstream picks it by SM count and occupancy (NVIDIA thresholds, measured on Ada/Ampere). On Xe the
    // key-head kernel is an A/B until measured: STRATA_GDN_KEYHEAD=1 runs it, anything else keeps the kernel before.
    static const bool on = [] { const char* v = std::getenv("STRATA_GDN_KEYHEAD"); return v != nullptr && std::atoi(v) == 1; }();
    return on;
}
#endif

}  // namespace

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host, const strata::kernels::KvHostPools* stage) {
    if (T <= 0) return;
    static const bool wide_env = [] { const char* v = std::getenv("STRATA_KV_WIDE_HOST"); return !(v && v[0] == '0'); }();
    const bool wide = wide_env;
    k_kernels_kv_append(strata::ze::stream_of(stream), {(unsigned)T, 2, 8}, {64, 1, 1}, 0, K, V, pos0, page_table,
                        page_size, k_pool, v_pool, k_q, v_q, k_scale, v_scale,
                        host ? *host : strata::kernels::KvHostPools{}, stage ? *stage : strata::kernels::KvHostPools{},
                        wide);
}
void set_act_f16(bool on) {
#if !defined(__HIPCC__)
    (void) on;   // CUDA: the activation image is always BF16
    return;
#else
    const int v = on ? 1 : 0;
    if (cudaMemcpyToSymbol(g_act_f16, &v, sizeof v) != cudaSuccess) {
        std::fprintf(stderr, "prefill: setting the FP16 activation image failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
#endif
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    k_kernels_to_f16(strata::ze::stream_of(stream), {(unsigned)((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 1, 1},
                     {256, 1, 1}, 0, x, y, n);
}
void round_f16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    k_kernels_round_f16(strata::ze::stream_of(stream),
                        {(unsigned)((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 1, 1}, {256, 1, 1}, 0, x, y, n);
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo) {
    if (n <= 0) return;
    k_kernels_to_bf16(strata::ze::stream_of(stream),
                      {(unsigned)((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 1, 1}, {256, 1, 1}, 0, x, y, ylo,
                      n);
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
             uint16_t* xn16_lo) {
    k_kernels_gr_norm(strata::ze::stream_of(stream), {(unsigned)(T * HC), 1, 1}, {256, 1, 1}, 0, R, w_norm, eps, xn,
                      xn16, xn16_lo);
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
                uint16_t* xn16_lo, int64_t ldx) {
    k_kernels_gr_norm_rs(strata::ze::stream_of(stream), {(unsigned)(T * HC), 1, 1}, {256, 1, 1}, 0, R, w_norm, eps, rs,
                         xn16, xn16_lo, ldx > 0 ? ldx : D);
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    k_kernels_gr_mix_r(strata::ze::stream_of(stream), {blocks_for(T * N), 1, 1}, {256, 1, 1}, 0, R, rs, w_norm, gated,
                       mixed, mixed16, T, mixed_h, mixed16_lo);
}
bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const float* rs, const float* w_norm,
              float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t T, void* stream) {
#if defined(__HIPCC__)
    static const bool gfx11 = [] {
        int dev = 0;
        cudaDeviceProp p;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        return strata::kernels::gfx_arch_is_gfx11_wmma(p.gcnArchName);
    }();
    if (!gfx11 || T <= 0) return false;
    gr_upmix_kernel<<<dim3(N / UM_D, (unsigned) ((T + UM_T - 1) / UM_T)), 256, 0, (cudaStream_t) stream>>>(
        lo16, w_up, R, rs, w_norm, mixed, mixed16, mixed_h, T);
    check("gr_upmix");
    return true;
#else
    (void) lo16; (void) w_up; (void) R; (void) rs; (void) w_norm; (void) mixed; (void) mixed16; (void) mixed_h;
    (void) T; (void) stream;
    return false;
#endif
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx) {
    k_kernels_gr_write_norm_rs(strata::ze::stream_of(stream), {(unsigned)(T * HC), 1, 1}, {256, 1, 1}, 0, R, bo, inj,
                               inj_ld, w_norm_next, eps, rs, xn16, xn16_lo, ldx > 0 ? ldx : D);
}
void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* v_l,
                           const float* s_l, const int* on, int mode, const float* w_norm_next, float eps, float* rs,
                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx) {
    k_kernels_gr_write_cvec_norm_rs(strata::ze::stream_of(stream), {(unsigned)(T * HC), 1, 1}, {256, 1, 1}, 0, R, bo,
                                    inj, inj_ld, v_l, s_l, on, mode, w_norm_next, eps, rs, xn16, xn16_lo,
                                    ldx > 0 ? ldx : D);
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo) {
    k_kernels_gr_silu(strata::ze::stream_of(stream), {blocks_for(T * LR), 1, 1}, {256, 1, 1}, 0, lo, lo16, lo16_lo,
                      T * LR);
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h, uint16_t* mixed16_lo) {
    k_kernels_gr_mix(strata::ze::stream_of(stream), {blocks_for(T * N), 1, 1}, {256, 1, 1}, 0, xn, gated, mixed,
                     mixed16, T, mixed_h, mixed16_lo);
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    k_kernels_gr_write(strata::ze::stream_of(stream), {blocks_for(T * D), 1, 1}, {256, 1, 1}, 0, R, bo, inj, inj_ld, T);
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    k_kernels_gr_broadcast(strata::ze::stream_of(stream), {blocks_for(T * D), 1, 1}, {256, 1, 1}, 0, e, R, T);
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    k_kernels_gdn_gates(strata::ze::stream_of(stream), {blocks_for(T * HV), 1, 1}, {256, 1, 1}, 0, ab, dt, ssm_a, gate,
                        beta, T);
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_CONV_SERIAL") != nullptr;   // the old walk (A/B)
    if (serial || T <= CONV_TILE) {
        k_kernels_gdn_conv(strata::ze::stream_of(stream), {C / 128, 1, 1}, {128, 1, 1}, 0, history, qkv, conv_w, h, T);
    } else {
        static const bool fused = [] { const char* v = std::getenv("STRATA_GDN_CONVL2"); return v && std::atoi(v) != 0; }();
        if (fused) {
            k_kernels_gdn_conv_l2(strata::ze::stream_of(stream),
                                  {C / 128, (unsigned)((T + CONV_TILE - 1) / CONV_TILE), 1}, {128, 1, 1}, 0, history,
                                  qkv, conv_w, h, T, eps);
            k_kernels_gdn_conv_hist(strata::ze::stream_of(stream), {C / 128, 1, 1}, {128, 1, 1}, 0, history, qkv, T);
            return;
        }
        k_kernels_gdn_conv_tiled(strata::ze::stream_of(stream),
                                 {C / 128, (unsigned)((T + CONV_TILE - 1) / CONV_TILE), 1}, {128, 1, 1}, 0, history,
                                 qkv, conv_w, h, T);
        k_kernels_gdn_conv_hist(strata::ze::stream_of(stream), {C / 128, 1, 1}, {128, 1, 1}, 0, history, qkv, T);
    }
    k_kernels_gdn_l2(strata::ze::stream_of(stream), {2 * HK, (unsigned)T, 1}, {S, 1, 1}, 0, h, eps);
}
void gdn_recurrence_variant(int variant, float* state, const float* h, const float* gate, const float* beta,
                            const float* z, const float* gamma, float eps, float* y, uint16_t* y16, int64_t T,
                            void* stream, int64_t ld16) {
    if (ld16 <= 0) ld16 = (int64_t) HV * S;
#if defined(__HIPCC__)
    if (variant == 3) {   // diagnostics: the quad recurrence + the old norm kernel
        if (T > 0) {
            gdn_rec_quad_kernel<<<HV * (S / QCB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16, ld16);
        }
        return;
    }
    if (variant == 4 || variant == 5) {   // diagnostics: the quad recurrence's PP / two-column kernels + the norm
        if (T > 0) {
            if (variant == 5)
                gdn_rec_quad2c_kernel<<<HV * (S / C2CB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            else
                gdn_rec_quad_pp_kernel<<<HV * (S / QCB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            gdn_out_norm_loop_kernel<true><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                z, gamma, eps, y, y16, T * HV, ld16);
        }
        check("gdn_recurrence (quad pp)");
        return;
    }
    if (variant == 1) {
        if (T > 0) {
            static const int pp = [] { const char* v = std::getenv("STRATA_GDN_PP"); return v ? std::atoi(v) : 0; }();
            if (pp == 2)
                gdn_rec_quad2c_kernel<<<HV * (S / C2CB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            else if (pp)
                gdn_rec_quad_pp_kernel<<<HV * (S / QCB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            else
                gdn_rec_quad_kernel<<<HV * (S / QCB), QTH, 0, (cudaStream_t) stream>>>(state, h, gate, beta, y, T);
            static const bool noy = [] { const char* v = std::getenv("STRATA_GDN_NOY"); return v && std::atoi(v) != 0; }();
            if (noy)
                gdn_out_norm_loop_kernel<false><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                    z, gamma, eps, y, y16, T * HV, ld16);
            else
                gdn_out_norm_loop_kernel<true><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                    z, gamma, eps, y, y16, T * HV, ld16);
        }
        check("gdn_recurrence (quad)");
        return;
    }
#else
    (void) variant;   // the quad recurrence kernels are AMD-only
#endif
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;   // the one-block-per-head kernel (A/B)
    if (serial || T <= 0 || variant == 2) {
        k_kernels_gdn_rec(strata::ze::stream_of(stream), {HV, 1, 1}, {S, RG, 1}, 0, state, h, gate, beta, z, gamma, eps,
                          y, y16, T, ld16);
    } else {
        static const bool pipe = [] { const char* v = std::getenv("STRATA_GDN_PIPELINE"); return v == nullptr || std::atoi(v) != 0; }();
#if !defined(__HIPCC__)
        if (pipe && gdn_keyhead_ok())   // the value heads of a key head in one thread (same bits)
        {
            k_kernels_gdn_rec_kh(strata::ze::stream_of(stream), {HK * NCB, 1, 1}, {CB, RG, 1}, 0, state, h, gate, beta,
                                 y, T);
        } else
#endif
            if (pipe) // the software-pipelined loads (same bits)
        {
            k_kernels_gdn_rec_cols_pipe(strata::ze::stream_of(stream), {HV * NCB, 1, 1}, {CB, RG, 1}, 0, state, h, gate,
                                        beta, y, T);
        } else {
            k_kernels_gdn_rec_cols(strata::ze::stream_of(stream), {HV * NCB, 1, 1}, {CB, RG, 1}, 0, state, h, gate,
                                   beta, y, T);
        }
        k_kernels_gdn_out_norm(strata::ze::stream_of(stream), {(unsigned)T, HV, 1}, {S, 1, 1}, 0, z, gamma, eps, y, y16,
                               ld16);
    }
}
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream, int64_t ld16) {
    // Aurora (S23): four lanes per column + the grid-stride norm, the same bits (tests/hip/gdn_rec_head.cpp);
    // STRATA_GDN_HEAD=1 (on by default on gfx1151; off with STRATA_GDN_HEAD=0 or STRATA_GDN_REC_HEADS) takes them
    // (AMD builds only; the arch defaults set STRATA_GDN_HEAD=1 on gfx1151 and leave other cards on the earlier kernels)
    static const bool head = [] { const char* v = std::getenv("STRATA_GDN_HEAD"); return v != nullptr && std::atoi(v) != 0; }();
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;
    gdn_recurrence_variant(head && !serial ? 1 : 0, state, h, gate, beta, z, gamma, eps, y, y16, T, stream, ld16);
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (n_expert == 512)
    {
        k_kernels_route_16(strata::ze::stream_of(stream), {(unsigned)((T + 7) / 8), 1, 1}, {256, 1, 1}, 0, logits, ids,
                           weights, T);
    } else if (n_expert == 256)
    {
        k_kernels_route_8(strata::ze::stream_of(stream), {(unsigned)((T + 7) / 8), 1, 1}, {256, 1, 1}, 0, logits, ids,
                          weights, T);
    } else
        strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    k_kernels_blob_dequant_false(strata::ze::stream_of(stream), {blocks_for(1280LL * 640 + 2560LL * 160), 1, 1},
                                 {256, 1, 1}, 0, blob, gu16, down16);
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    k_kernels_blob_dequant_true(strata::ze::stream_of(stream), {blocks_for(1280LL * 640 + 2560LL * 160), 1, 1},
                                {256, 1, 1}, 0, blob, gu16, down16);
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    k_kernels_swiglu_il(strata::ze::stream_of(stream), {blocks_for(n * 640), 1, 1}, {256, 1, 1}, 0, gu, h16, n);
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    k_kernels_swiglu_pair(strata::ze::stream_of(stream), {blocks_for(n * 640), 1, 1}, {256, 1, 1}, 0, g, u, h16, n);
}
void copy_f32_wide(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t n4 = n / 4, b = std::min<int64_t>((n4 + 255) / 256, 4096);
    if (n4 > 0) {
        k_kernels_copy_f4(strata::ze::stream_of(stream), {(unsigned)b, 1, 1}, {256, 1, 1}, 0, dst, src, n4);
    }
    if (n % 4) copy_i32((int32_t*) dst + n4 * 4, (const int32_t*) src + n4 * 4, n % 4, stream);
}
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t b = (n + 255) / 256;
    k_kernels_copy_i32(strata::ze::stream_of(stream), {(unsigned)(b < 256 ? b : 256), 1, 1}, {256, 1, 1}, 0, dst, src,
                       n);
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    k_kernels_gather_rows16(strata::ze::stream_of(stream), {blocks_for(n * (width / 8)), 1, 1}, {256, 1, 1}, 0, x16,
                            src, dst16, n, width);
}
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream) {
#ifndef STRATA_W_NO_COMB
    if (((reinterpret_cast<uintptr_t>(Dm) | reinterpret_cast<uintptr_t>(shared) | reinterpret_cast<uintptr_t>(bo)) & 15) == 0) {
        k_kernels_moe_combine4(strata::ze::stream_of(stream), {blocks_for(T * (N / 4)), 1, 1}, {256, 1, 1}, 0, Dm, slot,
                               w, shared, sg, bo, T);
        return;
    }
#endif
    k_kernels_moe_combine(strata::ze::stream_of(stream), {blocks_for(T * N), 1, 1}, {256, 1, 1}, 0, Dm, slot, w, shared,
                          sg, bo, T);
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    k_kernels_rms_rows(strata::ze::stream_of(stream), {(unsigned)rows, 1, 1}, {256, 1, 1}, 0, x, w, cols, ld, eps);
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
          const strata::kernels::RopeScaling& scaling, void* stream) {
    // The engine validates the resolved config at startup with the same rule (generate.cpp), so this only
    // fires for a caller that bypassed it; the prompt path has no error return here, so it stops the process.
    if (const char* why = strata::kernels::rope_scaling_invalid(scaling)) {
        std::fprintf(stderr, "prefill rope: invalid rope scaling: %s\n", why);
        std::exit(1);
    }
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / 64.0f);
    const strata::kernels::RopeKernelArgs k = scaling.kernel_args(64);   // none: the identity constants
    const strata::kernels::RopeTab rt = strata::kernels::rope_table_for(scaling);
    if (rt.cos != nullptr)
    {
        k_kernels_rope_true(strata::ze::stream_of(stream), {(unsigned)(T * heads), 1, 1}, {32, 1, 1}, 0, x, heads, dim,
                            ld, pos0, theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor, k.attn_factor,
                            strata::kernels::mrope_table(), rt);
    } else {
        k_kernels_rope_false(strata::ze::stream_of(stream), {(unsigned)(T * heads), 1, 1}, {32, 1, 1}, 0, x, heads, dim,
                             ld, pos0, theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor, k.attn_factor,
                             strata::kernels::mrope_table(), rt);
    }
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    k_kernels_split_q(strata::ze::stream_of(stream), {blocks_for(T * 24 * 256), 1, 1}, {256, 1, 1}, 0, q_full, q, T);
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream, int64_t ld16) {
    k_kernels_gate_attn(strata::ze::stream_of(stream), {blocks_for(T * 24 * 256), 1, 1}, {256, 1, 1}, 0, attn, q_full,
                        out16, T, ld16 > 0 ? ld16 : 24 * 256);
}

}  // namespace strata::prefill
