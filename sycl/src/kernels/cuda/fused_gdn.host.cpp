// src/kernels/cuda/fused_gdn.host.cpp - the host side of fused_gdn.dp.cpp: argument checks and the launchers.
#include "strata/kernels/fused_gdn.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "fused_gdn.kern.inc"

namespace {

constexpr int S = 128;          // state size (rows = cols = 128)
constexpr int RG = 4;           // row groups

}  // namespace

void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || qk_heads < 0 || qk_heads > channels / S) {
        std::fprintf(stderr, "fused_gdn_conv_l2: invalid arguments\n");
        std::exit(1);
    }
    k_fused_gdn_conv_l2(ze::stream_of(stream), {(unsigned)(channels / S), 1, 1}, {S, 1, 1}, 0, history, qkv, conv_w,
                        h, qk_heads, eps);
}

void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    if (!x || !w_alpha || !w_beta || !dt || !ssm_a || !gate || !beta || n_embd % 8 != 0 || h_v <= 0) {
        std::fprintf(stderr, "fused_gdn_ab: invalid arguments\n");
        std::exit(1);
    }
    k_fused_gdn_ab(ze::stream_of(stream), {(unsigned)((2 * h_v + 7) / 8), 1, 1}, {256, 1, 1}, 0, x, w_alpha, w_beta,
                   dt, ssm_a, gate, beta, n_embd, h_v);
}

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k) {
        std::fprintf(stderr, "fused_gdn_step_norm: invalid arguments\n");
        std::exit(1);
    }
    k_fused_gdn_step_norm(ze::stream_of(stream), {(unsigned)h_v, 1, 1}, {S, RG, 1}, 0, state, q, k, v, gate, beta, z,
                          gamma, eps, y, h_k, h_v);
}

}  // namespace strata::kernels
