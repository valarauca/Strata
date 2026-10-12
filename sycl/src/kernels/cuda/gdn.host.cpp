// src/kernels/cuda/gdn.host.cpp - the host side of gdn.dp.cpp: the launchers.
#include "strata/kernels/gdn.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "gdn.kern.inc"

namespace {

constexpr int JTHREADS = 32;   ///< threads along j, the state's fast axis
constexpr int MAX_H = 1;       ///< heads staged in shared memory per block

}  // namespace

void gdn_step(float *state, const float *q, const float *k, const float *v,
              const float *gate, const float *beta, float *o,
              const GdnShapes &s, void *stream) {
    if (s.S <= 0 || s.h_k <= 0 || s.h_v <= 0) return;
    if (s.S > 128) {
        std::fprintf(stderr, "gdn_step: S = %lld exceeds the staged 128 (`ks`/`qs` are [8][128])\n",
                     (long long) s.S);
        std::exit(1);
    }
    const ze::Dim3 grid{(unsigned)((s.S + JTHREADS - 1) / JTHREADS), (unsigned)((s.h_v + MAX_H - 1) / MAX_H), 1};
    k_gdn_step(ze::stream_of(stream), grid, {JTHREADS, 1, 1}, 0, state, q, k, v, gate, beta, o, (int)s.S, (int)s.h_k,
               (int)s.h_v);
    if (stream == nullptr) ze::default_stream().sync();
}

void gdn_conv_step(float *conv_state, const float *x, const float *kW,
                   float *out, int64_t channels, int64_t d_conv,
                   void *stream) {
    if (channels <= 0 || d_conv < 1) return;
    const int blocks = (int) ((channels + 255) / 256);
    k_gdn_conv(ze::stream_of(stream), {(unsigned)blocks, 1, 1}, {256, 1, 1}, 0, conv_state, x, kW, out, (int)channels,
               (int)d_conv);
    if (stream == nullptr) ze::default_stream().sync();
}

void gdn_l2_norm(float *x, int64_t rows, int64_t cols, float eps,
                 void *stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols > 1024) {
        std::fprintf(stderr, "gdn_l2_norm: cols = %lld exceeds the 1024-wide warp reduction\n", (long long) cols);
        std::exit(1);
    }
    k_gdn_l2(ze::stream_of(stream), {(unsigned)rows, 1, 1}, {32, 1, 1}, 0, x, (int)cols, eps);
    if (stream == nullptr) ze::default_stream().sync();
}

void gdn_beta_gate(float *beta, int64_t h_v, void *stream) {
    if (beta == nullptr || h_v <= 0) return;
    const int n = (int) h_v;
    k_gdn_beta_gate(ze::stream_of(stream), {(unsigned)((n + 127) / 128), 1, 1}, {128, 1, 1}, 0, beta, n);
    if (stream == nullptr) ze::default_stream().sync();
}

void gdn_out_norm(const float *o, const float *z, const float *ssm_norm,
                  float *y, int64_t h_v, int64_t S, float eps,
                  void *stream) {
    if (h_v <= 0 || S <= 0) return;
    k_gdn_out_norm(ze::stream_of(stream), {(unsigned)h_v, 1, 1}, {32, 1, 1}, 0, o, z, ssm_norm, y, (int)S, eps);
    if (stream == nullptr) ze::default_stream().sync();
}

}  // namespace strata::kernels
