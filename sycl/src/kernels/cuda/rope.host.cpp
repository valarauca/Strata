// src/kernels/cuda/rope.host.cpp - the host side of rope.dp.cpp: the float64 tables and the launcher.
#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "rope.kern.inc"

namespace {
// The process's rope config (rope_scaling.hpp).  One writer - the engine's startup thread, before
// session_init builds any table or captures any graph - and readers after it.
RopeScaling g_rope_scaling;
}  // namespace

void rope_scaling_set(const RopeScaling& scaling) { g_rope_scaling = scaling; }
const RopeScaling& rope_scaling() { return g_rope_scaling; }

void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            // float64 throughout, in the reference's order: inv, then ang, then cos/sin
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}

void build_rope_table(int n_rot, const RopeScaling& sc, int max_pos, float* cos_tab, float* sin_tab) {
    if (sc.type == RopeScalingType::None) {
        build_rope_table(n_rot, sc.freq_base, max_pos, cos_tab, sin_tab);   // the original loop, verbatim
        return;
    }
    const int half = n_rot / 2;
    const double fs = sc.freq_scale();
    const double ms = sc.mscale();
    double cd[2];
    sc.corr_dims(n_rot, cd);
    const bool correct = sc.ext_factor != 0;   // ggml: the correction rides on ext_factor, not the type
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(sc.freq_base, -2.0 * (double) i / (double) n_rot);
            const double extrap = (double) p * inv;    // the trained angle, ggml's theta_extrap
            const double interp = fs * extrap;         // ggml's theta_interp
            double ang = interp;
            if (correct) {
                const double ramp = (double) rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
                ang = interp * (1.0 - ramp) + extrap * ramp;
            }
            cos_tab[(size_t) p * half + i] = (float) (std::cos(ang) * ms);
            sin_tab[(size_t) p * half + i] = (float) (std::sin(ang) * ms);
        }
    }
}

void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot, const float* cos_tab,
                     const float* sin_tab, const int* pos, void* stream) {
    if (rows <= 0 || n_rot <= 0) return;
    if (n_rot % 2 != 0 || n_rot > head_dim) {
        std::fprintf(stderr, "rope_neox_apply: n_rot %d must be even and <= head_dim %d\n", n_rot, head_dim);
        std::exit(1);
    }
    const int threads = 128;
    const unsigned grid = (unsigned) ((rows + threads - 1) / threads);
    k_rope_neox(ze::stream_of(stream), {grid, 1, 1}, {threads, 1, 1}, 0, x, out, rows, head_dim, n_rot, cos_tab, sin_tab,
                pos, mrope_table());
    if (stream == nullptr) ze::device_sync();
}

}  // namespace strata::kernels
