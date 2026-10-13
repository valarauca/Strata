// src/kernels/cuda/native_rope.host.cpp - the host side of native_rope.dp.cpp: the rope tables, checks and the launchers.
//
// Numerical contract: pinned ggml/src/ggml-cuda/rope.cu, rope_multi/rope_yarn.
// Text positions are equal across the four IMRoPE sections, so section routing
// reduces to the one position associated with each contiguous row.
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/ze/kernel_abi.hpp"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strata::kernels {

#include "native_rope.kern.inc"

namespace {
std::atomic<bool> enabled{false};
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}
}
// one table per device (a layer split runs the rope kernels on several): set and read for the current device
namespace {
constexpr int kMropeDevices = 64;
std::atomic<const int32_t*> mrope_tab[kMropeDevices] = {};
int mrope_dev(int d) {
    if (d < 0 || d >= kMropeDevices) d = 0;
    return d;
}
int mrope_dev() { return mrope_dev(ze::current_device()); }
}  // namespace
void mrope_table_set(const int32_t* device_table) { mrope_tab[mrope_dev()].store(device_table, std::memory_order_relaxed); }
const int32_t* mrope_table(int dev) { return mrope_tab[mrope_dev(dev)].load(std::memory_order_relaxed); }
namespace {
// #280: the float64 angle table of each device, set at session init (before any graph is captured)
struct RopeReg {
    RopeTab tab;
    RopeScaling scaling;
};
RopeReg rope_tab[kMropeDevices] = {};
bool same_scaling(const RopeScaling& a, const RopeScaling& b) {
    return a.type == b.type && a.freq_base == b.freq_base && a.factor == b.factor && a.freq_scale_in == b.freq_scale_in &&
           a.orig_ctx == b.orig_ctx && a.ext_factor == b.ext_factor && a.attn_factor == b.attn_factor &&
           a.beta_fast == b.beta_fast && a.beta_slow == b.beta_slow;
}
// opt-in: STRATA_ROPE_TABLE=1 (the table's angles differ from the fast-math ones in the last bits, so outputs move)
bool rope_table_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_ROPE_TABLE");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}
}  // namespace
void rope_table_set(const float* cos_tab, const float* sin_tab, int max_pos, const RopeScaling& scaling) {
    rope_tab[mrope_dev()] = RopeReg{RopeTab{cos_tab, sin_tab, max_pos}, scaling};
}
void rope_table_release(const float* cos_tab) {
    // every device's entry that points at it (the caller may free it from another current device)
    for (RopeReg& r : rope_tab)
        if (cos_tab != nullptr && r.tab.cos == cos_tab) r = RopeReg{};
}
RopeTab rope_table_for(const RopeScaling& scaling, int dev) {
    if (!rope_table_enabled()) return {};
    const RopeReg& r = rope_tab[mrope_dev(dev)];
    return r.tab.cos != nullptr && same_scaling(r.scaling, scaling) ? r.tab : RopeTab{};
}
void native_rope_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_rope_enabled() { return enabled.load(std::memory_order_relaxed); }
bool native_norm_rope_usable(int head_dim, int n_rot) {
    static const bool on = [] {
        const char* off = std::getenv("STRATA_NO_NORM_ROPE");
        if (off != nullptr && off[0] != '\0' && off[0] != '0') return false;
#if defined(__HIPCC__)
        const char* v = std::getenv("STRATA_NORM_ROPE");   // AMD: opt in until rope_parity check 6 passes there
        return v != nullptr && v[0] != '\0' && v[0] != '0';
#else
        return true;
#endif
    }();
    return on && (head_dim == 128 || head_dim == 256) && n_rot == 64;
}

void native_rope_apply(const float* x, float* out, int rows, int head_dim,
                       int n_rot, const RopeScaling& scaling, const int* positions, void* stream) {
    if (!x || !out || !positions || !stream || rows < 1 || rows > 65535 ||
        (head_dim != 128 && head_dim != 256) || n_rot != 64 ||
        rope_scaling_invalid(scaling) != nullptr ||
        reinterpret_cast<uintptr_t>(x) % 4 || reinterpret_cast<uintptr_t>(out) % 4 ||
        reinterpret_cast<uintptr_t>(positions) % 4) {
        throw std::invalid_argument("native RoPE requires aligned F32 rows, width 128/256, rotation 64, valid base/scaling and explicit stream");
    }
    const size_t bytes = size_t(rows) * head_dim * sizeof(float);
    if ((x != out && overlaps(x, bytes, out, bytes)) ||
        overlaps(positions, size_t(rows) * sizeof(int), out, bytes) ||
        overlaps(positions, size_t(rows) * sizeof(int), x, bytes)) {
        throw std::invalid_argument("native RoPE buffers partially overlap");
    }
    // Match pinned host-side float powf before device fast powf/trigonometry.
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / n_rot);
    const RopeKernelArgs k = scaling.kernel_args(n_rot);   // none: the identity constants
    const RopeTab rt = rope_table_for(scaling, ze::stream_of(stream)->device());
    const unsigned threads = (x == out && n_rot == 64 && native_norm_rope_usable(head_dim, n_rot)) ? 32u : 128u;
    const ze::Dim3 grid{threads == 32u ? 1u : unsigned((head_dim / 2 + 127) / 128), unsigned(rows), 1};
    if (rt.cos != nullptr)
        k_native_rope_apply_tab(ze::stream_of(stream), grid, {threads, 1, 1}, 0, x, out, rows, head_dim, n_rot,
                                theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor, k.attn_factor,
                                positions, mrope_table(ze::stream_of(stream)->device()), rt);
    else
        k_native_rope_apply(ze::stream_of(stream), grid, {threads, 1, 1}, 0, x, out, rows, head_dim, n_rot,
                            theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor, k.attn_factor,
                            positions, mrope_table(ze::stream_of(stream)->device()), rt);
}

void native_qsa_rms_norm_rope(const float* x, int in_stride, const float* gamma, float* out,
                              int rows, int head_dim, int n_rot, float epsilon,
                              const RopeScaling& scaling, const int* positions, void* stream) {
    if (!x || !gamma || !out || !positions || !stream || rows < 1 || rows > 65535 ||
        (head_dim != 128 && head_dim != 256) || in_stride < head_dim || n_rot != 64 ||
        !std::isfinite(epsilon) || epsilon < 0.0f ||
        rope_scaling_invalid(scaling) != nullptr ||
        reinterpret_cast<uintptr_t>(x) % 4 || reinterpret_cast<uintptr_t>(gamma) % 4 ||
        reinterpret_cast<uintptr_t>(out) % 4 || reinterpret_cast<uintptr_t>(positions) % 4) {
        throw std::invalid_argument("native_qsa_rms_norm_rope: invalid arguments");
    }
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / n_rot);
    const RopeKernelArgs k = scaling.kernel_args(n_rot);
    const RopeTab rt = rope_table_for(scaling, ze::stream_of(stream)->device());
    if (rt.cos != nullptr)
        k_native_rope_norm_rope_tab(ze::stream_of(stream), {unsigned(rows), 1, 1}, {256, 1, 1}, 0, x, in_stride, gamma,
                                    out, head_dim, n_rot, epsilon, theta_scale, k.freq_scale, k.corr_low, k.corr_high,
                                    k.ext_factor, k.attn_factor, positions, mrope_table(ze::stream_of(stream)->device()), rt);
    else
        k_native_rope_norm_rope(ze::stream_of(stream), {unsigned(rows), 1, 1}, {256, 1, 1}, 0, x, in_stride, gamma,
                                out, head_dim, n_rot, epsilon, theta_scale, k.freq_scale, k.corr_low, k.corr_high,
                                k.ext_factor, k.attn_factor, positions, mrope_table(ze::stream_of(stream)->device()), rt);
}
}
