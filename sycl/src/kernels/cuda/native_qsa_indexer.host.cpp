// src/kernels/cuda/native_qsa_indexer.host.cpp - the host side of native_qsa_indexer.dp.cpp: validation and the launchers.
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/ze/kernel_abi.hpp"
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {

#include "native_qsa_indexer.kern.inc"

namespace {
std::atomic<bool> enabled{false};
constexpr int D = 128, R = 4, ROT = 64, THREADS = 256;
struct Span { const void* p; std::size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<std::uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA indexer requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.p), y = reinterpret_cast<std::uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
} // namespace

void native_qsa_indexer_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_qsa_indexer_append_steps(const float* raw, const int32_t* relative_pos_device,
                                     int pos_stride, int n_steps, int32_t pos_base,
                                     const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                                     const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling, void* stream) {
    if (n_steps <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency/scaling and explicit stream");
    const std::size_t pos_span = (n_steps == 1 || pos_stride <= 0) ? 4 : (std::size_t(n_steps - 1) * pos_stride + 1) * 4;
    const Span spans[] = {{raw,std::size_t(n_steps)*D*4},{relative_pos_device,pos_span},{gamma,D*4},{b.tail,(R-1)*D*4},
        {b.dead,D*4},{b.pooled,std::size_t(max_cells/R+1)*D*4},{b.block_pos,4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / ROT);
    const RopeKernelArgs k = scaling.kernel_args(ROT);   // none: the identity constants
    const RopeTab rt = rope_table_for(scaling, ze::stream_of(stream)->device());
    if (rt.cos != nullptr)
    {
        k_native_qsa_indexer_append_tab(ze::stream_of(stream), {1, 1, 1}, {THREADS, 1, 1}, 0, raw, relative_pos_device,
                                        pos_stride, n_steps, pos_base, gamma, epsilon, b.tail, b.dead, b.pooled,
                                        b.block_pos, int(max_cells), theta_scale, k.freq_scale, k.corr_low, k.corr_high,
                                        k.ext_factor, k.attn_factor, mrope_table(ze::stream_of(stream)->device()), rt);
    } else {
        k_native_qsa_indexer_append_plain(ze::stream_of(stream), {1, 1, 1}, {THREADS, 1, 1}, 0, raw, relative_pos_device,
                                          pos_stride, n_steps, pos_base, gamma, epsilon, b.tail, b.dead, b.pooled,
                                          b.block_pos, int(max_cells), theta_scale, k.freq_scale, k.corr_low,
                                          k.corr_high, k.ext_factor, k.attn_factor, mrope_table(ze::stream_of(stream)->device()), rt);
    }
}

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                               const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling, void* stream) {
    native_qsa_indexer_append_steps(raw, relative_pos_device, 0, 1, pos_base, gamma, epsilon, b, s, max_cells, scaling, stream);
}
void native_qsa_indexer_append_batch(const float* raw, int64_t n, int64_t p0, int32_t pos_base, const float* gamma,
                                     float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s, int64_t max_cells,
                                     const RopeScaling& scaling, void* stream) {
    if (n <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT || p0 < 0 || p0 + n > max_cells ||
        max_cells > INT32_MAX || pos_base < 0 || pos_base % R || int64_t(pos_base) + max_cells > INT32_MAX ||
        !std::isfinite(epsilon) || epsilon <= 0.0f || rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer (batch): bad geometry, positions, parameters or scaling");
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / ROT);
    const RopeKernelArgs k = scaling.kernel_args(ROT);   // none: the identity constants
    const float fsf = k.freq_scale, cl = k.corr_low, ch = k.corr_high, ef = k.ext_factor, ms = k.attn_factor;
    ze::Stream* const st = ze::stream_of(stream);
    const int32_t* mtab = mrope_table(ze::stream_of(stream)->device());
    const RopeTab rt = rope_table_for(scaling, ze::stream_of(stream)->device());
    if (p0 == 0) {
        if (rt.cos != nullptr)
        {
            k_native_qsa_indexer_first_tab(st, {1, 1, 1}, {THREADS, 1, 1}, 0, raw, gamma, epsilon, b.dead, b.pooled,
                                           theta_scale, fsf, cl, ch, ef, ms, mtab, rt);
        } else {
            k_native_qsa_indexer_first_plain(st, {1, 1, 1}, {THREADS, 1, 1}, 0, raw, gamma, epsilon, b.dead, b.pooled,
                                             theta_scale, fsf, cl, ch, ef, ms, mtab, rt);
        }
    }
    // completed blocks: those whose last cell (4b+3) lies in [p0, p0 + n)
    const int64_t first = p0 <= R - 1 ? 0 : (p0 - (R - 1) + R - 1) / R;       // the smallest b with 4b+3 >= p0
    const int64_t hi = p0 + n - 1 >= R - 1 ? (p0 + n - 1 - (R - 1)) / R : -1;   // the largest b with 4b+3 <= p0+n-1
    if (hi >= first && rt.cos != nullptr)
    {
        k_native_qsa_indexer_blocks_tab(st, {(unsigned) (hi - first + 1), 1, 1}, {THREADS, 1, 1}, 0, raw, n, p0, pos_base,
                                        gamma, epsilon, b.tail, b.dead, b.pooled, b.block_pos, first, hi, theta_scale,
                                        fsf, cl, ch, ef, ms, mtab, rt);
    } else if (hi >= first)
    {
        k_native_qsa_indexer_blocks_plain(st, {(unsigned) (hi - first + 1), 1, 1}, {THREADS, 1, 1}, 0, raw, n, p0,
                                          pos_base, gamma, epsilon, b.tail, b.dead, b.pooled, b.block_pos, first, hi,
                                          theta_scale, fsf, cl, ch, ef, ms, mtab, rt);
    }
    k_native_qsa_indexer_tail(st, {R - 1, 1, 1}, {D, 1, 1}, 0, raw, n, p0, b.tail);
}
} // namespace strata::kernels
