// src/kernels/cuda/native_ple_postops.host.cpp - the host side of native_ple_postops.dp.cpp: span checks and launchers.
// Arithmetic adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp and ggml-cuda/{reduce_rows.cuh,sumrows.cu,unary.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/ze/kernel_abi.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {

#include "native_ple_postops.kern.inc"

namespace {
constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;

struct Span { const void* p; size_t bytes; size_t alignment; };
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.bytes && y < x + a.bytes;
}
void validate(Span span) {
    const auto p = reinterpret_cast<uintptr_t>(span.p);
    if (!p || p % span.alignment || p > std::numeric_limits<uintptr_t>::max() - span.bytes)
        throw std::invalid_argument("native PLE postops require nonnull aligned bounded spans");
}
void launch_check() {
}
} // namespace

void native_ple_postops(const float* projected_key, const float* hidden,
                        const float* value, const float* history,
                        const PleWeights& w, const NativePlePostopsBuffers& b, void* stream) {
    if (!stream) throw std::invalid_argument("native PLE postops require an explicit stream");
    const Span inputs[] = {{projected_key,D*4,4}, {hidden,D*4,4}, {value,N*4,4},
        {history,HISTORY*D*4,4}, {w.norm_key,D*4,4}, {w.norm_query,D*4,4},
        {w.norm_conv,D*4,4}, {w.conv1d_f16,4*D*2,2}};
    const Span outputs[] = {{b.key,D*4,4}, {b.query,D*4,4}, {b.gate,H*4,4},
        {b.gated,D*4,4}, {b.normalized,D*4,4}, {b.conv,D*4,4}, {b.result,D*4,4}};
    for (const auto& span : inputs) validate(span);
    for (const auto& span : outputs) validate(span);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 8; ++j)
            if (!(i == 6 && j == 1 && b.result == hidden) && overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("native PLE postops output overlaps an input or weight");
        for (size_t j = i + 1; j < 7; ++j)
            if (!(i == 1 && j == 4 && b.query == b.normalized) && overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("native PLE postops writable spans overlap");
    }
    native_gr_rms_norm_weighted(projected_key,w.norm_key,b.key,N,H,NG_RMS_EPS,stream);
    native_gr_rms_norm_weighted(hidden,w.norm_query,b.query,N,H,NG_RMS_EPS,stream);
    auto st = ze::stream_of(stream);
    k_native_ple_postops_gate(st, {H, 1, 1}, {512, 1, 1}, 0, b.key, b.query, b.gate, 1.0f / std::sqrt(float(N)));
    k_native_ple_postops_broadcast(st, {D / 256, 1, 1}, {256, 1, 1}, 0, value, b.gate, b.gated);
    launch_check();
    native_gr_rms_norm_weighted(b.gated,w.norm_conv,b.normalized,N,H,NG_RMS_EPS,stream);
    k_native_ple_postops_conv_residual(st, {D / 256, 1, 1}, {256, 1, 1}, 0, history, b.normalized, w.conv1d_f16, hidden,
                                       b.gated, b.conv, b.result);
    launch_check();
}

void native_ple_postops_batch(float* key, float* hidden, const float* value, float* history, const PleWeights& w,
                              float* query_norm, float* gated, float* gate, int T, void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate)
        throw std::invalid_argument("native PLE postops batch: null input or empty batch");
    auto st = ze::stream_of(stream);
    const unsigned rows = unsigned(T) * H;
    const unsigned blocks = unsigned((size_t(T) * D + 255) / 256);
    k_native_ple_postops_rms_rep(st, {rows, 1, 1}, {1024, 1, 1}, 0, key, w.norm_key, key);
    k_native_ple_postops_rms_rep(st, {rows, 1, 1}, {1024, 1, 1}, 0, hidden, w.norm_query, query_norm);
    k_native_ple_postops_gate(st, {rows, 1, 1}, {512, 1, 1}, 0, key, query_norm, gate, 1.0f / std::sqrt(float(N)));
    k_native_ple_postops_broadcast_batch(st, {blocks, 1, 1}, {256, 1, 1}, 0, value, gate, gated, T);
    k_native_ple_postops_rms_rep(st, {rows, 1, 1}, {1024, 1, 1}, 0, gated, w.norm_conv, query_norm);
    k_native_ple_postops_conv_residual_batch(st, {blocks, 1, 1}, {256, 1, 1}, 0, history, query_norm, w.conv1d_f16,
                                             hidden, gated, T);
    k_native_ple_postops_history_batch(st, {D / 256, 1, 1}, {256, 1, 1}, 0, history, query_norm, T);
    launch_check();
}
} // namespace strata::kernels
