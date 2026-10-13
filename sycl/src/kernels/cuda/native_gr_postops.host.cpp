// src/kernels/cuda/native_gr_postops.host.cpp - the host side of native_gr_postops.dp.cpp: checks and the launchers.
//
// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{dsv4-hc.cu,scale.cu,unary.cu,unary.cuh}.
//
// MIT License
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

#include "strata/kernels/native_gr_postops.hpp"
#include "strata/ze/kernel_abi.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <cmath>

namespace strata::kernels {

#include "native_gr_postops.kern.inc"

namespace {
constexpr int THREADS = 256;
void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float))
        throw std::invalid_argument("native GR postops require non-null four-byte aligned pointers");
}
void check_shape(int n, int hc) {
    if (n <= 0 || hc <= 0 || std::uint64_t(n) * hc > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native GR postops require positive bounded dimensions");
}
unsigned blocks(std::size_t n) { return unsigned((n + THREADS - 1) / THREADS); }
} // namespace

void native_gr_down_silu(float* lo, int hc_lr, int hc, void* stream) {
    check_shape(hc_lr, hc);
    check_pointer(lo);
    k_native_gr_postops_down_silu(ze::stream_of(stream), {blocks(hc_lr), 1, 1}, {THREADS, 1, 1}, 0, lo, hc_lr,
                                  1.0f / float(hc));
}
void native_gr_pre_gated(const float* xn, float* gate, float* mixed,
                         int n_embd, int hc, bool fused_layer, void* stream) {
    native_gr_pre_gated_multi(xn, gate, mixed, n_embd, hc, 1, fused_layer, stream);
}
void native_gr_pre_gated_multi(const float* xn, float* gate, float* mixed,
                               int n_embd, int hc, int n_tok, bool fused_layer, void* stream) {
    if (n_tok <= 0)
        throw std::invalid_argument("native GR postops require positive n_tok");
    check_shape(n_embd, hc);
    check_pointer(xn); check_pointer(gate); check_pointer(mixed);
    const ze::Dim3 grid{blocks(n_embd), unsigned(n_tok), 1u};
    if (fused_layer)
        k_native_gr_postops_pre_gated_fused(ze::stream_of(stream), grid, {THREADS, 1, 1}, 0, xn, gate, mixed, n_embd,
                                            hc, 1.0f / float(hc));
    else
        k_native_gr_postops_pre_gated(ze::stream_of(stream), grid, {THREADS, 1, 1}, 0, xn, gate, mixed, n_embd, hc,
                                      1.0f / float(hc));
}
void native_gr_post(const float* residual, const float* block_out, const float* inject,
                    float* output, int n_embd, int hc, void* stream) {
    native_gr_post_multi(residual, block_out, inject, output, n_embd, hc, 1, stream);
}
void native_gr_post_multi(const float* residual, const float* block_out, const float* inject,
                          float* output, int n_embd, int hc, int n_tok, void* stream) {
    if (n_tok <= 0)
        throw std::invalid_argument("native GR postops require positive n_tok");
    check_shape(n_embd, hc);
    check_pointer(residual); check_pointer(block_out); check_pointer(inject); check_pointer(output);
    const ze::Dim3 grid{blocks(std::size_t(n_embd) * hc), unsigned(n_tok), 1u};
    k_native_gr_postops_post(ze::stream_of(stream), grid, {THREADS, 1, 1}, 0, residual, block_out, inject, output,
                             n_embd, hc, 1.0f / float(hc));
}
} // namespace strata::kernels
