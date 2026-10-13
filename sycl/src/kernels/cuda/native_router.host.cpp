// src/kernels/cuda/native_router.host.cpp - the host side of native_router.dp.cpp: checks and the launchers.
//
// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
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
#include "strata/kernels/native_router.hpp"
#include "strata/ze/kernel_abi.hpp"
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {

#include "native_router.kern.inc"

namespace {
std::atomic<bool> enabled{false};
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    k_native_router_route_512(ze::stream_of(stream), {1, 1, 1}, {32, 8, 1}, 0, logits, ids, weights);
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    // SYCL port: one block per token running the single-token route<512> (the same kernel as the 256-expert multi
    // launch below).  Upstream's one-warp-per-token route_multi gives the same bits (native_multi_parity checks
    // that) but hung an Arc A750 (Alchemist) on its first launch.
    k_native_router_route_512(ze::stream_of(stream), {(unsigned) n_tok, 1, 1}, {32, 8, 1}, 0, logits, ids, weights);
}

void native_router_top10_multi_ne(const float* logits, int32_t* ids, float* weights, int n_tok, int n_expert, void* stream) {
    if (n_expert == 512) { native_router_top10_multi(logits, ids, weights, n_tok, stream); return; }
    if (!stream || n_tok < 1 || n_expert != 256 || !valid(logits, (size_t) n_tok * 256 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi, 256): a stream, 256 experts and aligned [n,256]/[n,10] buffers");
    k_native_router_route_256(ze::stream_of(stream), {(unsigned) n_tok, 1, 1}, {32, 8, 1}, 0, logits, ids, weights);
}
}
