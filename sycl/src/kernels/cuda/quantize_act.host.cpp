// src/kernels/cuda/quantize_act.host.cpp - the host side of quantize_act.dp.cpp: the launchers.
#include "strata/kernels/quantize_act.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "quantize_act.kern.inc"

namespace {

constexpr int QK8_0 = 32;
constexpr int QK_K = 256;

}  // namespace

void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    k_quantize_act_q8_0(ze::stream_of(stream), {grid, 1, 1}, {threads, 1, 1}, 0, x, blocks, nb);
    if (stream == nullptr) ze::default_stream().sync();
}

/// See `quantize_q8_0_scaled_kernel`.  Writes the same 34-byte `block_q8_0` layout as `quantize_q8_0`, plus
/// `scales[n/32]` carrying the fp32 `s` the CPU path used - so a hit can be computed with the CPU's
/// multiplier instead of the block's fp16 `d`.  `scales` must not be null.
void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0_scaled: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
        std::exit(1);
    }
    if (scales == nullptr) {
        std::fprintf(stderr, "quantize_q8_0_scaled: scales is null\n");
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    const int threads = 128;
#if defined(__HIPCC__)
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
#else
    const unsigned grid = (unsigned) ((nb + threads / 32 - 1) / (threads / 32));   // a warp a block
#endif
    k_quantize_act_q8_0_scaled(ze::stream_of(stream), {grid, 1, 1}, {threads, 1, 1}, 0, x, blocks, scales, nb);
    if (stream == nullptr) ze::default_stream().sync();
}

void dequant_q8_0(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const long long nb = n / QK8_0;
    const int threads = 128;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    k_quantize_act_dequant_q8_0(ze::stream_of(stream), {grid, 1, 1}, {threads, 1, 1}, 0, blocks, x, nb);
    if (stream == nullptr) ze::default_stream().sync();
}

void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK_K != 0) {
        std::fprintf(stderr, "quantize_q8_K: n %lld is not a multiple of %d\n", (long long) n, QK_K);
        std::exit(1);
    }
    const long long nb = n / QK_K;
    const int threads = 64;                       // one block per thread, and a block is 256 elements
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    k_quantize_act_q8_K(ze::stream_of(stream), {grid, 1, 1}, {threads, 1, 1}, 0, x, blocks, nb);
    if (stream == nullptr) ze::default_stream().sync();
}

void dequant_q8_K(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const long long nb = n / QK_K;
    const int threads = 64;
    const unsigned grid = (unsigned) ((nb + threads - 1) / threads);
    k_quantize_act_dequant_q8_K(ze::stream_of(stream), {grid, 1, 1}, {threads, 1, 1}, 0, blocks, x, nb);
    if (stream == nullptr) ze::default_stream().sync();
}

}  // namespace strata::kernels
