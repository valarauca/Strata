// src/kernels/cuda/s2_gemv_q8.host.cpp - the host side of s2_gemv_q8.dp.cpp: the launcher.
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "s2_gemv_q8.kern.inc"

namespace {

constexpr int QK_S2 = 64;
constexpr int QK8_0 = 32;

}  // namespace

void s2_gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                int64_t n_out, int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK8_0 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_q8: n_in %lld must be a multiple of %d\n", (long long) n_in, QK_S2);
        std::exit(1);
    }
    const size_t smem = (size_t)threads_per_row * sizeof(float);
    k_s2_gemv_q8(ze::stream_of(stream), {(unsigned) n_out, 1, 1}, {(unsigned) threads_per_row, 1, 1}, smem, act, codes,
                 scales, y, n_in, n_out, threads_per_row);

    if (stream == nullptr) ze::default_stream().sync();
}

}  // namespace strata::kernels
