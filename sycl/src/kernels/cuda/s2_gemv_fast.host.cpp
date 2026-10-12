// src/kernels/cuda/s2_gemv_fast.host.cpp - the host side of s2_gemv_fast.dp.cpp: the launcher.
#include "strata/kernels/s_gemv.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "s2_gemv_fast.kern.inc"

namespace {

constexpr int QK_S2 = 64;
constexpr int MAX_SHARED_HALVES = 4096;      // 8 KB of shared for x; n_embd 2560 fits with room

}  // namespace

void s2_gemv_fast(const uint16_t *x, const uint8_t *codes, const float *scales,
                  float *y, int64_t n_in, int64_t n_out, int threads_per_row,
                  bool stage_x) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_fast: n_in %lld must be a multiple of %d\n", (long long) n_in, QK_S2);
        std::exit(1);
    }
    if (stage_x && n_in > MAX_SHARED_HALVES) {
        std::fprintf(stderr, "s2_gemv_fast: n_in %lld exceeds the %d-half shared staging limit\n",
                     (long long) n_in, MAX_SHARED_HALVES);
        std::exit(1);
    }
    const size_t smem = (stage_x ? MAX_SHARED_HALVES * sizeof(uint16_t) : 0) +
                        (size_t)threads_per_row * sizeof(float);
    if (stage_x) {
        k_s2_gemv_fast_staged(&ze::default_stream(), {(unsigned) n_out, 1, 1}, {(unsigned) threads_per_row, 1, 1}, smem,
                              x, codes, scales, y, n_in, n_out, threads_per_row);
    } else {
        k_s2_gemv_fast_direct(&ze::default_stream(), {(unsigned) n_out, 1, 1}, {(unsigned) threads_per_row, 1, 1}, smem,
                              x, codes, scales, y, n_in, n_out, threads_per_row);
    }
    ze::device_sync();
}

}  // namespace strata::kernels
