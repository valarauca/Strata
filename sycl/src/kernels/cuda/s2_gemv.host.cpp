// src/kernels/cuda/s2_gemv.host.cpp - the host side of s2_gemv.dp.cpp: the launcher.
#include "strata/kernels/s2_gemv.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "s2_gemv.kern.inc"

namespace {

constexpr int QK = 64;

}  // namespace

void s2_gemv(const uint16_t *x, const uint8_t *codes, const float *scales,
             float *y, int64_t n_in, int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK != 0) {
        std::fprintf(stderr, "s2_gemv: n_in %lld is not a multiple of %d\n", (long long) n_in, QK);
        std::exit(1);
    }
    const int threads = 128;
    const long long blocks = (n_out + threads - 1) / threads;
    k_s2_gemv(&ze::default_stream(), {(unsigned) blocks, 1, 1}, {threads, 1, 1}, 0, x, codes, scales, y, n_in, n_out);
    ze::device_sync();
}

}  // namespace strata::kernels
