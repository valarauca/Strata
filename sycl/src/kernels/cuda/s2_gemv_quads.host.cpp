// src/kernels/cuda/s2_gemv_quads.host.cpp - the host side of s2_gemv_quads.dp.cpp: the launcher.
#include "strata/kernels/s_gemv.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "s2_gemv_quads.kern.inc"

void s2_gemv_quads(const uint16_t *x, const uint8_t *codes, const float *scales,
                   float *y, int64_t n_in, int64_t n_out,
                   int threads_per_row) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0) {
        std::fprintf(stderr, "s2_gemv_quads: n_in %lld is not a multiple of 4\n", (long long) n_in);
        std::exit(1);
    }
    const size_t smem = (size_t)threads_per_row * sizeof(float);
    k_s2_gemv_quads(&ze::default_stream(), {(unsigned) n_out, 1, 1}, {(unsigned) threads_per_row, 1, 1}, smem, x, codes,
                    scales, y, n_in, n_out, threads_per_row);
    ze::device_sync();
}

}  // namespace strata::kernels
