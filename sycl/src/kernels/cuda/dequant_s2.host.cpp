// src/kernels/cuda/dequant_s2.host.cpp - the host side of dequant_s2.dp.cpp: the launcher.
#include "strata/kernels/dequant_s2.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "dequant_s2.kern.inc"

void dequant_s2(const uint8_t *codes, const float *scales, float *out,
                int64_t n_blocks) {
    if (n_blocks <= 0) return;
    const int threads = 256;
    const long long blocks = (n_blocks + threads - 1) / threads;
    k_dequant_s2(&ze::default_stream(), {(unsigned) blocks, 1, 1}, {threads, 1, 1}, 0, codes, scales, out, n_blocks);
    ze::device_sync();
}

}  // namespace strata::kernels
