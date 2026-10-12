// src/prefill/q4pack.host.cpp - the host side of q4pack.dp.cpp: the launcher.
#include "strata/prefill/q4pack.hpp"
#include "strata/ze/kernel_abi.hpp"

namespace strata::prefill {

#include "q4pack.kern.inc"

void q4_repack(const void* blocks, int64_t rows, int64_t K, uint8_t* qs, uint16_t* scales, int64_t ld_scales,
               int64_t row_step, int64_t row_offset, void* stream) {
    const int64_t n = rows * (K / 32);
    if (n <= 0) return;
    const uint32_t threads = 256;
    k_q4_repack(ze::stream_of(stream), {(uint32_t) ((n + threads - 1) / threads), 1, 1}, {threads, 1, 1}, 0,
                (const uint8_t*) blocks, rows, K, qs, scales, ld_scales, row_step, row_offset);
}

}  // namespace strata::prefill
