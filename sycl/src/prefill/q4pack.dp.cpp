// src/prefill/q4pack.dp.cpp - include/strata/prefill/q4pack.hpp: one work-item per Q4_0 block. Output byte j of a
// block holds elements 2j and 2j + 1, which sit in the low nibbles of source bytes 2j, 2j + 1 for j < 8 and in the high
// nibbles of source bytes 2j - 16, 2j - 15 after that; XOR 8 turns q into the signed q - 8.
#include "strata/port.hpp"

namespace strata::prefill {
namespace {

inline void q4_repack_kernel(const uint8_t* blocks, long long rows, long long K, uint8_t* qs, uint16_t* scales,
                             long long ld_scales, long long row_step, long long row_offset) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long nb = K / 32;
    const long long i = (long long) it.get_group(2) * it.get_local_range(2) + it.get_local_id(2);
    if (i >= rows * nb) return;
    const long long r = i / nb, g = i % nb;
    const uint8_t* src = blocks + i * 18;
    const long long dr = r * row_step + row_offset;
    uint8_t* dst = qs + dr * (K / 2) + g * 16;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int lo = j < 8 ? (src[2 + 2 * j] & 0xF) : (src[2 + 2 * j - 16] >> 4);
        const int hi = j < 8 ? (src[3 + 2 * j] & 0xF) : (src[3 + 2 * j - 16] >> 4);
        dst[j] = (uint8_t) ((lo ^ 8) | ((hi ^ 8) << 4));
    }
    scales[g * ld_scales + dr] = (uint16_t) (src[0] | (src[1] << 8));
}

}  // namespace

#include "q4pack.kern.inc"

}  // namespace strata::prefill
