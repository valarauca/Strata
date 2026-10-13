// include/strata/prefill/q4pack.hpp - Q4_0 rows (ggml block_q4_0: an fp16 scale d, then 16 bytes holding element j in
// the low and element j + 16 in the high nibble, value (q - 8) * d) rewritten into the operands strata::gs::gemm takes
// for Weights::Q4.
#pragma once

#include <cstdint>

namespace strata::prefill {

/// Rewrites `rows` Q4_0 rows of `K` elements (K a multiple of 32) at `blocks` into signed 4-bit rows `qs` (K / 2 bytes
/// each, even k in the low nibble, q - 8 stored) and fp16 `scales` (scales[g * ld_scales + r] for group g of row r), on
/// `stream` (nullptr: the default stream). Source row r lands at row r * row_step + row_offset of both outputs, so two
/// calls with step 2 interleave gate and up rows.
void q4_repack(const void* blocks, int64_t rows, int64_t K, uint8_t* qs, uint16_t* scales, int64_t ld_scales,
               int64_t row_step, int64_t row_offset, void* stream);

}  // namespace strata::prefill
