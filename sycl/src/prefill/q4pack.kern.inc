// q4pack.kern.inc - entry points of q4pack.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(q4_repack, (const uint8_t* blocks, long long rows, long long K, uint8_t* qs, uint16_t* scales,
                          long long ld_scales, long long row_step, long long row_offset),
              q4_repack_kernel(blocks, rows, K, qs, scales, ld_scales, row_step, row_offset))
