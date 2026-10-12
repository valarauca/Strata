// dequant_bf16.kern.inc - entry points of dequant_bf16.dp.cpp (strata/ze/kernel_abi.hpp).
#define DQ_TY(TY)                                                                                                  \
    STRATA_KERNEL(dequant_bf16_bf16_##TY, (const uint8_t* blocks, int64_t row_bytes, int64_t row0, int64_t rows,  \
                                           int64_t groups_per_row, int64_t ld, uint16_t* out),                    \
                  dequant_kernel<TY, uint16_t>(blocks, row_bytes, row0, rows, groups_per_row, ld, out))           \
    STRATA_KERNEL(dequant_bf16_f16_##TY, (const uint8_t* blocks, int64_t row_bytes, int64_t row0, int64_t rows,   \
                                          int64_t groups_per_row, int64_t ld, uint16_t* out),                     \
                  dequant_kernel<TY, H16>(blocks, row_bytes, row0, rows, groups_per_row, ld, (H16*) out))         \
    STRATA_KERNEL(dequant_bf16_f32_##TY, (const uint8_t* blocks, int64_t row_bytes, int64_t row0, int64_t rows,   \
                                          int64_t groups_per_row, int64_t ld, float* out),                        \
                  dequant_kernel<TY, float>(blocks, row_bytes, row0, rows, groups_per_row, ld, out))
DQ_TY(2)
DQ_TY(6)
DQ_TY(7)
DQ_TY(8)
DQ_TY(11)
DQ_TY(12)
DQ_TY(13)
DQ_TY(14)
DQ_TY(20)
DQ_TY(23)
DQ_TY(42)
#undef DQ_TY
