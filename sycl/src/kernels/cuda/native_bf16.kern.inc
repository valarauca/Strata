// native_bf16.kern.inc - entry points of native_bf16.dp.cpp (strata/ze/kernel_abi.hpp).
#define NBF16_MULTI(N, NT, EXACT)                                                                                   \
    STRATA_KERNEL(native_bf16_multi_##N##_##NT, (const float* x, int64_t ldx, const uint16_t* w, float* y,          \
                                                 int64_t ldy, int n_in, int n_tok),                                 \
                  bf16_f32_mmvf_multi_kernel<N, NT, EXACT>(x, ldx, w, y, ldy, n_in, n_tok))
#define NBF16_ROWS(B, NT)                                                                                           \
    STRATA_KERNEL(native_bf16_rows_##B##_##NT, (const float* x, int64_t ldx, const uint16_t* w, float* y,           \
                                                int64_t ldy, int n_in, int n_out, int n_tok),                       \
                  bf16_f32_mmvf_rows_kernel<B, NT, 4>(x, ldx, w, y, ldy, n_in, n_out, n_tok))                       \
    STRATA_KERNEL(native_bf16_rows_ts_##B##_##NT, (const float* x, int64_t ldx, const uint16_t* w, float* y,        \
                                                   int64_t ldy, int n_in, int n_out, int n_tok),                    \
                  bf16_f32_mmvf_rows_kernel<B, NT, 4, true>(x, ldx, w, y, ldy, n_in, n_out, n_tok))                 \
    STRATA_KERNEL(native_bf16_rows_aux_##B##_##NT, (const float* x, int64_t ldx, const uint16_t* w, float* y,       \
                                                    int64_t ldy, int n_in, int n_out, int n_tok,                    \
                                                    const uint16_t* w_aux, float* y_aux, int64_t ldy_aux),          \
                  bf16_f32_mmvf_rows_kernel<B, NT, 4, false, true>(x, ldx, w, y, ldy, n_in, n_out, n_tok, w_aux,    \
                                                                   y_aux, ldy_aux))                                 \
    STRATA_KERNEL(native_bf16_rows_aux_ts_##B##_##NT, (const float* x, int64_t ldx, const uint16_t* w, float* y,    \
                                                       int64_t ldy, int n_in, int n_out, int n_tok,                 \
                                                       const uint16_t* w_aux, float* y_aux, int64_t ldy_aux),       \
                  bf16_f32_mmvf_rows_kernel<B, NT, 4, true, true>(x, ldx, w, y, ldy, n_in, n_out, n_tok, w_aux,     \
                                                                  y_aux, ldy_aux))
#define NBF16(N)                                                                                                    \
    STRATA_KERNEL(native_bf16_mmvf_##N, (const float* x, const uint16_t* w, float* y, int n_in),                    \
                  bf16_f32_mmvf_kernel<N>(x, w, y, n_in))                                                           \
    NBF16_MULTI(N, 1, true)                                                                                         \
    NBF16_MULTI(N, 2, true)                                                                                         \
    NBF16_MULTI(N, 3, true)                                                                                         \
    NBF16_MULTI(N, 4, true)                                                                                         \
    NBF16_MULTI(N, 5, true)                                                                                         \
    NBF16_MULTI(N, 6, true)                                                                                         \
    NBF16_MULTI(N, 8, false)                                                                                        \
    NBF16_ROWS(N, 4)                                                                                                \
    NBF16_ROWS(N, 8)
NBF16(32)
NBF16(64)
NBF16(96)
NBF16(128)
NBF16(160)
NBF16(192)
NBF16(224)
NBF16(256)
#undef NBF16
#undef NBF16_ROWS
#undef NBF16_MULTI
