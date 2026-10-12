// s_gemv.kern.inc - entry points of s_gemv.dp.cpp (strata/ze/kernel_abi.hpp).
#define S_GEMV(CB)                                                                                                   \
    STRATA_KERNEL(s_gemv_##CB, (const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,    \
                                float* y, long long n_in, long long n_out, int bias, int codebook, int group_elems,   \
                                int has_offset),                                                                     \
                  s_gemv_kernel<CB>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_elems, has_offset))
#define S_GEMV_SPLIT(CB)                                                                                             \
    STRATA_KERNEL(s_gemv_split_##CB, (const uint16_t* x, const uint8_t* codes, const float* scales,                  \
                                      const float* offset, float* y, long long n_in, long long n_out, int bias,      \
                                      int codebook, int group_elems, int group_shift, int has_offset,                \
                                      int threads_per_row),                                                          \
                  s_gemv_split_kernel<CB>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_elems,     \
                                          group_shift, has_offset, threads_per_row, strata_slm<uint8_t>()))
#define S_GEMV_Q8K(CB)                                                                                               \
    STRATA_KERNEL(s_gemv_q8k_##CB, (const uint8_t* x, const uint8_t* codes, const float* scales, const float* offset, \
                                    float* y, long long n_in, long long n_out, int bias, int codebook,               \
                                    int group_elems, int has_offset),                                                \
                  s_gemv_q8k_kernel<CB>(x, codes, scales, offset, y, n_in, n_out, bias, codebook, group_elems,       \
                                        has_offset))
#define S_GEMV_Q8_SPLIT(CB, Q8K, NAME)                                                                               \
    STRATA_KERNEL(s_gemv_##NAME##_split_##CB, (const uint8_t* x, const uint8_t* codes, const float* scales,          \
                                               const float* offset, float* y, long long n_in, long long n_out,       \
                                               int bias, int codebook, int group_shift, int has_offset),             \
                  s_gemv_q8_split_kernel<CB, Q8K>(x, codes, scales, offset, y, n_in, n_out, bias, codebook,          \
                                                  group_shift, has_offset))
S_GEMV(2)
S_GEMV(4)
S_GEMV(8)
S_GEMV_SPLIT(2)
S_GEMV_SPLIT(4)
S_GEMV_SPLIT(8)
S_GEMV_Q8K(4)
S_GEMV_Q8K(8)
S_GEMV_Q8_SPLIT(4, true, q8k)
S_GEMV_Q8_SPLIT(8, true, q8k)
S_GEMV_Q8_SPLIT(4, false, q8_0)
S_GEMV_Q8_SPLIT(8, false, q8_0)
#undef S_GEMV
#undef S_GEMV_SPLIT
#undef S_GEMV_Q8K
#undef S_GEMV_Q8_SPLIT
