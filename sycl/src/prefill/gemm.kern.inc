// gemm.kern.inc - entry points of gemm.dp.cpp (strata/ze/kernel_abi.hpp). <T>_<SGM>x<SGN>_<MT>x<NT>: a work-group of
// SGM x SGN sub-groups, each with MT x NT tiles of 8 x 16.
#define STRATA_XGEMM(name, T, SGM, SGN, MT, NT)                                                                          \
    STRATA_KERNEL_SG(16, xgemm_##name,                                                                                    \
                     (const uint16_t* X, long long ldx, const uint16_t* W, long long ldw, float* Y, long long ldy, int M, \
                      int N, int K, int accumulate),                                                                      \
                     gemm_tile<T, SGM, SGN, MT, NT>((const T*) X, ldx, (const T*) W, ldw, Y, ldy, M, N, K, accumulate))
STRATA_XGEMM(f16_4x4_4x4, half, 4, 4, 4, 4)
STRATA_XGEMM(f16_2x4_4x2, half, 2, 4, 4, 2)
STRATA_XGEMM(f16_1x8_4x1, half, 1, 8, 4, 1)
STRATA_XGEMM(bf16_4x4_4x4, bf16, 4, 4, 4, 4)
STRATA_XGEMM(bf16_2x4_4x2, bf16, 2, 4, 4, 2)
STRATA_XGEMM(bf16_1x8_4x1, bf16, 1, 8, 4, 1)
#undef STRATA_XGEMM
STRATA_KERNEL_SG(16, xgemm_narrow_f16, (const uint16_t* X, long long ldx, const uint16_t* W, long long ldw, float* Y,
                                        long long ldy, int M, int N, int K, int accumulate),
                 gemm_narrow<half>((const half*) X, ldx, (const half*) W, ldw, Y, ldy, M, N, K, accumulate))
STRATA_KERNEL_SG(16, xgemm_narrow_bf16, (const uint16_t* X, long long ldx, const uint16_t* W, long long ldw, float* Y,
                                         long long ldy, int M, int N, int K, int accumulate),
                 gemm_narrow<bf16>((const bf16*) X, ldx, (const bf16*) W, ldw, Y, ldy, M, N, K, accumulate))
