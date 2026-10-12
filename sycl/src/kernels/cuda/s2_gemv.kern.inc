// s2_gemv.kern.inc - entry points of s2_gemv.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(s2_gemv, (const uint16_t* x, const uint8_t* codes, const float* scales, float* y, long long n_in,
                        long long n_out),
              s2_gemv_kernel(x, codes, scales, y, n_in, n_out))
