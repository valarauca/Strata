// s2_gemv_q8.kern.inc - entry points of s2_gemv_q8.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(s2_gemv_q8, (const uint8_t* act, const uint8_t* codes, const float* scales, float* y, long long n_in,
                           long long n_out, int threads_per_row),
              s2_gemv_q8_kernel(act, codes, scales, y, n_in, n_out, threads_per_row, strata_slm<uint8_t>()))
