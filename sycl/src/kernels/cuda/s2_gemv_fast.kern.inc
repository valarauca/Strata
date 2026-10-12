// s2_gemv_fast.kern.inc - entry points of s2_gemv_fast.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(s2_gemv_fast_staged, (const uint16_t* x, const uint8_t* codes, const float* scales, float* y,
                                    long long n_in, long long n_out, int threads_per_row),
              s2_gemv_fast_kernel<true>(x, codes, scales, y, n_in, n_out, threads_per_row, strata_slm<uint8_t>(),
                                        c_codes_lut.v))
STRATA_KERNEL(s2_gemv_fast_direct, (const uint16_t* x, const uint8_t* codes, const float* scales, float* y,
                                    long long n_in, long long n_out, int threads_per_row),
              s2_gemv_fast_kernel<false>(x, codes, scales, y, n_in, n_out, threads_per_row, strata_slm<uint8_t>(),
                                         c_codes_lut.v))
