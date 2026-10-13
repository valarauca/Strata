// s2_gemv_quads.kern.inc - entry points of s2_gemv_quads.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(s2_gemv_quads, (const uint16_t* x, const uint8_t* codes, const float* scales, float* y, long long n_in,
                              long long n_out, int threads_per_row),
              s2_gemv_quads_kernel(x, codes, scales, y, n_in, n_out, threads_per_row, strata_slm<uint8_t>()))
