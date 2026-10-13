// dequant_s2.kern.inc - entry points of dequant_s2.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(dequant_s2, (const uint8_t* codes, const float* scales, float* out, long long n_blocks),
              dequant_s2_kernel(codes, scales, out, n_blocks))
