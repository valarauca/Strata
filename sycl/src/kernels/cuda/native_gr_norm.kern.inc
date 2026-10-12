// native_gr_norm.kern.inc - entry points of native_gr_norm.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(native_gr_norm_256, (const float* input, const float* gamma, float* output, int n_cols, float epsilon),
              weighted_rms_norm<256>(input, gamma, output, n_cols, epsilon))
STRATA_KERNEL(native_gr_norm_1024, (const float* input, const float* gamma, float* output, int n_cols, float epsilon),
              weighted_rms_norm<1024>(input, gamma, output, n_cols, epsilon))
