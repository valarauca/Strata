// native_rope.kern.inc - entry points of native_rope.dp.cpp (strata/ze/kernel_abi.hpp).
#define NR_TAB(SUFFIX, TAB)                                                                                         \
    STRATA_KERNEL(native_rope_apply##SUFFIX, (const float* x, float* out, int rows, int width, int n_rot,           \
                                              float theta_scale, float freq_scale, float corr_low, float corr_high, \
                                              float ext_factor, float mscale, const int* positions,                 \
                                              const int32_t* mtab, RopeTab rt),                                     \
                  apply<TAB>(x, out, rows, width, n_rot, theta_scale, freq_scale, corr_low, corr_high, ext_factor,  \
                             mscale, positions, mtab, rt))                                                          \
    STRATA_KERNEL(native_rope_norm_rope##SUFFIX, (const float* x, int in_stride, const float* gamma, float* out,    \
                                                  int n_cols, int n_rot, float epsilon, float theta_scale,          \
                                                  float freq_scale, float corr_low, float corr_high,                \
                                                  float ext_factor, float mscale, const int* positions,             \
                                                  const int32_t* mtab, RopeTab rt),                                 \
                  norm_rope_kernel<TAB>(x, in_stride, gamma, out, n_cols, n_rot, epsilon, theta_scale, freq_scale,  \
                                        corr_low, corr_high, ext_factor, mscale, positions, mtab, rt))
NR_TAB(, false)
NR_TAB(_tab, true)
#undef NR_TAB
