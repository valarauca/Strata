// native_gr_postops.kern.inc - entry points of native_gr_postops.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(native_gr_postops_down_silu, (float* lo, int count, float scale), down_silu(lo, count, scale))
STRATA_KERNEL(native_gr_postops_pre_gated, (const float* xn, float* gate, float* mixed, int n_embd, int hc, float scale),
              pre_gated<false>(xn, gate, mixed, n_embd, hc, scale))
STRATA_KERNEL(native_gr_postops_pre_gated_fused,
              (const float* xn, float* gate, float* mixed, int n_embd, int hc, float scale),
              pre_gated<true>(xn, gate, mixed, n_embd, hc, scale))
STRATA_KERNEL(native_gr_postops_post, (const float* residual, const float* block_out, const float* inject,
                                       float* output, int n_embd, int hc, float scale),
              post(residual, block_out, inject, output, n_embd, hc, scale))
