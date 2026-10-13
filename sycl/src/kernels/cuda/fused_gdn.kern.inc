// fused_gdn.kern.inc - entry points of fused_gdn.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(fused_gdn_step_norm, (float* state, const float* q, const float* k, const float* v, const float* gate,
                                    const float* beta, const float* z, const float* gamma, float eps, float* y,
                                    int h_k, int h_v),
              gdn_step_norm_kernel(state, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v))
STRATA_KERNEL(fused_gdn_conv_l2, (float* hist, const float* qkv, const float* w, float* h, int qk_heads, float eps),
              gdn_conv_l2_kernel(hist, qkv, w, h, qk_heads, eps))
STRATA_KERNEL(fused_gdn_ab, (const float* x, const uint16_t* wa, const uint16_t* wb, const float* dt,
                             const float* ssm_a, float* gate, float* beta, int n, int h_v),
              gdn_ab_kernel(x, wa, wb, dt, ssm_a, gate, beta, n, h_v))
