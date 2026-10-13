// gdn.kern.inc - entry points of gdn.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(gdn_step, (float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, float* o, int S, int h_k, int h_v),
              gdn_step_kernel(state, q, k, v, gate, beta, o, S, h_k, h_v))
STRATA_KERNEL(gdn_conv, (float* cs, const float* x, const float* kW, float* out, int C, int dc),
              gdn_conv_kernel(cs, x, kW, out, C, dc))
STRATA_KERNEL(gdn_l2, (float* x, int cols, float eps), gdn_l2_kernel(x, cols, eps))
STRATA_KERNEL(gdn_out_norm, (const float* o, const float* z, const float* ssm_norm, float* y, int S, float eps),
              gdn_out_norm_kernel(o, z, ssm_norm, y, S, eps))
STRATA_KERNEL(gdn_beta_gate, (float* beta, int n), gdn_beta_gate_kernel(beta, n))
