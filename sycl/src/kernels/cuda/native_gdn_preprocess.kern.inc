// native_gdn_preprocess.kern.inc - entry points of native_gdn_preprocess.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(native_gdn_preprocess_conv_silu, (float* history, const float* input, const float* weights,
                                                float* raw_output, float* silu_output, int channels),
              conv_silu(history, input, weights, raw_output, silu_output, channels))
STRATA_KERNEL(native_gdn_preprocess_l2_norm, (float* input, float epsilon, float scale_after),
              l2_norm(input, epsilon, scale_after))
STRATA_KERNEL(native_gdn_preprocess_beta_sigmoid, (float* beta, int count), beta_sigmoid(beta, count))
STRATA_KERNEL(native_gdn_preprocess_gate_softplus, (const float* alpha, const float* dt, const float* ssm_a,
                                                    float* gate, int count),
              gate_softplus(alpha, dt, ssm_a, gate, count))
STRATA_KERNEL(native_gdn_preprocess_out_norm, (const float* input, const float* z, const float* gamma, float* output,
                                               float epsilon),
              out_norm(input, z, gamma, output, epsilon))
