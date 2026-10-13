// native_ple_postops.kern.inc - entry points of native_ple_postops.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(native_ple_postops_gate, (const float* key, const float* query, float* gate, float scale),
              gate_kernel(key, query, gate, scale))
STRATA_KERNEL(native_ple_postops_broadcast, (const float* value, const float* gate, float* gated),
              broadcast_kernel(value, gate, gated))
STRATA_KERNEL(native_ple_postops_conv_residual, (const float* history, const float* normalized, const uint16_t* weights,
                                                 const float* hidden, const float* gated, float* conv, float* result),
              conv_residual_kernel(history, normalized, weights, hidden, gated, conv, result))
STRATA_KERNEL(native_ple_postops_rms_rep, (const float* input, const float* gamma, float* output),
              rms_rep_kernel(input, gamma, output))
STRATA_KERNEL(native_ple_postops_broadcast_batch, (const float* value, const float* gate, float* gated, int T),
              broadcast_batch_kernel(value, gate, gated, T))
STRATA_KERNEL(native_ple_postops_conv_residual_batch, (const float* history, const float* normalized,
                                                       const uint16_t* weights, float* hidden, const float* gated, int T),
              conv_residual_batch_kernel(history, normalized, weights, hidden, gated, T))
STRATA_KERNEL(native_ple_postops_history_batch, (float* history, const float* normalized, int T),
              history_batch_kernel(history, normalized, T))
