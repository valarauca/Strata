// native_gdn.kern.inc - entry points of native_gdn.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(native_gdn_step, (float* state, const float* q, const float* k, const float* v, const float* gate,
                                const float* beta, float* output, int h_k, int h_v, float scale),
              step(state, q, k, v, gate, beta, output, h_k, h_v, scale))
