// ple.kern.inc - entry points of ple.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(ple_history_advance, (float* history, const float* normalized), history_advance_kernel(history, normalized))
STRATA_KERNEL(ple_gnorm, (const float* x, const float* w, float* y, int n_embd, float eps),
              gnorm_kernel(x, w, y, n_embd, eps))
STRATA_KERNEL(ple_gate, (const float* key, const float* query, float* gate, int n_embd, float inv_sqrt_n),
              gate_kernel(key, query, gate, n_embd, inv_sqrt_n))
STRATA_KERNEL(ple_bcast, (const float* value, const float* gate, float* gated, int n_embd, int hc),
              bcast_kernel(value, gate, gated, n_embd, hc))
STRATA_KERNEL(ple_conv, (const float* hist, const float* norm, const uint16_t* kW, float* out, int hc_dim, int kern,
                         int dil, int nhist),
              conv_kernel(hist, norm, kW, out, hc_dim, kern, dil, nhist))
STRATA_KERNEL(ple_add3, (const float* hidden, const float* gated, const float* conv, float* result, int n),
              add3_kernel(hidden, gated, conv, result, n))
STRATA_KERNEL(ple_bf16_gemv, (const uint16_t* x, const uint16_t* w, float* y, int n_in, int n_out),
              bf16_gemv_kernel(x, w, y, n_in, n_out))
STRATA_KERNEL(ple_to_bf16, (const float* x, uint16_t* y, int n), to_bf16_kernel(x, y, n))
