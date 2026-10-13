// shared_expert.kern.inc - entry points of shared_expert.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(shared_expert_swiglu, (const float* gate, const float* up, float* out, int n), swiglu_kernel(gate, up, out, n))
STRATA_KERNEL(shared_expert_native_swiglu, (const float* gate, const float* up, float* out, int n),
              native_swiglu_kernel(gate, up, out, n))
STRATA_KERNEL(shared_expert_scalar_gate, (const uint16_t* x_bf16, const uint16_t* w_bf16, float* out, int n_embd),
              scalar_gate_kernel(x_bf16, w_bf16, out, n_embd))
STRATA_KERNEL(shared_expert_scale, (float* out, const float* g, int n), scale_kernel(out, g, n))
STRATA_KERNEL(shared_expert_native_scalar_sigmoid, (float* gate), native_scalar_sigmoid_kernel(gate))
STRATA_KERNEL(shared_expert_moe_combine, (const float* parts, const float* weights, const float* shared, float* y,
                                          int n_embd, int k, int has_shared),
              moe_combine_kernel(parts, weights, shared, y, n_embd, k, has_shared))
STRATA_KERNEL(shared_expert_scale_rows, (float* out, const float* g, int n), scale_rows_kernel(out, g, n))
STRATA_KERNEL(shared_expert_sigmoid_scale_rows, (float* out, const float* g, int n), sigmoid_scale_rows_kernel(out, g, n))
STRATA_KERNEL(shared_expert_sigmoid_scale_rows_vec4, (float* out, const float* g, int n4),
              sigmoid_scale_rows_vec4_kernel(reinterpret_cast<sycl::float4*>(out), g, n4))
