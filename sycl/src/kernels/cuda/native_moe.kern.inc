// native_moe.kern.inc - entry points of native_moe.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(native_moe_combine_k10_vec4, (const float* parts, const float* weights, const float* shared,
                                            float* output, int64_t n4),
              combine_k10_vec4((const sycl::float4*) parts, weights, (const sycl::float4*) shared,
                               (sycl::float4*) output, n4))
STRATA_KERNEL(native_moe_combine, (const float* parts, const float* weights, const float* shared, float* output,
                                   int64_t n_embd, int k),
              combine(parts, weights, shared, output, n_embd, k, nullptr))
STRATA_KERNEL(native_moe_combine_zadd, (const float* parts, const float* weights, const float* shared, float* output,
                                        int64_t n_embd, int k),
              combine<true>(parts, weights, shared, output, n_embd, k, nullptr))
STRATA_KERNEL(native_moe_combine_zadd_gated, (const float* parts, const float* weights, const float* shared,
                                              float* output, int64_t n_embd, int k, const float* sg),
              combine<true, true>(parts, weights, shared, output, n_embd, k, sg))
