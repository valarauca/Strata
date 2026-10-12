// elementwise.kern.inc - entry points of elementwise.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(elementwise_embedding_gather, (const uint8_t* codes, const float* scales, const float* offsets, int64_t n,
                                             int code_bits, int code_bias, int group_elems, float* out),
              embedding_gather_kernel(codes, scales, offsets, n, code_bits, code_bias, group_elems, out))
STRATA_KERNEL(elementwise_gdn_gate, (const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t h_v),
              gdn_gate_kernel(alpha, dt, ssm_a, gate, h_v))
STRATA_KERNEL(elementwise_scale, (float* x, int64_t n, float s), scale_kernel(x, n, s))
STRATA_KERNEL(elementwise_add, (float* dst, const float* src, long long n), add_kernel(dst, src, n))
STRATA_KERNEL(elementwise_to_f16, (const float* x, uint16_t* y, int64_t n), to_f16_kernel(x, y, n))
STRATA_KERNEL(elementwise_to_bf16, (const float* x, uint16_t* y, int64_t n), to_bf16_kernel(x, y, n))
STRATA_KERNEL(elementwise_silu, (float* x, int64_t n), silu_kernel(x, n))
STRATA_KERNEL(elementwise_doorbell_wait, (const uint32_t* flag, const uint32_t* seq, uint32_t spin_max),
              doorbell_wait_kernel(flag, seq, spin_max))
STRATA_KERNEL(elementwise_scatter_rows, (const float* src, float* dst, const int32_t* rows, int64_t w4),
              scatter_rows_kernel((const sycl::float4*) src, (sycl::float4*) dst, rows, w4))
STRATA_KERNEL(elementwise_copy_rows_from_mapped, (float* dst, const float* src, int64_t row4, const int32_t* hit_rows,
                                                  const int32_t* count),
              copy_rows_from_mapped_kernel((sycl::float4*) dst, (const sycl::float4*) src, row4, hit_rows, count))
STRATA_KERNEL(elementwise_copy_from_mapped, (float* dst, const float* src, int64_t n4),
              copy_from_mapped_kernel((sycl::float4*) dst, (const sycl::float4*) src, n4))
STRATA_KERNEL(elementwise_doorbell_publish_value, (const float* x, const int32_t* ids, const float* w, int n, int k,
                                                   float* x_out, int32_t* ids_out, float* w_out, uint32_t* seq,
                                                   uint32_t value, bool plain, bool pair),
              doorbell_publish_value_kernel(x, ids, w, n, k, x_out, ids_out, w_out, seq, value, plain, pair))
STRATA_KERNEL(elementwise_doorbell_publish, (const float* x, const int32_t* ids, const float* w, int n, int k, float* x_out,
                                             int32_t* ids_out, float* w_out, uint32_t* seq, bool plain, bool pair),
              doorbell_publish_kernel(x, ids, w, n, k, x_out, ids_out, w_out, seq, plain, pair))
STRATA_KERNEL(elementwise_doorbell_publish_res, (const float* x, const int32_t* ids, const int32_t* d_res, int n_expert,
                                                 int n, int k, float* x_out, int32_t* ids_out, uint32_t* seq, bool full,
                                                 bool plain, bool pair),
              doorbell_publish_res_kernel(x, ids, d_res, n_expert, n, k, x_out, ids_out, seq, full, plain, pair))
STRATA_KERNEL(elementwise_copy_from_mapped_multi, (MappedCopies a), copy_from_mapped_multi_kernel(a))
STRATA_KERNEL(elementwise_copy_i32_from_mapped, (int32_t* dst, const int32_t* src, int n),
              copy_i32_from_mapped_kernel(dst, (const volatile int32_t*) src, n))
STRATA_KERNEL(elementwise_doorbell_ring, (uint32_t* seq), doorbell_ring_kernel(seq))
STRATA_KERNEL(elementwise_rms_norm_weighted, (float* x, const float* w, int64_t rows, int64_t cols, float eps),
              rms_norm_weighted_kernel(x, w, rows, cols, eps))
