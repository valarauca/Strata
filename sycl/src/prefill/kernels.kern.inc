// kernels.kern.inc - entry points of kernels.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(kernels_kv_append, (const float* K, const float* V, int64_t pos0, const int32_t* table, int64_t page_size,
                                  uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale,
                                  uint16_t* v_scale, strata::kernels::KvHostPools host,
                                  strata::kernels::KvHostPools stage, bool wide_host),
              kv_append_kernel(K, V, pos0, table, page_size, k_pool, v_pool, k_q, v_q, k_scale, v_scale, host, stage, wide_host))
STRATA_KERNEL(kernels_to_f16, (const float* x, uint16_t* y, int64_t n),
              to_f16_kernel(x, y, n))
STRATA_KERNEL(kernels_round_f16, (const float* x, float* y, int64_t n),
              round_f16_kernel(x, y, n))
STRATA_KERNEL(kernels_to_bf16, (const float* x, uint16_t* y, uint16_t* ylo, int64_t n),
              to_bf16_kernel(x, y, ylo, n))
STRATA_KERNEL(kernels_gr_norm, (const float* R, const float* w, float eps, float* xn, uint16_t* xn16,
                                uint16_t* xn16_lo),
              gr_norm_kernel(R, w, eps, xn, xn16, xn16_lo))
STRATA_KERNEL(kernels_gr_norm_rs, (const float* R, const float* w, float eps, float* rs_out, uint16_t* xn16,
                                   uint16_t* xn16_lo, int64_t ldx),
              gr_norm_rs_kernel(R, w, eps, rs_out, xn16, xn16_lo, ldx))
STRATA_KERNEL(kernels_gr_mix_r, (const float* R, const float* rs, const float* w, const float* g, float* mixed,
                                 uint16_t* mixed16, int64_t T, uint16_t* mixed_h, uint16_t* mixed16_lo),
              gr_mix_r_kernel(R, rs, w, g, mixed, mixed16, T, mixed_h, mixed16_lo))
STRATA_KERNEL(kernels_gr_write_norm_rs, (float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w,
                                         float eps, float* rs_out, uint16_t* xn16, uint16_t* xn16_lo, int64_t ldx),
              gr_write_norm_rs_kernel(R, bo, inj, inj_ld, w, eps, rs_out, xn16, xn16_lo, ldx))
STRATA_KERNEL(kernels_gr_write_cvec_norm_rs, (float* R, const float* bo, const float* inj, int64_t inj_ld,
                                              const float* v_l, const float* s_l, const int* on, int mode,
                                              const float* w, float eps, float* rs_out, uint16_t* xn16,
                                              uint16_t* xn16_lo, int64_t ldx),
              gr_write_cvec_norm_rs_kernel(R, bo, inj, inj_ld, v_l, s_l, on, mode, w, eps, rs_out, xn16, xn16_lo, ldx))
STRATA_KERNEL(kernels_gr_silu, (const float* lo, uint16_t* lo16, uint16_t* lo16_lo, int64_t n),
              gr_silu_kernel(lo, lo16, lo16_lo, n))
STRATA_KERNEL(kernels_gr_mix, (const float* xn, const float* g, float* mixed, uint16_t* mixed16, int64_t T,
                               uint16_t* mixed_h, uint16_t* mixed16_lo),
              gr_mix_kernel(xn, g, mixed, mixed16, T, mixed_h, mixed16_lo))
STRATA_KERNEL(kernels_gr_write, (float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T),
              gr_write_kernel(R, bo, inj, inj_ld, T))
STRATA_KERNEL(kernels_gr_broadcast, (const float* e, float* R, int64_t T),
              gr_broadcast_kernel(e, R, T))
STRATA_KERNEL(kernels_gdn_gates, (const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta,
                                  int64_t T),
              gdn_gates_kernel(ab, dt, ssm_a, gate, beta, T))
STRATA_KERNEL(kernels_gdn_conv, (float* hist, const float* qkv, const float* w, float* h, int64_t T),
              gdn_conv_kernel(hist, qkv, w, h, T))
STRATA_KERNEL(kernels_gdn_conv_l2, (const float* hist, const float* qkv, const float* w, float* h, int64_t T,
                                    float eps),
              gdn_conv_l2_kernel(hist, qkv, w, h, T, eps))
STRATA_KERNEL(kernels_gdn_conv_hist, (float* hist, const float* qkv, int64_t T),
              gdn_conv_hist_kernel(hist, qkv, T))
STRATA_KERNEL(kernels_gdn_conv_tiled, (const float* hist, const float* qkv, const float* w, float* h, int64_t T),
              gdn_conv_tiled_kernel(hist, qkv, w, h, T))
STRATA_KERNEL(kernels_gdn_l2, (float* h, float eps),
              gdn_l2_kernel(h, eps))
STRATA_KERNEL(kernels_gdn_rec, (float* state, const float* h, const float* gate, const float* beta, const float* z,
                                const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, int64_t ld16),
              gdn_rec_kernel(state, h, gate, beta, z, gamma, eps, y, y16, T, ld16))
STRATA_KERNEL(kernels_gdn_rec_kh, (float* state, const float* h, const float* gate, const float* beta, float* oc_out,
                                   int64_t T),
              gdn_rec_kh_kernel(state, h, gate, beta, oc_out, T))
STRATA_KERNEL(kernels_gdn_rec_cols_pipe, (float* state, const float* h, const float* gate, const float* beta,
                                          float* oc_out, int64_t T),
              gdn_rec_cols_pipe_kernel(state, h, gate, beta, oc_out, T))
STRATA_KERNEL(kernels_gdn_rec_cols, (float* state, const float* h, const float* gate, const float* beta, float* oc_out,
                                     int64_t T),
              gdn_rec_cols_kernel(state, h, gate, beta, oc_out, T))
STRATA_KERNEL(kernels_gdn_out_norm, (const float* z, const float* gamma, float eps, float* y, uint16_t* y16,
                                     int64_t ld16),
              gdn_out_norm_kernel(z, gamma, eps, y, y16, ld16))
STRATA_KERNEL(kernels_route_16, (const float* logits, int32_t* ids, float* wout, int64_t T),
              route_kernel<16>(logits, ids, wout, T))
STRATA_KERNEL(kernels_route_8, (const float* logits, int32_t* ids, float* wout, int64_t T),
              route_kernel<8>(logits, ids, wout, T))
STRATA_KERNEL(kernels_blob_dequant_false, (const uint8_t* blob, uint16_t* gu16, uint16_t* d16),
              blob_dequant_kernel<false>(blob, gu16, d16))
STRATA_KERNEL(kernels_blob_dequant_true, (const uint8_t* blob, uint16_t* gu16, uint16_t* d16),
              blob_dequant_kernel<true>(blob, gu16, d16))
STRATA_KERNEL(kernels_swiglu_il, (const float* gu, uint16_t* h16, int64_t n),
              swiglu_il_kernel(gu, h16, n))
STRATA_KERNEL(kernels_swiglu_pair, (const float* g, const float* u, uint16_t* h16, int64_t n),
              swiglu_pair_kernel(g, u, h16, n))
STRATA_KERNEL(kernels_copy_f4, (float* dst, const float* src, int64_t n4),
              copy_f4_kernel((sycl::float4 *)dst, (const sycl::float4 *)src, n4))
STRATA_KERNEL(kernels_copy_i32, (int32_t* dst, const int32_t* src, int64_t n),
              copy_i32_kernel(dst, src, n))
STRATA_KERNEL(kernels_gather_rows16, (const uint16_t* x, const int32_t* src, uint16_t* dst, int64_t n, int64_t width),
              gather_rows16_kernel(x, src, dst, n, width))
STRATA_KERNEL(kernels_moe_combine4, (const float* Dm, const int32_t* slot, const float* w, const float* shared,
                                     const float* sg, float* bo, int64_t T),
              moe_combine4_kernel(Dm, slot, w, reinterpret_cast<const sycl::float4 *>(shared), sg, reinterpret_cast<sycl::float4 *>(bo), T))
STRATA_KERNEL(kernels_moe_combine, (const float* Dm, const int32_t* slot, const float* w, const float* shared,
                                    const float* sg, float* bo, int64_t T),
              moe_combine_kernel(Dm, slot, w, shared, sg, bo, T))
STRATA_KERNEL(kernels_rms_rows, (float* x, const float* w, int64_t cols, int64_t ld, float eps),
              rms_rows_kernel(x, w, cols, ld, eps))
STRATA_KERNEL(kernels_rope_true, (float* x, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float theta_scale,
                                  float freq_scale, float corr_low, float corr_high, float ext_factor, float mscale,
                                  const int32_t* mtab, strata::kernels::RopeTab rt),
              rope_kernel<true>(x, heads, dim, ld, pos0, theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale, mtab, rt))
STRATA_KERNEL(kernels_rope_false, (float* x, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float theta_scale,
                                   float freq_scale, float corr_low, float corr_high, float ext_factor, float mscale,
                                   const int32_t* mtab, strata::kernels::RopeTab rt),
              rope_kernel<false>(x, heads, dim, ld, pos0, theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale, mtab, rt))
STRATA_KERNEL(kernels_split_q, (const float* qf, float* q, int64_t T),
              split_q_kernel(qf, q, T))
STRATA_KERNEL(kernels_gate_attn, (const float* a, const float* qf, uint16_t* o16, int64_t T, int64_t ld16),
              gate_attn_kernel(a, qf, o16, T, ld16))
