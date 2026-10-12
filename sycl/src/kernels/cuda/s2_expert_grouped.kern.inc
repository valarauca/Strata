// s2_expert_grouped.kern.inc - entry points of s2_expert_grouped.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(s2_expert_grouped_gu_pair, (const uint8_t* blob_base, const int32_t* slot_index, long long blob_bytes,
                                          const uint8_t* x_q8_0, const float* x_scales, float* gate_up, int n_hits,
                                          const int32_t* d_count, const int32_t* dst_index, int tok_div),
              gu_pair_kernel(blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, n_hits, d_count, dst_index,
                             tok_div))
STRATA_KERNEL(s2_expert_grouped_gu, (const uint8_t* blob_base, const int32_t* slot_index, long long blob_bytes,
                                     const uint8_t* x_q8_0, const float* x_scales, float* gate_up, int n_hits,
                                     const int32_t* d_count, const int32_t* dst_index, int tok_div),
              gu_kernel(blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, n_hits, d_count, dst_index, tok_div))
STRATA_KERNEL(s2_expert_grouped_down_pair, (const uint8_t* blob_base, const int32_t* slot_index,
                                            const int32_t* dst_index, long long blob_bytes, const uint8_t* h_q8_0,
                                            const float* h_scales, float* out, int n_hits, const int32_t* d_count),
              down_pair_kernel(blob_base, slot_index, dst_index, blob_bytes, h_q8_0, h_scales, out, n_hits, d_count))
STRATA_KERNEL(s2_expert_grouped_down, (const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                                       long long blob_bytes, const uint8_t* h_q8_0, const float* h_scales, float* out,
                                       int n_hits, const int32_t* d_count),
              down_kernel(blob_base, slot_index, dst_index, blob_bytes, h_q8_0, h_scales, out, n_hits, d_count))
STRATA_KERNEL(s2_expert_grouped_swiglu, (float* gate_up, long long n_pairs), swiglu_kernel(gate_up, n_pairs))
STRATA_KERNEL(s2_expert_grouped_swiglu_quantize_q8_0_scaled, (float* gate_up, long long n_pairs, uint8_t* blocks,
                                                              float* scales),
              swiglu_quantize_q8_0_scaled_kernel(gate_up, n_pairs, blocks, scales))
STRATA_KERNEL(s2_expert_grouped_hit_select, (const int32_t* ids, const int32_t* res_row, int k, int n_expert,
                                             int32_t* slot, int32_t* dst, int32_t* count),
              hit_select_kernel(ids, res_row, k, n_expert, slot, dst, count))
STRATA_KERNEL(s2_expert_grouped_hit_select_multi, (const int32_t* ids, const int32_t* res_row, int n, int n_expert,
                                                   int32_t* slot, int32_t* dst, int32_t* count),
              hit_select_multi_kernel(ids, res_row, n, n_expert, slot, dst, count))
STRATA_KERNEL(s2_expert_grouped_add_hits, (float* parts, const float* hit_out, const int32_t* dst, const int32_t* count,
                                           int n_embd),
              add_hits_kernel(parts, hit_out, dst, count, n_embd))
STRATA_KERNEL(s2_expert_grouped_group_resident, (const int32_t* ids, int n, int k_per_tok, const uint8_t* base,
                                                 long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                                                 int32_t* counts, int32_t* ent_dst, int32_t* ent_tok),
              group_resident_kernel(ids, n, k_per_tok, base, blob, grp_ptr, grp_start, counts, ent_dst, ent_tok))
STRATA_KERNEL(s2_expert_grouped_gu_grouped_t, (const unsigned long long* grp_ptr, const int32_t* grp_start,
                                               const int32_t* n_groups, const int32_t* ent_tok, const uint8_t* x_q8_0,
                                               const float* x_scales, float* gate_up, int cap_entries),
              gu_grouped_t_kernel(grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales, gate_up, cap_entries))
STRATA_KERNEL(s2_expert_grouped_gu_grouped, (const unsigned long long* grp_ptr, const int32_t* grp_start,
                                             const int32_t* n_groups, const int32_t* ent_tok, const uint8_t* x_q8_0,
                                             const float* x_scales, float* gate_up, int cap_entries),
              gu_grouped_kernel(grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales, gate_up, cap_entries))
STRATA_KERNEL(s2_expert_grouped_down_grouped_t, (const unsigned long long* grp_ptr, const int32_t* grp_start,
                                                 const int32_t* n_groups, const int32_t* ent_dst,
                                                 const uint8_t* h_q8_0, const float* h_scales, float* out),
              down_grouped_t_kernel(grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, h_scales, out))
STRATA_KERNEL(s2_expert_grouped_down_grouped, (const unsigned long long* grp_ptr, const int32_t* grp_start,
                                               const int32_t* n_groups, const int32_t* ent_dst, const uint8_t* h_q8_0,
                                               const float* h_scales, float* out),
              down_grouped_kernel(grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, h_scales, out))
STRATA_KERNEL(s2_expert_grouped_activation_correction, (const uint8_t* q8, const float* scales, float* hx, int chunks),
              activation_correction_kernel(q8, scales, hx, chunks))
STRATA_KERNEL(s2_expert_grouped_cpu_order_gu, (const uint8_t* blob_base, const int32_t* slots,
                                               const int32_t* destinations, long long blob_bytes, const uint8_t* xq,
                                               const float* xs, const float* hx, float* out, int n_hits),
              cpu_order_projection_kernel<false>(blob_base, slots, destinations, blob_bytes, xq, xs, hx, out, n_hits))
STRATA_KERNEL(s2_expert_grouped_cpu_order_down, (const uint8_t* blob_base, const int32_t* slots,
                                                 const int32_t* destinations, long long blob_bytes, const uint8_t* xq,
                                                 const float* xs, const float* hx, float* out, int n_hits),
              cpu_order_projection_kernel<true>(blob_base, slots, destinations, blob_bytes, xq, xs, hx, out, n_hits))
STRATA_KERNEL(s2_expert_grouped_cpu_order_swiglu, (float* gu, int pairs), cpu_order_swiglu_kernel(gu, pairs))
STRATA_KERNEL(s2_expert_grouped_cpu_order_quantize, (const float* x, uint8_t* blocks, float* scales, float* hx,
                                                     int chunks),
              cpu_order_quantize_kernel(x, blocks, scales, hx, chunks))
