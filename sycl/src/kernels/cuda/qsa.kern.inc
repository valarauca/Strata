// qsa.kern.inc - entry points of qsa.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(qsa_kv_append, (uint16_t* k_pool, uint16_t* v_pool, const int32_t* table, const int32_t* step,
                              const float* kcur, const float* vcur, int kv_heads, int head_dim, int page_size,
                              KvHostPools host),
              kv_append_kernel(k_pool, v_pool, table, step, kcur, vcur, kv_heads, head_dim, page_size, host))
STRATA_KERNEL(qsa_indexer_key_append, (const float* raw, const int32_t* pos_dev, int pos_base, const float* w_k_norm,
                                       float eps, float* tail, float* dead, float* pooled, int32_t* block_pos,
                                       int idx_dim, int r, int n_rot, const float* cos_tab, const float* sin_tab,
                                       const int32_t* mtab),
              indexer_key_append_kernel(raw, pos_dev, pos_base, w_k_norm, eps, tail, dead, pooled, block_pos, idx_dim, r,
                                        n_rot, cos_tab, sin_tab, mtab, strata_slm<uint8_t>()))
STRATA_KERNEL(qsa_index, (const float* pooled, const float* q_idx, const float* bias, int idx_n_head, int idx_dim,
                          long long r, const int32_t* step, float* cell_scores),
              qsa_index_kernel(pooled, q_idx, bias, idx_n_head, idx_dim, r, step, cell_scores))
STRATA_KERNEL(qsa_topk, (const float* scores, const int32_t* step, int* out_ids), topk_kernel(scores, step, out_ids))
STRATA_KERNEL(qsa_kv_gather, (const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* table, const int32_t* ids,
                              const int32_t* step, int kv_heads, int head_dim, int page_size, uint16_t* k_scratch,
                              uint16_t* v_scratch),
              kv_gather_kernel(k_pool, v_pool, table, ids, step, kv_heads, head_dim, page_size, k_scratch, v_scratch))
STRATA_KERNEL(qsa_attend, (const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch, const int32_t* step,
                           int n_head, int n_head_kv, int head_dim, float* attn, float* weights),
              qsa_attend_kernel(q, k_scratch, v_scratch, step, n_head, n_head_kv, head_dim, attn, weights,
                                strata_slm<uint8_t>()))
STRATA_KERNEL(qsa_gate_apply, (const float* attn, const float* q_full, int n_head, int head_dim, uint16_t* out),
              qsa_gate_apply_kernel(attn, q_full, n_head, head_dim, out))
STRATA_KERNEL(qsa_gate_apply_f32, (const float* attn, const float* q_full, int n_head, int head_dim, float* out),
              qsa_gate_apply_f32_kernel(attn, q_full, n_head, head_dim, out))
