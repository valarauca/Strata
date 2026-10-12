// kv_q4.kern.inc - entry points of kv_q4.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(kv_q4_fwht256, (const float* src, float* dst, int64_t n_rows, float scale),
              fwht256_kernel(src, dst, n_rows, scale))
STRATA_KERNEL(kv_q4_append, (uint8_t* k_q4, uint8_t* v_q4, const int32_t* table, const int32_t* step,
                             int step_stride, int planes, const float* kcur, const float* vcur, int kv_heads,
                             int head_dim, int page_size, KvHostPools host),
              kv_append_q4_kernel(k_q4, v_q4, table, step, step_stride, planes, kcur, vcur, kv_heads, head_dim,
                                  page_size, host))
STRATA_KERNEL(kv_q4_append_batch, (uint8_t* k_q4, uint8_t* v_q4, const int32_t* table, int64_t pos0, const float* K,
                                   const float* V, int kv_heads, int head_dim, int page_size, int is_v_grid,
                                   KvHostPools host, KvHostPools stage),
              kv_append_q4_batch_kernel(k_q4, v_q4, table, pos0, K, V, kv_heads, head_dim, page_size, is_v_grid, host,
                                        stage))
STRATA_KERNEL(kv_q4_gather, (const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* table, const int32_t* ids,
                             const int32_t* step, int kv_heads, int head_dim, int page_size, uint16_t* k_scratch,
                             uint16_t* v_scratch),
              kv_gather_q4_kernel(k_q4, v_q4, table, ids, step, kv_heads, head_dim, page_size, k_scratch, v_scratch))
