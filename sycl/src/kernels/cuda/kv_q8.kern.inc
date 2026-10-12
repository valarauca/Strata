// kv_q8.kern.inc - entry points of kv_q8.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(kv_q8_append, (int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* table,
                             const int32_t* step, const float* kcur, const float* vcur, int kv_heads, int head_dim,
                             int page_size, KvHostPools host),
              kv_append_q8_kernel(k_q, v_q, k_scale, v_scale, table, step, kcur, vcur, kv_heads, head_dim, page_size,
                                  host, 0, 0))
STRATA_KERNEL(kv_q8_append_multi, (int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                                   const int32_t* table, const int32_t* step, const float* kcur, const float* vcur,
                                   int kv_heads, int head_dim, int page_size, KvHostPools host, int step_stride,
                                   int cur_stride),
              kv_append_q8_kernel<true>(k_q, v_q, k_scale, v_scale, table, step, kcur, vcur, kv_heads, head_dim,
                                        page_size, host, step_stride, cur_stride))
STRATA_KERNEL(kv_q8_gather, (const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                             const int32_t* table, const int32_t* ids, const int32_t* step, int kv_heads, int head_dim,
                             int page_size, uint16_t* k_scratch, uint16_t* v_scratch),
              kv_gather_q8_kernel(k_q, v_q, k_scale, v_scale, table, ids, step, kv_heads, head_dim, page_size,
                                  k_scratch, v_scratch))
