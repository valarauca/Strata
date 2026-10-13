// qsa_decode_attn.kern.inc - entry points of qsa_decode_attn.dp.cpp (strata/ze/kernel_abi.hpp).
#define QDA_CHUNK(NAME, M, LC)                                                                                     \
    STRATA_KERNEL(qsa_decode_attn_##NAME, (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* step, \
                                           int n_kv_heads, int page_size, float scale, float* part_acc,             \
                                           float* part_m, float* part_l, int n_chunks, int cap,                     \
                                           long long scratch_stride),                                               \
                  attn_chunk_kernel<M, LC>(q, p, ids, step, n_kv_heads, page_size, scale, part_acc, part_m, part_l,  \
                                           n_chunks, cap, scratch_stride))
QDA_CHUNK(chunk_0, 0, false)
QDA_CHUNK(chunk_1, 1, false)
QDA_CHUNK(chunk_2, 2, false)
QDA_CHUNK(chunk_3, 3, false)
QDA_CHUNK(chunk_lc_0, 0, true)
QDA_CHUNK(chunk_lc_1, 1, true)
QDA_CHUNK(chunk_lc_2, 2, true)
QDA_CHUNK(chunk_lc_3, 3, true)
#undef QDA_CHUNK
STRATA_KERNEL(qsa_decode_attn_merge, (const float* part_acc, const float* part_m, const float* part_l, int n_chunks,
                                      float* attn, long long scratch_stride),
              attn_merge_kernel(part_acc, part_m, part_l, n_chunks, attn, scratch_stride))
