// qsa_prompt_attn.kern.inc - entry points of qsa_prompt_attn.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(qsa_prompt_attn_m0, (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps,
                                    int n_kv_heads, int page_size, float scale_log2, float* attn, int cap),
              prompt_attn_kernel<0>(q, p, ids, steps, n_kv_heads, page_size, scale_log2, attn, cap, strata_slm<uint8_t>()))
STRATA_KERNEL(qsa_prompt_attn_m1, (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps,
                                    int n_kv_heads, int page_size, float scale_log2, float* attn, int cap),
              prompt_attn_kernel<1>(q, p, ids, steps, n_kv_heads, page_size, scale_log2, attn, cap, strata_slm<uint8_t>()))
STRATA_KERNEL(qsa_prompt_attn_m3, (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps,
                                    int n_kv_heads, int page_size, float scale_log2, float* attn, int cap),
              prompt_attn_kernel<3>(q, p, ids, steps, n_kv_heads, page_size, scale_log2, attn, cap, strata_slm<uint8_t>()))
STRATA_KERNEL(qsa_prompt_attn_m4, (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps,
                                    int n_kv_heads, int page_size, float scale_log2, float* attn, int cap),
              prompt_attn_kernel<4>(q, p, ids, steps, n_kv_heads, page_size, scale_log2, attn, cap, strata_slm<uint8_t>()))
STRATA_KERNEL(qsa_prompt_attn_i8, (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps,
                                   int n_kv_heads, int page_size, float scale_log2, float* attn, int cap),
              prompt_attn_i8_kernel(q, p, ids, steps, n_kv_heads, page_size, scale_log2, attn, cap, strata_slm<uint8_t>()))
