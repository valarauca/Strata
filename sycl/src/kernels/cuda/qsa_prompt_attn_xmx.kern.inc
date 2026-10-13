// qsa_prompt_attn_xmx.kern.inc - entry points of qsa_prompt_attn_xmx.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL_SG(16, qsa_prompt_attn_xmx_m0_c32,
                 (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps, int n_kv_heads, int page_size,
                  float scale_log2, float* attn, int cap),
                 prompt_attn_xmx2<0, 32>(q, p, ids, steps, n_kv_heads, page_size, scale_log2, attn, cap, strata_slm<uint8_t>()))
STRATA_KERNEL_SG(16, qsa_prompt_attn_xmx_m0_c64,
                 (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps, int n_kv_heads, int page_size,
                  float scale_log2, float* attn, int cap),
                 prompt_attn_xmx2<0, 64>(q, p, ids, steps, n_kv_heads, page_size, scale_log2, attn, cap, strata_slm<uint8_t>()))
STRATA_KERNEL_SG(16, qsa_prompt_attn_xmx_m1_c32,
                 (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps, int n_kv_heads, int page_size,
                  float scale_log2, float* attn, int cap),
                 prompt_attn_xmx2<1, 32>(q, p, ids, steps, n_kv_heads, page_size, scale_log2, attn, cap, strata_slm<uint8_t>()))
STRATA_KERNEL_SG(16, qsa_prompt_attn_xmx_m1_c64,
                 (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps, int n_kv_heads, int page_size,
                  float scale_log2, float* attn, int cap),
                 prompt_attn_xmx2<1, 64>(q, p, ids, steps, n_kv_heads, page_size, scale_log2, attn, cap, strata_slm<uint8_t>()))
STRATA_KERNEL_SG(16, qsa_prompt_attn_xmx_m3_c32,
                 (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps, int n_kv_heads, int page_size,
                  float scale_log2, float* attn, int cap),
                 prompt_attn_xmx2<3, 32>(q, p, ids, steps, n_kv_heads, page_size, scale_log2, attn, cap, strata_slm<uint8_t>()))
STRATA_KERNEL_SG(16, qsa_prompt_attn_xmx_m3_c64,
                 (const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps, int n_kv_heads, int page_size,
                  float scale_log2, float* attn, int cap),
                 prompt_attn_xmx2<3, 64>(q, p, ids, steps, n_kv_heads, page_size, scale_log2, attn, cap, strata_slm<uint8_t>()))
