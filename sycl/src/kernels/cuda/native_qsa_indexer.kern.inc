// native_qsa_indexer.kern.inc - entry points of native_qsa_indexer.dp.cpp (strata/ze/kernel_abi.hpp).
#define NQI_APPEND(NAME, TAB)                                                                                        \
    STRATA_KERNEL(native_qsa_indexer_append_##NAME,                                                                   \
                  (const float* raw, const int32_t* pos_dev, int pos_stride, int n_steps, int pos_base,              \
                   const float* gamma, float epsilon, float* tail, float* dead, float* pooled, int32_t* block_pos,   \
                   int max_cells, float theta_scale, float freq_scale, float corr_low, float corr_high,              \
                   float ext_factor, float mscale, const int32_t* mtab, RopeTab rt),                                 \
                  append<TAB>(raw, pos_dev, pos_stride, n_steps, pos_base, gamma, epsilon, tail, dead, pooled,       \
                              block_pos, max_cells, theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale, \
                              mtab, rt))
#define NQI_FIRST(NAME, TAB)                                                                                         \
    STRATA_KERNEL(native_qsa_indexer_first_##NAME,                                                                    \
                  (const float* raw, const float* gamma, float epsilon, float* dead, float* pooled,                  \
                   float theta_scale, float freq_scale, float corr_low, float corr_high, float ext_factor,           \
                   float mscale, const int32_t* mtab, RopeTab rt),                                                   \
                  append_first<TAB>(raw, gamma, epsilon, dead, pooled, theta_scale, freq_scale, corr_low, corr_high, \
                                    ext_factor, mscale, mtab, rt))
#define NQI_BLOCKS(NAME, TAB)                                                                                        \
    STRATA_KERNEL(native_qsa_indexer_blocks_##NAME,                                                                   \
                  (const float* raw, int64_t n, int64_t p0, int pos_base, const float* gamma, float epsilon,         \
                   const float* tail, const float* dead, float* pooled, int32_t* block_pos, int64_t first_block,     \
                   int64_t last_block, float theta_scale, float freq_scale, float corr_low, float corr_high,         \
                   float ext_factor, float mscale, const int32_t* mtab, RopeTab rt),                                 \
                  append_blocks<TAB>(raw, n, p0, pos_base, gamma, epsilon, tail, dead, pooled, block_pos,            \
                                     first_block, last_block, theta_scale, freq_scale, corr_low, corr_high,          \
                                     ext_factor, mscale, mtab, rt))
NQI_APPEND(tab, true)
NQI_APPEND(plain, false)
NQI_FIRST(tab, true)
NQI_FIRST(plain, false)
NQI_BLOCKS(tab, true)
NQI_BLOCKS(plain, false)
#undef NQI_APPEND
#undef NQI_FIRST
#undef NQI_BLOCKS
STRATA_KERNEL(native_qsa_indexer_tail, (const float* raw, int64_t n, int64_t p0, float* tail), append_tail(raw, n, p0, tail))
