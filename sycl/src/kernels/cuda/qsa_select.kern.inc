// qsa_select.kern.inc - entry points of qsa_select.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(qsa_select_block_scores_multi, (const float* pooled, const float* dead, const float* q_idx,
                                              const int32_t* steps, int nq, int64_t max_blocks, float* out,
                                              int early_exit),
              block_scores_multi_kernel(pooled, dead, q_idx, steps, nq, max_blocks, out, early_exit != 0))
STRATA_KERNEL(qsa_select_block_scores, (const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                                        int64_t max_blocks, float* out),
              block_scores_kernel(pooled, dead, q_idx, steps, max_blocks, out))
STRATA_KERNEL(qsa_select_block_scores_tc, (const float* pooled, const float* q_idx, const int32_t* steps, int64_t nq,
                                           int64_t max_blocks, int64_t reach, float* out),
              block_scores_tc_kernel(pooled, q_idx, steps, nq, max_blocks, reach, out, strata_slm<uint8_t>()))
STRATA_KERNEL(qsa_select_block_scores_tail, (const float* dead, const float* q_idx, const int32_t* steps,
                                             int64_t max_blocks, float* out),
              block_scores_tail_kernel(dead, q_idx, steps, max_blocks, out))
STRATA_KERNEL(qsa_select_block_topk, (const float* scores, const int32_t* steps, int64_t max_blocks, int64_t cap,
                                      int32_t* ids),
              block_topk_kernel(scores, steps, max_blocks, cap, ids))
STRATA_KERNEL(qsa_select_block_topk_wide, (const float* scores, const int32_t* steps, int64_t max_blocks, int64_t cap,
                                           int32_t* ids),
              block_topk_wide_kernel(scores, steps, max_blocks, cap, ids))
STRATA_KERNEL(qsa_select_block_topk_reg, (const float* scores, const int32_t* steps, int64_t max_blocks, int64_t cap,
                                          int32_t* ids),
              block_topk_reg_kernel<TK_PER>(scores, steps, max_blocks, cap, ids))
