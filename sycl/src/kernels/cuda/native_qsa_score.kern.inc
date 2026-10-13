// native_qsa_score.kern.inc - entry points of native_qsa_score.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(native_qsa_score, (const float* pooled, const float* query, const float* bias, const int32_t* step,
                                 int max_cells, float* cells),
              score_kernel(pooled, query, bias, step, max_cells, cells))
