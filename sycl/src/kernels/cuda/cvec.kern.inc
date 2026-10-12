// cvec.kern.inc - entry points of cvec.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(cvec_apply, (float* R, const float* dir, const float* s_l, const int* on, int mode, int64_t layer, int n,
                           int hc, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj, int64_t inj_ld,
                           int write),
              cvec_kernel(R, dir, s_l, on, mode, layer, n, hc, r_ld, bo, bo_ld, inj, inj_ld, write))
