// rope.kern.inc - entry points of rope.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(rope_neox, (const float* x, float* out, long long rows, int head_dim, int n_rot, const float* cos_tab,
                          const float* sin_tab, const int* pos, const int32_t* mtab),
              rope_neox_kernel(x, out, rows, head_dim, n_rot, cos_tab, sin_tab, pos, mtab))
