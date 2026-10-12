// xmx_moe.kern.inc - entry points of xmx_moe.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL_SG(16, xmx_moe_small, (const uint16_t* X, float* Y, Grp gp, int N, int K),
                 grouped_kernel<Small>((const sycl::half*) X, Y, gp, N, K))
STRATA_KERNEL_SG(16, xmx_moe_big, (const uint16_t* X, float* Y, Grp gp, int N, int K),
                 grouped_kernel<Big>((const sycl::half*) X, Y, gp, N, K))
