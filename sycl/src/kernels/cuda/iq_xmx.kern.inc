// iq_xmx.kern.inc - the grouped XMX entries of iq_kernels.dp.cpp (xmxg::launch), a bundle of their own (iq_xmx.dp.cpp)
// built with the large register file, as main launched them with grf_size<256>. The host reads them through
// iq_kernels.kern.inc.
#if defined(__SYCL_DEVICE_ONLY__)
#define IQK_SEL(name, ...)
#endif
#ifndef STRATA_NO_DG2_XMX
#define IQK_XG(name, C)                                                                                                \
    STRATA_KERNEL_SG(xmxg::kSgW, iq_kernels_##name, (int ty, int gu, const uint16_t* X, float* Y, xmxg::Grp gp, int N,  \
                                                     int K),                                                           \
                     xmxg::grouped_kernel<C>(ty, gu != 0, (const sycl::half*) X, Y, gp, N, K))                         \
    IQK_SEL(iq_kernels_##name, IqXmxg<C>)
IQK_XG(xg_small, xmxg::Small)
IQK_XG(xg_big, xmxg::Big)
#undef IQK_XG
#endif
#if defined(__SYCL_DEVICE_ONLY__)
#undef IQK_SEL
#endif
