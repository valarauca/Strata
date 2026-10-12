// native_mmvq.kern.inc - entry points of native_mmvq.dp.cpp (strata/ze/kernel_abi.hpp).
#define NM_P (const void* w, const void* x, float* y, int n_in, int n_out)
#define NM_X (const Q81Block*) x
STRATA_KERNEL(native_mmvq_quantize_q8_1, (const float* x, void* y, int n_in),
              native_quantize_q8_1_kernel(x, (Q81Block*) y, n_in))
STRATA_KERNEL(native_mmvq_swiglu_quantize_q8_1, (const float* gate, const float* up, void* y, int n_in),
              native_swiglu_quantize_q8_1_kernel(gate, up, (Q81Block*) y, n_in))
#define NM_SINGLE(FMT, KERNEL, BLOCK)                                                                               \
    STRATA_KERNEL(native_mmvq_single_##FMT##_small, NM_P, KERNEL<true>((const BLOCK*) w, NM_X, y, n_in, n_out))     \
    STRATA_KERNEL(native_mmvq_single_##FMT, NM_P, KERNEL<false>((const BLOCK*) w, NM_X, y, n_in, n_out))
NM_SINGLE(q5k, native_q5_k_mmvq_kernel, Q5KBlock)
NM_SINGLE(q20, native_q2_0_mmvq_kernel, Q20Block)
NM_SINGLE(q3k, native_q3_k_mmvq_kernel, Q3KBlock)
NM_SINGLE(q4k, native_q4_k_mmvq_kernel, Q4KBlock)
NM_SINGLE(q6k, native_q6_k_mmvq_kernel, Q6KBlock)
STRATA_KERNEL(native_mmvq_single_iq4xs_small, NM_P,
              native_iq4_xs_mmvq_kernel<true>((const IQ4XSBlock*) w, NM_X, y, n_in, n_out,
                                              const_cast<int8_t*>(kIq4nlTable)))
STRATA_KERNEL(native_mmvq_single_iq4xs, NM_P,
              native_iq4_xs_mmvq_kernel<false>((const IQ4XSBlock*) w, NM_X, y, n_in, n_out,
                                               const_cast<int8_t*>(kIq4nlTable)))
#define NM_SMALL(FMT, BLOCK, QI_)                                                                                   \
    STRATA_KERNEL(native_mmvq_single_##FMT##_small, NM_P,                                                           \
                  native_small_mmvq_kernel<BLOCK, QI_, true>((const BLOCK*) w, NM_X, y, n_in, n_out,               \
                                                             const_cast<int8_t*>(kIq4nlTable)))                     \
    STRATA_KERNEL(native_mmvq_single_##FMT, NM_P,                                                                   \
                  native_small_mmvq_kernel<BLOCK, QI_, false>((const BLOCK*) w, NM_X, y, n_in, n_out,              \
                                                              const_cast<int8_t*>(kIq4nlTable)))
NM_SMALL(q40, Q40Block, 4)
NM_SMALL(q50, Q50Block, 4)
NM_SMALL(q80, Q80Block, 8)
NM_SMALL(iq4nl, IQ4NLBlock, 4)
#define NM_MULTI_NC(FMT, NC, ...)                                                                                   \
    STRATA_KERNEL(native_mmvq_multi_##FMT##_up_##NC, NM_P,                                                          \
                  native_mmvq_multi_kernel<__VA_ARGS__, NC, (NC <= 4 ? 4 : 2), 1>(                                  \
                      (const __VA_ARGS__::Block*) w, NM_X, y, n_in, n_out))                                         \
    STRATA_KERNEL_SG(16, native_mmvq_multi_##FMT##_sg16s_##NC, NM_P,                                                \
                     native_mmvq_multi_kernel<__VA_ARGS__, NC, WARPS * WARP / 16, WARPS, false, false, 16>(         \
                         (const __VA_ARGS__::Block*) w, NM_X, y, n_in, n_out))                                      \
    STRATA_KERNEL_SG(16, native_mmvq_multi_##FMT##_sg16_##NC, NM_P,                                                 \
                     native_mmvq_multi_kernel<__VA_ARGS__, NC, WARPS * WARP / 16, 1, false, false, 16>(             \
                         (const __VA_ARGS__::Block*) w, NM_X, y, n_in, n_out))                                      \
    STRATA_KERNEL(native_mmvq_multi_##FMT##_r1ts_##NC, NM_P,                                                        \
                  native_mmvq_multi_kernel<__VA_ARGS__, NC, WARPS, 1, true>(                                        \
                      (const __VA_ARGS__::Block*) w, NM_X, y, n_in, n_out))                                         \
    STRATA_KERNEL(native_mmvq_multi_##FMT##_r1_##NC, NM_P,                                                          \
                  native_mmvq_multi_kernel<__VA_ARGS__, NC, WARPS, 1>(                                              \
                      (const __VA_ARGS__::Block*) w, NM_X, y, n_in, n_out))                                         \
    STRATA_KERNEL(native_mmvq_multi_##FMT##_r2ts_##NC, NM_P,                                                        \
                  native_mmvq_multi_kernel<__VA_ARGS__, NC, WARPS, 2, true>(                                        \
                      (const __VA_ARGS__::Block*) w, NM_X, y, n_in, n_out))                                         \
    STRATA_KERNEL(native_mmvq_multi_##FMT##_r2_##NC, NM_P,                                                          \
                  native_mmvq_multi_kernel<__VA_ARGS__, NC, WARPS, 2>(                                              \
                      (const __VA_ARGS__::Block*) w, NM_X, y, n_in, n_out))
#define NM_MULTI(FMT, ...)                                                                                          \
    NM_MULTI_NC(FMT, 2, __VA_ARGS__) NM_MULTI_NC(FMT, 3, __VA_ARGS__) NM_MULTI_NC(FMT, 4, __VA_ARGS__)              \
    NM_MULTI_NC(FMT, 5, __VA_ARGS__) NM_MULTI_NC(FMT, 6, __VA_ARGS__) NM_MULTI_NC(FMT, 7, __VA_ARGS__)              \
    NM_MULTI_NC(FMT, 8, __VA_ARGS__)
NM_MULTI(q5k, Q5KTraits)
NM_MULTI(q4k, Q4KTraits)
NM_MULTI(q20, Q20Traits)
NM_MULTI(q3k, Q3KTraits)
NM_MULTI(q6k, Q6KTraits)
NM_MULTI(iq4xs, IQ4XSTraits)
NM_MULTI(q40, SmallTraits<Q40Block, 4>)
NM_MULTI(q50, SmallTraits<Q50Block, 4>)
NM_MULTI(q80, SmallTraits<Q80Block, 8>)
NM_MULTI(iq4nl, SmallTraits<IQ4NLBlock, 4>)
#define NM_NC(NC)                                                                                                   \
    STRATA_KERNEL(native_mmvq_q6k_pf2_##NC, NM_P,                                                                   \
                  native_mmvq_q6k_pf_kernel<NC, 2>((const Q6KBlock*) w, NM_X, y, n_in, n_out))                      \
    STRATA_KERNEL(native_mmvq_q6k_pf3_##NC, NM_P,                                                                   \
                  native_mmvq_q6k_pf_kernel<NC, 3>((const Q6KBlock*) w, NM_X, y, n_in, n_out))                      \
    STRATA_KERNEL(native_mmvq_q6k_pf4_##NC, NM_P,                                                                   \
                  native_mmvq_q6k_pf_kernel<NC, 4>((const Q6KBlock*) w, NM_X, y, n_in, n_out))                      \
    STRATA_KERNEL(native_mmvq_q6k_pf6_##NC, NM_P,                                                                   \
                  native_mmvq_q6k_pf_kernel<NC, 6>((const Q6KBlock*) w, NM_X, y, n_in, n_out))                      \
    STRATA_KERNEL(native_mmvq_q6k_a2_##NC, NM_P,                                                                    \
                  native_mmvq_q6k_wide_kernel<NC, 1, WARP, (int) sizeof(Q6KBlock), true>(                           \
                      (const Q6KBlock*) w, NM_X, y, n_in, n_out))                                                   \
    STRATA_KERNEL_SG(16, native_mmvq_q6k_w16_##NC, NM_P,                                                            \
                     native_mmvq_q6k_wide_kernel<NC, 1, 16>((const Q6KBlock*) w, NM_X, y, n_in, n_out))             \
    STRATA_KERNEL(native_mmvq_q6k_w1_##NC, NM_P,                                                                    \
                  native_mmvq_q6k_wide_kernel<NC, 1>((const Q6KBlock*) w, NM_X, y, n_in, n_out))                    \
    STRATA_KERNEL(native_mmvq_q6k_w2_##NC, NM_P,                                                                    \
                  native_mmvq_q6k_wide_kernel<NC, 2>((const Q6KBlock*) w, NM_X, y, n_in, n_out))                    \
    STRATA_KERNEL(native_mmvq_q6k_w4_##NC, NM_P,                                                                    \
                  native_mmvq_q6k_wide_kernel<NC, 4>((const Q6KBlock*) w, NM_X, y, n_in, n_out))                    \
    STRATA_KERNEL(native_mmvq_wide_q4k_##NC, NM_P,                                                                  \
                  native_mmvq_wide_kernel<WideQ4K, NC>((const Q4KBlock*) w, NM_X, y, n_in, n_out))                  \
    STRATA_KERNEL(native_mmvq_wide_q5k_##NC, NM_P,                                                                  \
                  native_mmvq_wide_kernel<WideQ5K, NC>((const Q5KBlock*) w, NM_X, y, n_in, n_out))                  \
    STRATA_KERNEL(native_mmvq_wide_iq4xs_##NC, NM_P,                                                                \
                  native_mmvq_wide_kernel<WideIQ4XS, NC>((const IQ4XSBlock*) w, NM_X, y, n_in, n_out))              \
    STRATA_KERNEL(native_mmvq_wide32_q80_##NC, NM_P,                                                                \
                  native_mmvq_wide32_kernel<Wide32Q8, NC>((const Q80Block*) w, NM_X, y, n_in, n_out))               \
    STRATA_KERNEL(native_mmvq_wide32_iq4nl_##NC, NM_P,                                                              \
                  native_mmvq_wide32_kernel<Wide32IQ4NL, NC>((const IQ4NLBlock*) w, NM_X, y, n_in, n_out))          \
    STRATA_KERNEL(native_mmvq_q8p_##NC, (const int8_t* qs, const uint16_t* d, const void* x, float* y, int n_in,   \
                                         int n_out, int row_step),                                                  \
                  native_q8_0_packed_kernel<NC>(qs, (const sycl::half*) d, NM_X, y, n_in, n_out, row_step))         \
    STRATA_KERNEL(native_mmvq_q8p_ts_##NC, (const int8_t* qs, const uint16_t* d, const void* x, float* y,          \
                                            int n_in, int n_out, int row_step),                                     \
                  native_q8_0_packed_kernel<NC, true>(qs, (const sycl::half*) d, NM_X, y, n_in, n_out, row_step))   \
    STRATA_KERNEL(native_mmvq_q6p_##NC, (const int* ql, const int* qh, const int8_t* sc, const uint16_t* d,        \
                                         const void* x, float* y, int n_in, int n_out, int row_step),               \
                  native_q6_k_packed_kernel<NC>(ql, qh, sc, (const sycl::half*) d, NM_X, y, n_in, n_out, row_step)) \
    STRATA_KERNEL(native_mmvq_q6p_ts_##NC, (const int* ql, const int* qh, const int8_t* sc, const uint16_t* d,     \
                                            const void* x, float* y, int n_in, int n_out, int row_step),            \
                  native_q6_k_packed_kernel<NC, true>(ql, qh, sc, (const sycl::half*) d, NM_X, y, n_in, n_out,      \
                                                      row_step))
NM_NC(1)
NM_NC(2)
NM_NC(3)
NM_NC(4)
NM_NC(5)
NM_NC(6)
NM_NC(7)
NM_NC(8)
#define NM_W224(NC)                                                                                                 \
    STRATA_KERNEL(native_mmvq_q6k_w224_##NC, NM_P,                                                                  \
                  native_mmvq_q6k_wide_kernel<NC, 1, WARP, 224>((const Q6KBlock*) w, NM_X, y, n_in, n_out))
NM_W224(1)
NM_W224(2)
NM_W224(3)
NM_W224(4)
NM_W224(6)
#define NM_ROWWARP(NC)                                                                                              \
    STRATA_KERNEL(native_mmvq_rowwarp_q6k_rp1_##NC, NM_P,                                                           \
                  native_mmvq_rowwarp_kernel<Q6KTraits, NC, 1>((const Q6KBlock*) w, NM_X, y, n_in, n_out))          \
    STRATA_KERNEL(native_mmvq_rowwarp_q6k_rp2_##NC, NM_P,                                                           \
                  native_mmvq_rowwarp_kernel<Q6KTraits, NC, 2>((const Q6KBlock*) w, NM_X, y, n_in, n_out))          \
    STRATA_KERNEL(native_mmvq_rowwarp_q6k_rp4_##NC, NM_P,                                                           \
                  native_mmvq_rowwarp_kernel<Q6KTraits, NC, 4>((const Q6KBlock*) w, NM_X, y, n_in, n_out))
NM_ROWWARP(1)
NM_ROWWARP(2)
NM_ROWWARP(3)
NM_ROWWARP(4)
#define NM_PAIR(NC)                                                                                                 \
    STRATA_KERNEL(native_mmvq_pair_q80_##NC, (const void* w1, const void* w2, const void* x, float* y1, float* y2, \
                                              int n_in, int n_out),                                                 \
                  native_mmvq_multi_kernel<SmallTraits<Q80Block, 8>, NC, WARPS, 1, false, true>(                    \
                      (const Q80Block*) w1, NM_X, y1, n_in, n_out, (const Q80Block*) w2, y2))                       \
    STRATA_KERNEL(native_mmvq_pair_q80_ts_##NC, (const void* w1, const void* w2, const void* x, float* y1,         \
                                                 float* y2, int n_in, int n_out),                                   \
                  native_mmvq_multi_kernel<SmallTraits<Q80Block, 8>, NC, WARPS, 1, true, true>(                     \
                      (const Q80Block*) w1, NM_X, y1, n_in, n_out, (const Q80Block*) w2, y2))
NM_PAIR(2)
NM_PAIR(3)
NM_PAIR(4)
NM_PAIR(5)
NM_PAIR(6)
NM_PAIR(7)
NM_PAIR(8)
#undef NM_PAIR
#undef NM_ROWWARP
#undef NM_W224
#undef NM_NC
#undef NM_MULTI
#undef NM_MULTI_NC
#undef NM_SMALL
#undef NM_SINGLE
#undef NM_X
#undef NM_P
