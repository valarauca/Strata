// iq_kernels.kern.inc - entry points of iq_kernels.dp.cpp (strata/ze/kernel_abi.hpp).
//
// The templates are stamped once per instantiation the launchers in iq_kernels.host.cpp can request. On the host each
// stamped entry also specializes the selector its launcher reads (IqMmvq<TY>::k is k_iq_kernels_mmvq_<TY>), so a
// launcher that asks for an instantiation missing here does not compile.
#if defined(__SYCL_DEVICE_ONLY__)
#define IQK_SEL(name, ...)
#else
template <int TY> struct IqMmvq;
template <int TY, int NC, bool STAGE_GRID, bool EXACT_N> struct IqMmvqMulti;
template <int TG, int LN> struct IqGu;
template <int TG, bool STAGE_GRID> struct IqGuMulti;
template <int TD, int LN> struct IqDown;
template <int TD> struct IqDownMulti;
template <class C> struct IqXmxg;
#define IQK_SEL(name, ...) template <> struct __VA_ARGS__ { static constexpr const auto& k = k_##name; };
#endif

// ---------------------------------------------------------------- non-template kernels
STRATA_KERNEL(iq_kernels_quantize_q8_1, (const float* x, void* y, long long n), quantize_q8_1_kernel(x, (block_q8_1*) y, n))
STRATA_KERNEL(iq_kernels_swiglu, (const float* gate, const float* up, float* h, long long n),
              swiglu_entries_kernel(gate, up, h, n))
STRATA_KERNEL(iq_kernels_swiglu_q8_1, (const float* gate, const float* up, const int32_t* grp_start, const int32_t* n_groups,
                                       int n_ff, void* hq),
              swiglu_q8_1_entries_kernel(gate, up, grp_start, n_groups, n_ff, (block_q8_1*) hq))
STRATA_KERNEL(iq_kernels_dequant_f16, (int ty, const void* vx, uint16_t* y), dequant_flat_kernel<sycl::half>(ty, vx, (sycl::half*) y))
STRATA_KERNEL(iq_kernels_dequant_f32, (int ty, const void* vx, float* y), dequant_flat_kernel<float>(ty, vx, y))
STRATA_KERNEL(iq_kernels_embed_rows, (int ty, const uint8_t* table, size_t row_bytes, const int32_t* tokens, int64_t n_embd,
                                      float* y),
              embed_rows_kernel(ty, table, row_bytes, tokens, n_embd, y))
STRATA_KERNEL(iq_kernels_dequant_gu, (int ty, const void* gate, const void* up, int64_t per_row, uint16_t* y),
              dequant_gu_kernel(ty, gate, up, per_row, (sycl::half*) y))
STRATA_KERNEL(iq_kernels_dequant_expert, (int gt, int dt, const void* gate, const void* up, const void* down, int64_t per_row,
                                          int64_t gu_wgs, uint16_t* ygu, uint16_t* yd),
              dequant_expert_kernel(gt, dt, gate, up, down, per_row, gu_wgs, (sycl::half*) ygu, (sycl::half*) yd))
STRATA_KERNEL_SG(16, iq_kernels_xmx_gemm, (xmx::Src src, const uint16_t* X, int K, int slab_k, float* Y, int64_t ldy, int M,
                                           int n_out, size_t stage_halves),
                 xmx::gemm_slm(src, (const sycl::half*) X, K, slab_k, Y, ldy, M, n_out, stage_halves))

// ---------------------------------------------------------------- iq_mmvq (launch_mmvq, launch_mmvq_multi_nc)
#define IQK_MMVQ(T)                                                                                                    \
    STRATA_KERNEL(iq_kernels_mmvq_##T, (const uint8_t* w, size_t rb, const void* x, float* y, int n_in, int n_out,      \
                                        int ncols),                                                                    \
                  mmvq_kernel<T>(w, rb, (const block_q8_1*) x, y, n_in, n_out, ncols))                                 \
    IQK_SEL(iq_kernels_mmvq_##T, IqMmvq<T>)
#define IQK_MMVQM(T, NC, SG, EX)                                                                                       \
    STRATA_KERNEL(iq_kernels_mmvqm_##T##_##NC##_##SG##_##EX, (const uint8_t* w, size_t rb, const void* x, float* y,    \
                                                              int n_in, int n_out, int ncols),                         \
                  mmvq_multi_kernel<T, NC, SG, EX>(w, rb, (const block_q8_1*) x, y, n_in, n_out, ncols))               \
    IQK_SEL(iq_kernels_mmvqm_##T##_##NC##_##SG##_##EX, IqMmvqMulti<T, NC, SG, EX>)
// launch_mmvq_multi_nc's switch: 1..6 columns exactly, then 1 or 8 at a time
#define IQK_MMVQM_NC(T, SG)                                                                                            \
    IQK_MMVQM(T, 1, SG, true) IQK_MMVQM(T, 2, SG, true) IQK_MMVQM(T, 3, SG, true) IQK_MMVQM(T, 4, SG, true)            \
    IQK_MMVQM(T, 5, SG, true) IQK_MMVQM(T, 6, SG, true) IQK_MMVQM(T, 1, SG, false) IQK_MMVQM(T, 8, SG, false)

// every STRATA_MMVQ_FMTS type (the Split ones with STRATA_OLD_IQ_MMVQ=1)
IQK_MMVQ(16) IQK_MMVQ(17) IQK_MMVQ(18) IQK_MMVQ(20) IQK_MMVQ(21) IQK_MMVQ(22) IQK_MMVQ(23) IQK_MMVQ(29) IQK_MMVQ(42)
IQK_MMVQ(12) IQK_MMVQ(13) IQK_MMVQ(7) IQK_MMVQ(6) IQK_MMVQ(2) IQK_MMVQ(3) IQK_MMVQ(8)
// the Split types, unstaged
IQK_MMVQM_NC(16, false) IQK_MMVQM_NC(17, false) IQK_MMVQM_NC(18, false) IQK_MMVQM_NC(20, false) IQK_MMVQM_NC(21, false)
IQK_MMVQM_NC(22, false) IQK_MMVQM_NC(23, false) IQK_MMVQM_NC(29, false) IQK_MMVQM_NC(42, false)
// the kStageIqGrid types, staged (STRATA_IQ_STAGE_GRID_MMVQ=1)
IQK_MMVQM_NC(16, true) IQK_MMVQM_NC(17, true) IQK_MMVQM_NC(22, true) IQK_MMVQM_NC(29, true)

// ---------------------------------------------------------------- native_expert_grouped (launch_gu, launch_down)
#define IQK_GU(T, LN)                                                                                                  \
    STRATA_KERNEL(iq_kernels_gu_##T##_##LN, (const unsigned long long* grp_ptr, const int32_t* grp_start,              \
                                             const int32_t* n_groups, const int32_t* ent_tok, const void* xq,          \
                                             NativeExpertLayout L, float* gate, float* up),                            \
                  native_gu_kernel<T, LN>(grp_ptr, grp_start, n_groups, ent_tok, (const block_q8_1*) xq, L, gate, up))   \
    IQK_SEL(iq_kernels_gu_##T##_##LN, IqGu<T, LN>)
// launch_gu_lanes's switch: 4, 8, 16 or 32 lanes a row
#define IQK_GU_LANES(T) IQK_GU(T, 4) IQK_GU(T, 8) IQK_GU(T, 16) IQK_GU(T, 32)
#define IQK_GUM(T, SG)                                                                                                 \
    STRATA_KERNEL(iq_kernels_gum_##T##_##SG, (const unsigned long long* grp_ptr, const int32_t* grp_start,             \
                                              const int32_t* n_groups, const int32_t* ent_tok, const void* xq,         \
                                              NativeExpertLayout L, float* gate, float* up),                           \
                  native_gu_multi_kernel<T, SG, false>(grp_ptr, grp_start, n_groups, ent_tok, (const block_q8_1*) xq, L, \
                                                       gate, up))                                                      \
    IQK_SEL(iq_kernels_gum_##T##_##SG, IqGuMulti<T, SG>)
#define IQK_DOWN(T, LN)                                                                                                \
    STRATA_KERNEL(iq_kernels_down_##T##_##LN, (const unsigned long long* grp_ptr, const int32_t* grp_start,            \
                                               const int32_t* n_groups, const int32_t* ent_dst, const void* hq,        \
                                               NativeExpertLayout L, float* out),                                      \
                  native_down_kernel<T, LN>(grp_ptr, grp_start, n_groups, ent_dst, (const block_q8_1*) hq, L, out))     \
    IQK_SEL(iq_kernels_down_##T##_##LN, IqDown<T, LN>)
// launch_down_lanes's switch: 4, 8, 16 or 32 lanes a row
#define IQK_DOWN_LANES(T) IQK_DOWN(T, 4) IQK_DOWN(T, 8) IQK_DOWN(T, 16) IQK_DOWN(T, 32)
#define IQK_DOWNM(T)                                                                                                   \
    STRATA_KERNEL(iq_kernels_downm_##T, (const unsigned long long* grp_ptr, const int32_t* grp_start,                  \
                                         const int32_t* n_groups, const int32_t* ent_dst, const void* hq,              \
                                         NativeExpertLayout L, float* out),                                            \
                  native_down_multi_kernel<T, false>(grp_ptr, grp_start, n_groups, ent_dst, (const block_q8_1*) hq, L, out)) \
    IQK_SEL(iq_kernels_downm_##T, IqDownMulti<T>)

// gate/up: every STRATA_GU_FMTS type on the lane kernels
IQK_GU_LANES(16) IQK_GU_LANES(17) IQK_GU_LANES(18) IQK_GU_LANES(21) IQK_GU_LANES(22) IQK_GU_LANES(23) IQK_GU_LANES(29)
IQK_GU_LANES(42) IQK_GU_LANES(12) IQK_GU_LANES(13) IQK_GU_LANES(6) IQK_GU_LANES(2) IQK_GU_LANES(3) IQK_GU_LANES(8)
#ifdef STRATA_Q6K_EXPERTS
IQK_GU_LANES(14)
#endif
// gate/up, STRATA_EXPERT_SPLIT=1: the Split types unstaged, the kStageIqGrid ones also staged
IQK_GUM(16, false) IQK_GUM(17, false) IQK_GUM(18, false) IQK_GUM(21, false) IQK_GUM(22, false) IQK_GUM(23, false)
IQK_GUM(29, false) IQK_GUM(42, false)
IQK_GUM(16, true) IQK_GUM(17, true) IQK_GUM(22, true) IQK_GUM(29, true)
// down: every STRATA_D_FMTS type on the lane kernels, and the Split ones with STRATA_EXPERT_SPLIT=1
IQK_DOWN_LANES(20) IQK_DOWN_LANES(23) IQK_DOWN_LANES(42) IQK_DOWN_LANES(7) IQK_DOWN_LANES(6) IQK_DOWN_LANES(2)
IQK_DOWN_LANES(3) IQK_DOWN_LANES(8)
IQK_DOWNM(20) IQK_DOWNM(23) IQK_DOWNM(42)

// ---------------------------------------------------------------- iq_xmx_grouped (xmxg::launch)
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

#undef IQK_DOWNM
#undef IQK_DOWN_LANES
#undef IQK_DOWN
#undef IQK_GUM
#undef IQK_GU_LANES
#undef IQK_GU
#undef IQK_MMVQM_NC
#undef IQK_MMVQM
#undef IQK_MMVQ
#undef IQK_SEL
