// SYCL port: XMX (joint_matrix) GEMM straight from a ggml-quantized weight matrix - see iq_kernels.dp.cpp.
#pragma once
#include <cstdint>
#include "strata/kernels/iq_kernels.hpp"
namespace strata::kernels {
/// Y[M][n_out] (FP32, row stride ldy) = X[M][K] (FP16, row stride K) . W[n_out][K]^T with W's rows quantized as ggml
/// type `ty` (18 IQ3_XXS, 20 IQ4_NL, 21 IQ3_S, 22 IQ2_S, 42 Q2_0). `up` non-null: the gate/up pair, virtual rows
/// interleaved gate/up as iq_dequant_gu_f16 lays them out (n_out = 2 * n_ff). Returns false when the shape or type is
/// not handled (the caller keeps its dequant + oneMKL path).
bool xmx_gemm_iq(int ty, const void* gate, const void* up, int64_t K, int n_out, const uint16_t* X, int M, float* Y,
                 int64_t ldy, void* stream);

/// iq_xmx_grouped's launch shapes, read by the kernels (iq_kernels.dp.cpp) and the launcher (iq_kernels.host.cpp).
namespace xmxg {
// Sub-group width = the XMX tile N: 8 (DG2, 8x8x16) or 16 (Xe2, 8x16x16; CMake STRATA_XMX_SG16 for an AOT bmg / lnl build)
#ifdef STRATA_XMX_SG16
constexpr int kSgW = 16;
#else
constexpr int kSgW = 8;
#endif
/// The group's experts (a kernel argument by value): weight rows, first activation row and first tile of each.
struct Grp {
    const uint8_t* a[kXmxGroupMax];
    const uint8_t* b[kXmxGroupMax];
    int off[kXmxGroupMax + 1];
    int tstart[kXmxGroupMax + 1];
    int n;
};
/// A work-group's tiling: SGM x SGN sub-groups, MT x NT joint_matrix tiles each.
template <int SGM_, int SGN_, int MT_, int NT_>
struct Cfg {
    static constexpr int SGM = SGM_, SGN = SGN_, MT = MT_, NT = NT_, KS = 32, SG = kSgW;
    static constexpr int BM = SGM * MT * 8, BN = SGN * NT * kSgW, WG = SGM * SGN * kSgW;
};
#ifdef STRATA_XMX_SG16
using Small = Cfg<2, 4, 4, 2>;   // 64 x 128 (8 sub-groups of 16)
using Big = Cfg<4, 2, 4, 4>;     // 128 x 128
#else
using Small = Cfg<2, 8, 4, 2>;   // 64 x 128
using Big = Cfg<4, 4, 4, 4>;     // 128 x 128
#endif
}  // namespace xmxg

/// xmx_gemm_iq's launch shape, read by the kernel (iq_kernels.dp.cpp) and the launcher (iq_kernels.host.cpp).
namespace xmx {
constexpr int SG = 16;                 // XMX sub-group size
constexpr int NSG = 8;                 // sub-groups per work-group
constexpr int WG = SG * NSG;
constexpr int NT = 16;                 // features per work-group: one B tile
constexpr int MAXT = 4;                // C tiles per sub-group -> 512 rows per chunk
constexpr int APAD = 8;                // halves of padding on the staged A tile rows
/// The weight matrix (a kernel argument by value): gate rows, up rows (nullptr: not a gate/up pair), type, blocks per row.
struct Src { const uint8_t* gate; const uint8_t* up; int ty; int64_t per_row; };
}  // namespace xmx
}
