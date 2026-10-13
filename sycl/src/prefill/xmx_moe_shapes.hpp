// src/prefill/xmx_moe_shapes.hpp - the grouped XMX GEMM's tile shapes and per-launch group table, read by both builds.
#pragma once

#include "strata/prefill/xmx_moe.hpp"

#include <cstdint>

namespace strata::prefill::xmx {
// Sub-group width = the XMX tile's N: 8 on Alchemist (DG2, joint_matrix 8x8x16), 16 on Xe2 (Battlemage, 8x16x16; the
// CMake switch STRATA_XMX_SG16 for an AOT bmg / lnl build).  A build has one shape: the other is not valid there.
#ifdef STRATA_XMX_SG16
constexpr int kSgW = 16;
#else
constexpr int kSgW = 8;
#endif

struct Grp {
    const uint16_t* w[kMaxGroup];   // FP16 [N][K] per expert
    int off[kMaxGroup + 1];     // rows, relative to row0
    int tstart[kMaxGroup + 1];  // first M-tile of each expert
    int n;
};

template <int SGM_, int SGN_, int MT_, int NT_, int KS_>
struct Cfg {
    static constexpr int SGM = SGM_, SGN = SGN_, MT = MT_, NT = NT_, KS = KS_;
    static constexpr int SG = kSgW;
    static constexpr int BM = SGM * MT * 8, BN = SGN * NT * kSgW, NSG = SGM * SGN, WG = NSG * SG;
};

#ifdef STRATA_XMX_SG16
using Small = Cfg<2, 4, 4, 2, 32>;   // 64 x 128 tile: few rows per expert (8 sub-groups of 16)
using Big = Cfg<4, 2, 4, 4, 32>;     // 128 x 128 tile
#else
using Small = Cfg<2, 8, 4, 2, 32>;   // 64 x 128 tile: few rows per expert
using Big = Cfg<4, 4, 4, 4, 32>;     // 128 x 128 tile
#endif
}  // namespace strata::prefill::xmx
