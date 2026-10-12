// fused_gr_shared.hpp - the geometry and the multi-token argument block fused_gr.dp.cpp and fused_gr.host.cpp share.
#pragma once

#include "strata/kernels/fused_gr.hpp"

#include <cstdint>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
#if defined(__HIPCC__)
constexpr int TILE = 1280;             // eight-token tile fits gfx1100's 64 KiB LDS limit
#else
constexpr int TILE = 2560;             // xn floats per token staged at a time: 320 chunks of 8, 10 per lane
#endif
constexpr int TQ = TILE / 8 / 32;      // uint4 weight chunks per lane per tile
// SYCL port: the down kernel's 41 work-groups filled 15% of the B70's threads (metrics 2026-09-30); each row's
// D = 4 tiles are split over DOWN_SPLIT work-groups that write partial sums, summed in a fixed order by the up
// kernel (deterministic, unlike atomics) before the row's nonlinearity.
#ifndef STRATA_GR_DOWN_SPLIT
#define STRATA_GR_DOWN_SPLIT (D / TILE)   // 4: one tile per split (1: the unsplit order, for A/B)
#endif
constexpr int DOWN_SPLIT = STRATA_GR_DOWN_SPLIT;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80

constexpr int GRS_COLS = 128, GRS_SLICES = D / GRS_COLS, GRS_ROWS = LR + HC;   // 80 slices, 324 rows

constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160

constexpr int PR = LR + HC;                        // partial rows per (token, stream): 320 down + 4 inject

constexpr int kHcPlain = 1, kHcSplit = 2, kHcStaged = 3;
constexpr int H_TILE = 1280;                          // staged tile: half a stream = 160 chunks of 8, 5 per lane
constexpr int HQ = H_TILE / 8 / 32;
constexpr int N_HTILES = D / H_TILE;                  // 8
static_assert(N % H_TILE == 0, "a staged tile never straddles two streams");
static_assert(H_TILE % 256 == 0, "a staged tile holds whole rounds of 32 chunks: the plain read's lane order");

struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
    float* part;   ///< [DOWN_SPLIT][kFusedGrMaxT][LR] partial row sums of the down kernel
    int nsplit = DOWN_SPLIT;   ///< SYCL port: splits the up kernel sums (1 after the sliced down kernel's reduce)
    float* part2 = nullptr;    ///< SYCL port: [GRS_SLICES][kFusedGrMaxT][LR + HC] slice partials of the sliced kernel
};

}  // namespace
}  // namespace strata::kernels
