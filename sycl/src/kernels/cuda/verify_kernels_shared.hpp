// verify_kernels_shared.hpp - the geometry and scratch layouts verify_kernels.dp.cpp and verify_kernels.host.cpp share.
#pragma once

#include "strata/kernels/verify_kernels.hpp"

#include <cstdint>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

constexpr int GS_COLS = 32;

constexpr int kArgMaxBlocks = 128, kArgThreads = 256, kTopBlocks = 8;

// A row's scratch, whatever the launch's row count (one scratch serves windows of every size): its counter, then
// the blocks' values and indices.
struct ArgRow {
    unsigned counter;
    unsigned pad[3];
    float v[kArgMaxBlocks];
    int i[kArgMaxBlocks];
};

// a row's top-probability scratch: its counter, then its 32 warps' sums
struct TopRow {
    unsigned counter;
    unsigned pad[3];
    float parts[32];
};

// one thread per window entry: up to kVerifyMaxT tokens x 10 routed experts (80), so 128 (#646 had 64: a window of
// 7+ tokens lost its last entries)
constexpr int kResidentPlanMax = 128;
static_assert(kVerifyMaxT * 10 <= kResidentPlanMax, "resident_plan: one thread per entry");
// the parallel scan keeps one partial sum per warp in s_wsum[4] and packs (entries << 16 | groups) in an int
static_assert(kResidentPlanMax <= 128 && kResidentPlanMax % 32 == 0, "resident_plan: at most 4 warps");
static_assert(kResidentPlanMax < 32768, "resident_plan: the packed entry count must stay below 2^15");

}  // namespace
}  // namespace strata::kernels
