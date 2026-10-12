// src/kernels/cuda/native_mmvq.host.cpp - the host side of native_mmvq.dp.cpp: kernel choice, validation, packed copies and launchers.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <unordered_map>
#include <cstdio>
#include <vector>
#include <cmath>

namespace strata::kernels {

#include "native_mmvq.kern.inc"

namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;
constexpr int MAX_NCOLS = 8;

// The blocks the host reads or writes, fp16 fields as their bits (the device's are in native_mmvq.dp.cpp).
struct alignas(4) Q81Block {
    uint16_t ds[2];
    int8_t qs[32];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};
struct Q80Block {
    uint16_t d;
    int8_t qs[32];
};
struct Q40Block;
struct Q50Block;
struct IQ4NLBlock;
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);

using MmvqKernel = decltype(k_native_mmvq_single_q5k);
#define NMH_ROW4(P) {&k_##P##1, &k_##P##2, &k_##P##3, &k_##P##4}
#define NMH_ROW7(P) {&k_##P##2, &k_##P##3, &k_##P##4, &k_##P##5, &k_##P##6, &k_##P##7, &k_##P##8}
#define NMH_ROW8(P) {&k_##P##1, &k_##P##2, &k_##P##3, &k_##P##4, &k_##P##5, &k_##P##6, &k_##P##7, &k_##P##8}

// native_mmvq_multi_kernel's launch layouts (launch_multi_n), each indexed by ncols - 2.
enum MultiLayout { M_UP, M_SG16_SMALL, M_SG16, M_ROWS1_TS, M_ROWS1, M_ROWS2_TS, M_ROWS2, M_LAYOUTS };
#define NMH_MULTI(FMT)                                                                                              \
    {NMH_ROW7(native_mmvq_multi_##FMT##_up_), NMH_ROW7(native_mmvq_multi_##FMT##_sg16s_),                         \
     NMH_ROW7(native_mmvq_multi_##FMT##_sg16_), NMH_ROW7(native_mmvq_multi_##FMT##_r1ts_),                        \
     NMH_ROW7(native_mmvq_multi_##FMT##_r1_), NMH_ROW7(native_mmvq_multi_##FMT##_r2ts_),                          \
     NMH_ROW7(native_mmvq_multi_##FMT##_r2_)}

// The host half of each format's traits: the iteration constants the launchers test and the format's entry points.
struct Q5KTraits {
    static constexpr int DIV = QK, BPI = VDR * WARPS * WARP / QI;
    static constexpr const MmvqKernel* multi[M_LAYOUTS][7] = NMH_MULTI(q5k);
};
struct Q4KTraits {
    static constexpr int DIV = QK, BPI = VDR * WARPS * WARP / QI;
    static constexpr const MmvqKernel* multi[M_LAYOUTS][7] = NMH_MULTI(q4k);
};
struct Q20Traits {
    static constexpr int DIV = 64, BPI = WARPS * WARP / 2;
    static constexpr const MmvqKernel* multi[M_LAYOUTS][7] = NMH_MULTI(q20);
};
struct Q3KTraits {
    static constexpr int DIV = 256, BPI = WARPS * WARP / 16;
    static constexpr const MmvqKernel* multi[M_LAYOUTS][7] = NMH_MULTI(q3k);
};
struct Q6KTraits {
    static constexpr int DIV = 256, BPI = WARPS * WARP / 32;
    static constexpr const MmvqKernel* multi[M_LAYOUTS][7] = NMH_MULTI(q6k);
    static constexpr const MmvqKernel* rowwarp[3][4] = {NMH_ROW4(native_mmvq_rowwarp_q6k_rp1_),
                                                        NMH_ROW4(native_mmvq_rowwarp_q6k_rp2_),
                                                        NMH_ROW4(native_mmvq_rowwarp_q6k_rp4_)};
};
struct IQ4XSTraits {
    static constexpr int DIV = 256, BPI = 4 * WARPS * WARP / 32;
    static constexpr const MmvqKernel* multi[M_LAYOUTS][7] = NMH_MULTI(iq4xs);
};
template<typename Weight, int Qi>
struct SmallTraits;

template<>
struct SmallTraits<Q40Block, 4> {
    static constexpr int DIV = 32, BPI = 2 * WARPS * WARP / 4;
    static constexpr const MmvqKernel* single[2] = {&k_native_mmvq_single_q40_small, &k_native_mmvq_single_q40};
    static constexpr const MmvqKernel* multi[M_LAYOUTS][7] = NMH_MULTI(q40);
};

template<>
struct SmallTraits<Q50Block, 4> {
    static constexpr int DIV = 32, BPI = 2 * WARPS * WARP / 4;
    static constexpr const MmvqKernel* single[2] = {&k_native_mmvq_single_q50_small, &k_native_mmvq_single_q50};
    static constexpr const MmvqKernel* multi[M_LAYOUTS][7] = NMH_MULTI(q50);
};

template<>
struct SmallTraits<Q80Block, 8> {
    static constexpr int DIV = 32, BPI = 2 * WARPS * WARP / 8;
    static constexpr const MmvqKernel* single[2] = {&k_native_mmvq_single_q80_small, &k_native_mmvq_single_q80};
    static constexpr const MmvqKernel* multi[M_LAYOUTS][7] = NMH_MULTI(q80);
};

template<>
struct SmallTraits<IQ4NLBlock, 4> {
    static constexpr int DIV = 32, BPI = 2 * WARPS * WARP / 4;
    static constexpr const MmvqKernel* single[2] = {&k_native_mmvq_single_iq4nl_small, &k_native_mmvq_single_iq4nl};
    static constexpr const MmvqKernel* multi[M_LAYOUTS][7] = NMH_MULTI(iq4nl);
};
struct WideQ4K {
    static constexpr const MmvqKernel* k[8] = NMH_ROW8(native_mmvq_wide_q4k_);
};
struct WideQ5K {
    static constexpr const MmvqKernel* k[8] = NMH_ROW8(native_mmvq_wide_q5k_);
};
struct WideIQ4XS {
    static constexpr const MmvqKernel* k[8] = NMH_ROW8(native_mmvq_wide_iq4xs_);
};
struct Wide32Q8 {
    static constexpr const MmvqKernel* k[8] = NMH_ROW8(native_mmvq_wide32_q80_);
};
struct Wide32IQ4NL {
    static constexpr const MmvqKernel* k[8] = NMH_ROW8(native_mmvq_wide32_iq4nl_);
};

// SYCL port: the decode mmvq kernels as Xe2-native SIMD16 sub-groups (twice the registers per lane, same 128
// threads per group) or as the CUDA-shaped 32-lane warp (default). STRATA_MMVQ_SG: 16 = all SIMD16, 1 = IQ4_XS
// only, 2 = short outputs (n_out <= 1024) only, 3 = both. mmvq_sg_bench alone: SIMD16 wins on IQ4_XS (up to
// 1.45x) and short outputs (up to 1.3x), loses 10-25% on the large K-quant projections.
inline int mmvq_sg() {
    static const int v = std::getenv("STRATA_MMVQ_SG") ? std::atoi(std::getenv("STRATA_MMVQ_SG")) : 32;
    return v;
}
inline bool mmvq_sg16(bool iq4xs, int n_out) {
    const int v = mmvq_sg();
    return v == 16 || ((v & 1) && v < 4 && iq4xs) || ((v & 2) && v < 4 && n_out <= 1024);
}

// NW warps per block and ROWS rows per block. The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is the
// ncols = 1 layout and keeps every column bitwise equal to a single-column call. The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
// equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
bool g_multi_exact = true;   // until the upstream layout is timed on an idle GPU (plan rule: default only what is measured)

// S26 STRATA_TSUM=1 (TS): warp 0's NCOLS x ROWS sums as one transposed butterfly (s26_tsum.cuh), bitwise the same
static bool s26_tsum_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    return on;
}

inline bool q6k_a2() {   // STRATA_MMVQ_A2=0 turns the aligned-load path off
    static const bool v = std::getenv("STRATA_MMVQ_A2") == nullptr || std::atoi(std::getenv("STRATA_MMVQ_A2")) != 0;
    return v;
}

inline int q6k_pf() {   // STRATA_MMVQ_PF: weight-load steps issued ahead in the wide Q6_K kernel (0 = off)
    static const int v = std::getenv("STRATA_MMVQ_PF") ? std::atoi(std::getenv("STRATA_MMVQ_PF")) : 0;
    return v;
}
inline int q6k_wide_rpw() {
    static const int v = std::getenv("STRATA_MMVQ_WIDE_RPW") ? std::atoi(std::getenv("STRATA_MMVQ_WIDE_RPW")) : 1;
    return v;
}
// The wide Q6_K entry points by ncols - 1: weight prefetch (STRATA_MMVQ_PF 2, 3, 4, 6), aligned loads, SIMD16,
// rows per warp (1, 2, 4), and the 224-byte stride experiment (ncols 1-4 and 6).
constexpr const MmvqKernel* q6k_pf_k[4][8] = {NMH_ROW8(native_mmvq_q6k_pf2_), NMH_ROW8(native_mmvq_q6k_pf3_),
                                              NMH_ROW8(native_mmvq_q6k_pf4_), NMH_ROW8(native_mmvq_q6k_pf6_)};
constexpr const MmvqKernel* q6k_a2_k[8] = NMH_ROW8(native_mmvq_q6k_a2_);
constexpr const MmvqKernel* q6k_w16_k[8] = NMH_ROW8(native_mmvq_q6k_w16_);
constexpr const MmvqKernel* q6k_w_k[3][8] = {NMH_ROW8(native_mmvq_q6k_w1_), NMH_ROW8(native_mmvq_q6k_w2_),
                                             NMH_ROW8(native_mmvq_q6k_w4_)};
constexpr const MmvqKernel* q6k_w224_k[8] = {&k_native_mmvq_q6k_w224_1, &k_native_mmvq_q6k_w224_2,
                                             &k_native_mmvq_q6k_w224_3, &k_native_mmvq_q6k_w224_4, nullptr,
                                             &k_native_mmvq_q6k_w224_6, nullptr, nullptr};
template <int NCOLS>
void launch_q6k_wide(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, strata::ze::Stream* s) {
    const int rpw = q6k_wide_rpw();
    if (const int pf = q6k_pf(); pf > 0) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        const strata::ze::Dim3 t{WARP, WARPS, 1};
#define STRATA_Q6PF(UU, I) if (pf == UU) { \
            (*q6k_pf_k[I][NCOLS - 1])(s, {blocks, 1, 1}, t, 0, weights, x_q8_1, y, n_in, n_out); \
            return; }
        STRATA_Q6PF(2, 0) STRATA_Q6PF(3, 1) STRATA_Q6PF(4, 2) STRATA_Q6PF(6, 3)
#undef STRATA_Q6PF
    }
    if (rpw == 1 && q6k_a2()) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        (*q6k_a2_k[NCOLS - 1])(s, {blocks, 1, 1}, {WARP, WARPS, 1}, 0, weights, x_q8_1, y, n_in, n_out);
        return;
    }
    if (rpw == 1 && mmvq_sg16(false, n_out)) {
        constexpr int NWG = WARPS * WARP / 16;
        const unsigned blocks = unsigned((std::size_t(n_out) + NWG - 1) / NWG);
        (*q6k_w16_k[NCOLS - 1])(s, {blocks, 1, 1}, {16, NWG, 1}, 0, weights, x_q8_1, y, n_in, n_out);
        return;
    }
#define STRATA_Q6W(RP, I) if (rpw == RP) { \
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS * RP - 1) / (WARPS * RP)); \
        (*q6k_w_k[I][NCOLS - 1])(s, {blocks, 1, 1}, {WARP, WARPS, 1}, 0, weights, x_q8_1, y, n_in, n_out); \
        return; }
    STRATA_Q6W(1, 0) STRATA_Q6W(2, 1) STRATA_Q6W(4, 2)
#undef STRATA_Q6W
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    (*q6k_w_k[0][NCOLS - 1])(s, {blocks, 1, 1}, {WARP, WARPS, 1}, 0, weights, x_q8_1, y, n_in, n_out);
}
template <typename F, int NCOLS>
void launch_wide(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, strata::ze::Stream* s) {
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    (*F::k[NCOLS - 1])(s, {blocks, 1, 1}, {WARP, WARPS, 1}, 0, weights, x_q8_1, y, n_in, n_out);
}
inline bool wide_k() {   // STRATA_MMVQ_WIDE_K=0: Q4_K/Q5_K/IQ4_XS through the multi kernel (the old path)
    static const bool v = std::getenv("STRATA_MMVQ_WIDE_K") == nullptr || std::atoi(std::getenv("STRATA_MMVQ_WIDE_K")) != 0;
    return v;
}
template <typename F>
bool try_wide(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!wide_k() || ncols < 1 || ncols > 8 || n_in % QK != 0) return false;
    const auto s = strata::ze::stream_of(stream);
    switch (ncols) {
        case 1: launch_wide<F, 1>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 2: launch_wide<F, 2>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 3: launch_wide<F, 3>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 4: launch_wide<F, 4>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 5: launch_wide<F, 5>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 6: launch_wide<F, 6>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 7: launch_wide<F, 7>(weights, x_q8_1, y, n_in, n_out, s); return true;
        default: launch_wide<F, 8>(weights, x_q8_1, y, n_in, n_out, s); return true;
    }
}

template <typename F, int NCOLS>
void launch_wide32(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, strata::ze::Stream* s) {
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    (*F::k[NCOLS - 1])(s, {blocks, 1, 1}, {WARP, WARPS, 1}, 0, weights, x_q8_1, y, n_in, n_out);
}
inline bool wide_32() {   // STRATA_MMVQ_WIDE_32=0: Q8_0/IQ4_NL through the small/multi kernels (the old path)
    static const bool v = std::getenv("STRATA_MMVQ_WIDE_32") == nullptr || std::atoi(std::getenv("STRATA_MMVQ_WIDE_32")) != 0;
    return v;
}
template <typename F>
bool try_wide32(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!wide_32() || ncols < 1 || ncols > 8 || n_in % 32 != 0) return false;
    const auto s = strata::ze::stream_of(stream);
    switch (ncols) {
        case 1: launch_wide32<F, 1>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 2: launch_wide32<F, 2>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 3: launch_wide32<F, 3>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 4: launch_wide32<F, 4>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 5: launch_wide32<F, 5>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 6: launch_wide32<F, 6>(weights, x_q8_1, y, n_in, n_out, s); return true;
        case 7: launch_wide32<F, 7>(weights, x_q8_1, y, n_in, n_out, s); return true;
        default: launch_wide32<F, 8>(weights, x_q8_1, y, n_in, n_out, s); return true;
    }
}

// SYCL port: alignment experiment - Q6_K blocks repacked at a 224-byte stride (every block 16-byte aligned).
template <int NCOLS>
void launch_q6k_wide224(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, strata::ze::Stream* s) {
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    (*q6k_w224_k[NCOLS - 1])(s, {blocks, 1, 1}, {WARP, WARPS, 1}, 0, weights, x_q8_1, y, n_in, n_out);
}
bool g_q6k_wide = std::getenv("STRATA_MMVQ_WIDE") == nullptr || std::atoi(std::getenv("STRATA_MMVQ_WIDE")) != 0;

template <typename F, int NCOLS, int RPW>
void launch_rowwarp(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, strata::ze::Stream* s) {
    const unsigned blocks = unsigned((std::size_t(n_out) + WARPS * RPW - 1) / (WARPS * RPW));
    (*F::rowwarp[RPW == 1 ? 0 : RPW == 2 ? 1 : 2][NCOLS - 1])(s, {blocks, 1, 1}, {WARP, WARPS, 1}, 0, weights, x_q8_1,
                                                              y, n_in, n_out);
}
inline int rowwarp_rpw() {   // STRATA_MMVQ_RPW: 0 = the shared kernel, 1/2/4 = rows per warp (bench knob)
    static const int v = std::getenv("STRATA_MMVQ_RPW") ? std::atoi(std::getenv("STRATA_MMVQ_RPW")) : 0;   // default: the shared kernel (measured equal at warm clocks)
    return v;
}

template <typename F, int NCOLS>
void launch_multi_n(const void *weights, const void *x_q8_1, float *y, int n_in,
                    int n_out, strata::ze::Stream* s) {
    constexpr int c = NCOLS - 2;
    if (!g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        constexpr int ROWS = 1;   // SYCL port: one row per work-group (more groups: the kernel is latency-bound here)
        const unsigned blocks = unsigned((std::size_t(n_out) + ROWS - 1) / ROWS);
        (*F::multi[M_UP][c])(s, {blocks, 1, 1}, {WARP, NW, 1}, 0, weights, x_q8_1, y, n_in, n_out);
        return;
    }
    if (mmvq_sg16(std::is_same_v<F, IQ4XSTraits>, n_out)) {   // same 128 threads as 8 SIMD16 sub-groups
        constexpr int NW16 = WARPS * WARP / 16;
        const strata::ze::Dim3 t16{16, NW16, 1};
        if (n_in / F::DIV < F::BPI) {
            const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
            (*F::multi[M_SG16_SMALL][c])(s, {blocks, 1, 1}, t16, 0, weights, x_q8_1, y, n_in, n_out);
        } else {
            (*F::multi[M_SG16][c])(s, {unsigned(n_out), 1, 1}, t16, 0, weights, x_q8_1, y, n_in, n_out);
        }
        return;
    }
    const strata::ze::Dim3 threads{WARP, WARPS, 1};
    // #783 PR-h (stuchapin909): two rows per block for every K, not only small K - each row keeps its own partial sums and
    // reduction, so a row's result does not depend on its neighbour; STRATA_NO_MMVQ_ROWS2=1 keeps one row for large K
    static const bool rows1 = [] {
        const char* v = std::getenv("STRATA_NO_MMVQ_ROWS2");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (rows1 && n_in / F::DIV >= F::BPI) {
        if (s26_tsum_on()) {
            (*F::multi[M_ROWS1_TS][c])(s, {unsigned(n_out), 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
        } else {
            (*F::multi[M_ROWS1][c])(s, {unsigned(n_out), 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
        }
        return;
    }
    constexpr int ROWS = 2;
    const unsigned blocks = unsigned((std::size_t(n_out) + ROWS - 1) / ROWS);
    if (s26_tsum_on()) {
        (*F::multi[M_ROWS2_TS][c])(s, {blocks, 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    } else {
        (*F::multi[M_ROWS2][c])(s, {blocks, 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    }
}

template<typename F>
void launch_multi(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                  void* stream) {
    const auto s = strata::ze::stream_of(stream);
    switch (ncols) {
        case 2: launch_multi_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: launch_multi_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: launch_multi_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: launch_multi_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: launch_multi_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: launch_multi_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: launch_multi_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ multi-column launch requires 2 <= ncols <= 8");
    }
}

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");
}


#if defined(STRATA_HIP_GFX906)
bool g_wave_off = std::getenv("STRATA_MMVQ_WAVE") && std::string(std::getenv("STRATA_MMVQ_WAVE")) == "0";
template<typename F, int NCOLS>
void wave_launch_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    constexpr int R = 1;
    const unsigned blocks = unsigned((std::size_t(n_out) + 4 * R - 1) / (4 * R));
    native_mmvq_wave_kernel<F, NCOLS, R><<<blocks, 256, 0, s>>>(w, x, y, n_in, n_out);
}
template<typename F>
void wave_launch(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 1: wave_launch_n<F, 1>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 2: wave_launch_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: wave_launch_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: wave_launch_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: wave_launch_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: wave_launch_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: wave_launch_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: wave_launch_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ (wave) requires 1 <= ncols <= 8");
    }
}
#define STRATA_WAVE_MMVQ(...) \
    if (!g_wave_off) { wave_launch<__VA_ARGS__>(weights, x_q8_1, y, n_in, n_out, ncols, stream); return; }
#else
#define STRATA_WAVE_MMVQ(...)
#endif

template<typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y,
                int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(SmallTraits<Weight, Qi>)
    if (ncols > 1) {
        launch_multi<SmallTraits<Weight, Qi>>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        return;
    }
    const auto s = strata::ze::stream_of(stream);
    const strata::ze::Dim3 threads{WARP, WARPS, 1};
    if (n_in / 32 < 2 * WARPS * WARP / Qi) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        (*SmallTraits<Weight, Qi>::single[0])(s, {blocks, 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    } else {
        (*SmallTraits<Weight, Qi>::single[1])(s, {unsigned(n_out), 1, 1}, threads, 0, weights, x_q8_1, y, n_in,
                                              n_out);
    }
}

template<typename Weight, int Qi>
void small_f32(const void* weights, const float* x, void* scratch_q8_1,
               float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    small_mmvq<Weight, Qi>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

constexpr int Q8P_PERSIST_GRID = 640;
constexpr int Q8P_PERSIST_MAX_IN = 4096;

struct Q8Packed { const int8_t *qs; const uint16_t * d; int n_in, n_out; };
std::unordered_map<const void*, Q8Packed>& q8_packed_registry() {
    static std::unordered_map<const void*, Q8Packed> registry;
    return registry;
}

constexpr const decltype(k_native_mmvq_q8p_1)* q8p_k[8] = NMH_ROW8(native_mmvq_q8p_);
constexpr const decltype(k_native_mmvq_q8p_1)* q8p_ts_k[8] = NMH_ROW8(native_mmvq_q8p_ts_);
template <int NCOLS>
void q8_packed_launch(const Q8Packed &w, const Q81Block *x, float *y,
                      strata::ze::Stream* s) {
    const bool persist = w.n_in <= Q8P_PERSIST_MAX_IN;
    const int grid = persist ? (std::min)(w.n_out, Q8P_PERSIST_GRID) : w.n_out;
    if (s26_tsum_on())
        (*q8p_ts_k[NCOLS - 1])(s, {unsigned(grid), 1, 1}, {WARP, WARPS, 1}, 0, w.qs, w.d, x, y, w.n_in, w.n_out,
                               persist ? grid : w.n_out);
    else
        (*q8p_k[NCOLS - 1])(s, {unsigned(grid), 1, 1}, {WARP, WARPS, 1}, 0, w.qs, w.d, x, y, w.n_in, w.n_out,
                            persist ? grid : w.n_out);
}

bool q8_packed_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    auto& registry = q8_packed_registry();
    if (registry.empty() || (ncols > 1 && !g_multi_exact)) return false;   // packed = the EXACT layout only
    const auto found = registry.find(weights);
    if (found == registry.end() || found->second.n_in != n_in || found->second.n_out != n_out) return false;
    validate_shape(n_in, ncols, 32);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::ze::stream_of(stream);
    switch (ncols) {
        case 1: q8_packed_launch<1>(found->second, x, y, s); break;
        case 2: q8_packed_launch<2>(found->second, x, y, s); break;
        case 3: q8_packed_launch<3>(found->second, x, y, s); break;
        case 4: q8_packed_launch<4>(found->second, x, y, s); break;
        case 5: q8_packed_launch<5>(found->second, x, y, s); break;
        case 6: q8_packed_launch<6>(found->second, x, y, s); break;
        case 7: q8_packed_launch<7>(found->second, x, y, s); break;
        case 8: q8_packed_launch<8>(found->second, x, y, s); break;
        default: throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
    }
    return true;
}


int g_q6p_grid_override = 0;   // the self-test's grid sweep (STRATA_Q6P_SELFTEST=2)
int q6p_grid() {
    if (g_q6p_grid_override > 0) return g_q6p_grid_override;
    static const int grid = [] {
        const char* v = std::getenv("STRATA_Q6P_GRID");
        const int g = v ? std::atoi(v) : 0;
        return g > 0 ? g : 640;
    }();
    return grid;
}

bool g_q6_bypass = false;   // the load-time self-test times the GGUF-layout kernels with the copy registered

// the packed planes of one Q6_K matrix (owned: allocated by native_q6_k_pack, freed by native_q6_k_unpack)
struct Q6Packed {
    void *base; const int * ql; const int * qh; const int8_t * sc;
    const uint16_t * d;
    int n_in, n_out;
};
std::unordered_map<const void*, Q6Packed>& q6_packed_registry() {
    static std::unordered_map<const void*, Q6Packed> registry;
    return registry;
}

constexpr const decltype(k_native_mmvq_q6p_1)* q6p_k[8] = NMH_ROW8(native_mmvq_q6p_);
constexpr const decltype(k_native_mmvq_q6p_1)* q6p_ts_k[8] = NMH_ROW8(native_mmvq_q6p_ts_);
template <int NCOLS>
void q6_packed_launch(const Q6Packed &w, const Q81Block *x, float *y,
                      strata::ze::Stream* s) {
    const int grid = (std::min)(w.n_out, q6p_grid());
    if (s26_tsum_on())
        (*q6p_ts_k[NCOLS - 1])(s, {unsigned(grid), 1, 1}, {WARP, WARPS, 1}, 0, w.ql, w.qh, w.sc, w.d, x, y, w.n_in,
                               w.n_out, grid);
    else
        (*q6p_k[NCOLS - 1])(s, {unsigned(grid), 1, 1}, {WARP, WARPS, 1}, 0, w.ql, w.qh, w.sc, w.d, x, y, w.n_in,
                            w.n_out, grid);
}

bool q6_packed_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    auto& registry = q6_packed_registry();
    if (registry.empty() || g_q6_bypass || (ncols > 1 && !g_multi_exact)) return false;
    const auto found = registry.find(weights);
    if (found == registry.end() || found->second.n_in != n_in || found->second.n_out != n_out) return false;
    validate_shape(n_in, ncols, 256);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::ze::stream_of(stream);
    switch (ncols) {
        case 1: q6_packed_launch<1>(found->second, x, y, s); break;
        case 2: q6_packed_launch<2>(found->second, x, y, s); break;
        case 3: q6_packed_launch<3>(found->second, x, y, s); break;
        case 4: q6_packed_launch<4>(found->second, x, y, s); break;
        case 5: q6_packed_launch<5>(found->second, x, y, s); break;
        case 6: q6_packed_launch<6>(found->second, x, y, s); break;
        case 7: q6_packed_launch<7>(found->second, x, y, s); break;
        case 8: q6_packed_launch<8>(found->second, x, y, s); break;
        default: throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
    }
    return true;
}

constexpr const decltype(k_native_mmvq_pair_q80_2)* q80_pair_k[7] = NMH_ROW7(native_mmvq_pair_q80_);
constexpr const decltype(k_native_mmvq_pair_q80_2)* q80_pair_ts_k[7] = NMH_ROW7(native_mmvq_pair_q80_ts_);
#undef NMH_MULTI
#undef NMH_ROW8
#undef NMH_ROW7
#undef NMH_ROW4
} // namespace

bool native_q8_0_packed_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_Q8_PACKED");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}

bool native_q8_0_packed_eligible(int n_in, int n_out) {
    return n_in > 0 && n_in % 32 == 0 && n_in / 32 >= 2 * WARPS * WARP / 8 && n_in <= Q8P_PERSIST_MAX_IN &&
           n_out >= 2048;
}

void native_q8_0_pack_host(const void* gguf_blocks, void* out, int n_in, int n_out) {
    const auto* blocks = static_cast<const Q80Block*>(gguf_blocks);
    auto* qs = static_cast<int8_t*>(out);
    auto *d = reinterpret_cast<uint16_t *>(qs + std::size_t(n_in) * n_out);
    const std::size_t count = std::size_t(n_in / 32) * n_out;
    for (std::size_t b = 0; b < count; ++b) {
        std::memcpy(qs + b * 32, blocks[b].qs, 32);
        std::memcpy(d + b, &blocks[b].d, sizeof(uint16_t));
    }
}

void native_q8_0_packed_register(const void* gguf_weights, const void* packed, int n_in, int n_out) {
    validate_pointer(gguf_weights);
    validate_pointer(packed);
    if (!native_q8_0_packed_eligible(n_in, n_out)) throw std::invalid_argument("Q8_0 packed: ineligible shape");
    const auto* qs = static_cast<const int8_t*>(packed);
    q8_packed_registry()[gguf_weights] = Q8Packed{
        qs,
        reinterpret_cast<const uint16_t *>(qs + std::size_t(n_in) * n_out),
        n_in, n_out};
}

void native_q8_0_packed_unregister(const void* gguf_weights) { q8_packed_registry().erase(gguf_weights); }

bool native_q6_k_packed_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_Q6_PACKED");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}

bool native_q6_k_pack(const void *weights, int n_in, int n_out,
                      const char *what) {
    if (!weights || n_in <= 0 || n_in % 256 || n_in / 256 < WARPS * WARP / 32 || n_out < 2048) return false;
    if (q6_packed_registry().count(weights)) return true;
    const std::size_t nb = std::size_t(n_in / 256) * std::size_t(n_out);
    const std::size_t bytes = nb * sizeof(Q6KBlock);
    std::vector<Q6KBlock> host(nb);
    strata::ze::default_stream().memcpy(host.data(), weights, bytes);
    strata::ze::default_stream().sync();
    // planes: ql | qh | scales | d, each 256-byte aligned
    const std::size_t o_qh = (nb * 128 + 255) & ~std::size_t(255);
    const std::size_t o_sc = (o_qh + nb * 64 + 255) & ~std::size_t(255);
    const std::size_t o_d = (o_sc + nb * 16 + 255) & ~std::size_t(255);
    const std::size_t total = o_d + nb * 2;
    std::vector<uint8_t> packed(total, 0);
    for (std::size_t b = 0; b < nb; ++b) {
        std::memcpy(packed.data() + b * 128, host[b].ql, 128);
        std::memcpy(packed.data() + o_qh + b * 64, host[b].qh, 64);
        std::memcpy(packed.data() + o_sc + b * 16, host[b].scales, 16);
        std::memcpy(packed.data() + o_d + b * 2, &host[b].d, 2);
    }
    void* dev = strata::ze::malloc_device(total);
    if (dev == nullptr) return false;
    strata::ze::default_stream().memcpy(dev, packed.data(), total);
    strata::ze::default_stream().sync();
    auto* base = static_cast<uint8_t*>(dev);
    q6_packed_registry()[weights] =
        Q6Packed{dev,
                 reinterpret_cast<const int *>(base),
                 reinterpret_cast<const int *>(base + o_qh),
                 reinterpret_cast<const int8_t *>(base + o_sc),
                 reinterpret_cast<const uint16_t *>(base + o_d),
                 n_in,
                 n_out};
    std::fprintf(stderr, "strata: STRATA_Q6_PACKED=1 packed the %s (%d x %d, +%.1f MiB)\n", what, n_out, n_in,
                 total / 1048576.0);
    // STRATA_Q6P_SELFTEST=1: bitwise check against the GGUF-layout kernels (1..4 columns) and paired timings
    if (const char* v = std::getenv("STRATA_Q6P_SELFTEST"); v && (v[0] == '1' || v[0] == '2')) {
        strata::ze::Stream* s = new strata::ze::Stream();
        const int nq = n_in / Q8K;
        std::vector<Q81Block> hx(std::size_t(4) * nq);
        uint32_t r = 12345u;
        for (auto& blk : hx) {
            for (auto& q : blk.qs) { r = r * 1664525u + 1013904223u; q = int8_t(int((r >> 24) & 255u) - 128); }
            r = r * 1664525u + 1013904223u;
            blk.ds[0] = f16_from_f32(0.001f + float(r >> 8) * 1e-9f);
            blk.ds[1] = f16_from_f32(0.0f);
        }
        void *dx = nullptr, *ya = nullptr, *yb = nullptr;
        dx = strata::ze::malloc_device(hx.size() * sizeof(Q81Block));
        ya = strata::ze::malloc_device(std::size_t(4) * n_out * 4);
        yb = strata::ze::malloc_device(std::size_t(4) * n_out * 4);
        strata::ze::default_stream().memcpy(dx, hx.data(), hx.size() * sizeof(Q81Block));
        strata::ze::default_stream().sync();
        std::vector<float> a(std::size_t(4) * n_out), c(std::size_t(4) * n_out);
        bool same = true;
        for (int nc = 1; nc <= 4; ++nc) {
            g_q6_bypass = true;
            native_q6_k_mmvq(weights, dx, (float*) ya, n_in, n_out, nc, s);
            g_q6_bypass = false;
            native_q6_k_mmvq(weights, dx, (float*) yb, n_in, n_out, nc, s);
            s->sync();
            strata::ze::default_stream().memcpy(a.data(), ya, std::size_t(nc) * n_out * 4);
            strata::ze::default_stream().sync();
            strata::ze::default_stream().memcpy(c.data(), yb, std::size_t(nc) * n_out * 4);
            strata::ze::default_stream().sync();
            same = same && std::memcmp(a.data(), c.data(), std::size_t(nc) * n_out * 4) == 0;
        }
        strata::ze::Event *e0 = new strata::ze::Event(true), *e1 = new strata::ze::Event(true);
        for (int nc : {1, 2, 4}) {
            std::vector<float> tb, tc;
            for (int k = 0; k < 11; ++k) {
                for (int pass = 0; pass < 2; ++pass) {
                    const bool cand = (pass == 0) == (k % 2 == 1);
                    g_q6_bypass = !cand;
                    s->record(*e0);
                    for (int it = 0; it < 5; ++it) native_q6_k_mmvq(weights, dx, (float*) (cand ? yb : ya), n_in, n_out, nc, s);
                    s->record(*e1);
                    e1->sync();
                    const float ms = e1->elapsed_ms_since(*e0);
                    (cand ? tc : tb).push_back(ms * 1000.0f / 5);
                }
            }
            g_q6_bypass = false;
            std::sort(tb.begin(), tb.end()); std::sort(tc.begin(), tc.end());
            std::fprintf(stderr, "strata: Q6_K packed self-test %s ncols %d: GGUF %.1f us (%.1f-%.1f), packed %.1f us "
                         "(%.1f-%.1f), 11 paired samples\n", what, nc, tb[5], tb[0], tb[10], tc[5], tc[0], tc[10]);
        }
        std::fprintf(stderr, "strata: Q6_K packed self-test %s: outputs %s (1-4 columns)\n", what,
                     same ? "BITWISE EQUAL" : "DIFFER");
        if (v[0] == '2') {   // grid sweep of the packed kernel
            for (int grid : {160, 320, 640, 1280, 2560, 5120, 20480}) {
                g_q6p_grid_override = grid;
                for (int nc : {1, 2, 4}) {
                    std::vector<float> tc;
                    for (int k = 0; k < 7; ++k) {
                        s->record(*e0);
                        for (int it = 0; it < 5; ++it) native_q6_k_mmvq(weights, dx, (float*) yb, n_in, n_out, nc, s);
                        s->record(*e1);
                        e1->sync();
                        const float ms = e1->elapsed_ms_since(*e0);
                        tc.push_back(ms * 1000.0f / 5);
                    }
                    std::sort(tc.begin(), tc.end());
                    std::fprintf(stderr, "strata: Q6_K packed grid sweep %s grid %d ncols %d: %.1f us (%.1f-%.1f)\n",
                                 what, grid, nc, tc[3], tc[0], tc[6]);
                }
            }
            g_q6p_grid_override = 0;
        }
        delete e0; delete e1;
        strata::ze::free(dx);
        strata::ze::free(ya);
        strata::ze::free(yb);
        delete s;
        if (!same) { native_q6_k_unpack(weights); return false; }
    }
    return true;
}

void native_q6_k_unpack(const void* weights) {
    auto& registry = q6_packed_registry();
    const auto found = registry.find(weights);
    if (found == registry.end()) return;
    strata::ze::free(found->second.base);
    registry.erase(found);
}

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
void native_mmvq_set_q6k_wide(bool on) { g_q6k_wide = on; }
void native_q6_k_mmvq_stride224(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                                void* stream) {
    const auto s = strata::ze::stream_of(stream);
    switch (ncols) {
        case 1: launch_q6k_wide224<1>(weights, x_q8_1, y, n_in, n_out, s); return;
        case 2: launch_q6k_wide224<2>(weights, x_q8_1, y, n_in, n_out, s); return;
        case 3: launch_q6k_wide224<3>(weights, x_q8_1, y, n_in, n_out, s); return;
        case 4: launch_q6k_wide224<4>(weights, x_q8_1, y, n_in, n_out, s); return;
        default: launch_q6k_wide224<6>(weights, x_q8_1, y, n_in, n_out, s); return;
    }
}
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    k_native_mmvq_quantize_q8_1(strata::ze::stream_of(stream), {blocks, 1, 1}, {QUANT_THREADS, 1, 1}, 0, x, x_q8_1,
                                n_total);
}

void native_swiglu_quantize_q8_1(const float* gate, const float* up, void* x_q8_1,
                                 int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(gate);
    validate_pointer(up);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    k_native_mmvq_swiglu_quantize_q8_1(strata::ze::stream_of(stream), {blocks, 1, 1}, {QUANT_THREADS, 1, 1}, 0, gate,
                                       up, x_q8_1, n_total);
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (try_wide<WideQ5K>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;
    STRATA_WAVE_MMVQ(Q5KTraits)
    if (ncols > 1) {
        launch_multi<Q5KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        return;
    }
    const auto s = strata::ze::stream_of(stream);
    const strata::ze::Dim3 threads{WARP, WARPS, 1};
    if (n_in / QK < VDR * WARPS * WARP / QI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        k_native_mmvq_single_q5k_small(s, {blocks, 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    } else {
        k_native_mmvq_single_q5k(s, {unsigned(n_out), 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    }
}

void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q5_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q20Traits)
    if (ncols > 1) {
        launch_multi<Q20Traits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        return;
    }
    const auto s = strata::ze::stream_of(stream);
    const strata::ze::Dim3 threads{WARP, WARPS, 1};
    if (n_in / 64 < WARPS * WARP / 2) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        k_native_mmvq_single_q20_small(s, {blocks, 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    } else {
        k_native_mmvq_single_q20(s, {unsigned(n_out), 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    }
}

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q2_0_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q3KTraits)
    if (ncols > 1) {
        launch_multi<Q3KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        return;
    }
    const auto s = strata::ze::stream_of(stream);
    const strata::ze::Dim3 threads{WARP, WARPS, 1};
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        k_native_mmvq_single_q3k_small(s, {blocks, 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    } else {
        k_native_mmvq_single_q3k(s, {unsigned(n_out), 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    }
}

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q3_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (try_wide<WideIQ4XS>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;
    STRATA_WAVE_MMVQ(IQ4XSTraits)
    if (ncols > 1) {
        launch_multi<IQ4XSTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        return;
    }
    const auto s = strata::ze::stream_of(stream);
    const strata::ze::Dim3 threads{WARP, WARPS, 1};
    if (n_in / 256 < 4 * WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        k_native_mmvq_single_iq4xs_small(s, {blocks, 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    } else {
        k_native_mmvq_single_iq4xs(s, {unsigned(n_out), 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    }
}

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_iq4_xs_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (try_wide<WideQ4K>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;
    STRATA_WAVE_MMVQ(Q4KTraits)
    if (ncols > 1) {
        launch_multi<Q4KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        return;
    }
    const auto s = strata::ze::stream_of(stream);
    const strata::ze::Dim3 threads{WARP, WARPS, 1};
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        k_native_mmvq_single_q4k_small(s, {blocks, 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    } else {
        k_native_mmvq_single_q4k(s, {unsigned(n_out), 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    }
}

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q4_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    if (q6_packed_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;   // STRATA_Q6_PACKED=1 heads
    // SYCL port: 16-byte-load kernel for the decode shapes (ncols <= 4)
    // With the aligned loads (load16_a2) the wide kernel wins on every Q6_K shape and window width: 2.6-3.4x the
    // misaligned-load kernels in q6k_align_bench, decode 44.9 -> 54 tok/s on the Coder, output tokens identical.
    // Without them (STRATA_MMVQ_A2=0) it won only on n_out >= 4096 at 1-4 columns, the old gate.
    static const int wide_min = std::getenv("STRATA_MMVQ_WIDE_MIN") ? std::atoi(std::getenv("STRATA_MMVQ_WIDE_MIN")) : q6k_a2() ? 0 : 4096;
    static const int wide_maxc = std::getenv("STRATA_MMVQ_WIDE_MAXC") ? std::atoi(std::getenv("STRATA_MMVQ_WIDE_MAXC")) : q6k_a2() ? 8 : 4;
    if (g_q6k_wide && ncols >= 1 && ncols <= wide_maxc && n_in % 256 == 0 && n_out >= wide_min) {
        const auto s = strata::ze::stream_of(stream);
        switch (ncols) {
            case 1: launch_q6k_wide<1>(weights, x_q8_1, y, n_in, n_out, s); return;
            case 2: launch_q6k_wide<2>(weights, x_q8_1, y, n_in, n_out, s); return;
            case 3: launch_q6k_wide<3>(weights, x_q8_1, y, n_in, n_out, s); return;
            case 4: launch_q6k_wide<4>(weights, x_q8_1, y, n_in, n_out, s); return;
            case 5: launch_q6k_wide<5>(weights, x_q8_1, y, n_in, n_out, s); return;
            case 6: launch_q6k_wide<6>(weights, x_q8_1, y, n_in, n_out, s); return;
            case 7: launch_q6k_wide<7>(weights, x_q8_1, y, n_in, n_out, s); return;
            default: launch_q6k_wide<8>(weights, x_q8_1, y, n_in, n_out, s); return;
        }
    }
    // SYCL port: the row-blocked kernel for the decode shapes (ncols <= 4)
    if (ncols >= 1 && ncols <= 4 && rowwarp_rpw() > 0 && n_in % 256 == 0) {
        strata::ze::Stream* s = strata::ze::stream_of(stream);
        const int rpw = rowwarp_rpw();
#define STRATA_RW(NC, RP) if (ncols == NC && rpw == RP) { launch_rowwarp<Q6KTraits, NC, RP>(weights, x_q8_1, y, n_in, n_out, s); return; }
        STRATA_RW(1, 1) STRATA_RW(1, 2) STRATA_RW(1, 4) STRATA_RW(2, 1) STRATA_RW(2, 2) STRATA_RW(2, 4)
        STRATA_RW(3, 1) STRATA_RW(3, 2) STRATA_RW(3, 4) STRATA_RW(4, 1) STRATA_RW(4, 2) STRATA_RW(4, 4)
#undef STRATA_RW
    }

    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q6KTraits)
    if (ncols > 1) {
        launch_multi<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        return;
    }
    const auto s = strata::ze::stream_of(stream);
    const strata::ze::Dim3 threads{WARP, WARPS, 1};
    if (n_in / 256 < WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        k_native_mmvq_single_q6k_small(s, {blocks, 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    } else {
        k_native_mmvq_single_q6k(s, {unsigned(n_out), 1, 1}, threads, 0, weights, x_q8_1, y, n_in, n_out);
    }
}

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q6_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q40Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q40Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q50Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q50Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

template <int NCOLS>
void q8_0_pair_launch(const Q80Block *w1, const Q80Block *w2, const Q81Block *x,
                      float *y1, float *y2, int n_in, int n_out,
                      strata::ze::Stream* s) {
    const strata::ze::Dim3 threads{WARP, WARPS, 1};
    if (s26_tsum_on())
        (*q80_pair_ts_k[NCOLS - 2])(s, {unsigned(2 * n_out), 1, 1}, threads, 0, w1, w2, x, y1, y2, n_in, n_out);
    else
        (*q80_pair_k[NCOLS - 2])(s, {unsigned(2 * n_out), 1, 1}, threads, 0, w1, w2, x, y1, y2, n_in, n_out);
}

bool native_mmvq_pair(int ggml_type, const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2,
                      int n_in, int n_out, int ncols, void* stream) {
    using F = SmallTraits<Q80Block, 8>;
    // only where both calls would run native_mmvq_multi_kernel<F, ncols, WARPS, 1> (exact layout, not packed)
    if (ggml_type != 8 || !g_multi_exact || ncols < 2 || ncols > 8 || n_out <= 0 || n_in <= 0 || n_in % 32 != 0 ||
        n_in / F::DIV < F::BPI || !w1 || !w2 || !x_q8_1 || !y1 || !y2 || !stream)
        return false;
    const auto& reg = q8_packed_registry();
    if (reg.count(w1) || reg.count(w2)) return false;
    const auto* a = static_cast<const Q80Block*>(w1);
    const auto* b = static_cast<const Q80Block*>(w2);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = strata::ze::stream_of(stream);
    switch (ncols) {
        case 2: q8_0_pair_launch<2>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 3: q8_0_pair_launch<3>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 4: q8_0_pair_launch<4>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 5: q8_0_pair_launch<5>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 6: q8_0_pair_launch<6>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 7: q8_0_pair_launch<7>(a, b, x, y1, y2, n_in, n_out, s); break;
        default: q8_0_pair_launch<8>(a, b, x, y1, y2, n_in, n_out, s); break;
    }
    return true;
}

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    if (q8_packed_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;   // STRATA_Q8_PACKED=1 tensors
    if (try_wide32<Wide32Q8>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;
    small_mmvq<Q80Block, 8>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q80Block, 8>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    if (try_wide32<Wide32IQ4NL>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;
    small_mmvq<IQ4NLBlock, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<IQ4NLBlock, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 7: block_elems = 32; block_bytes = 24; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 7: iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

} // namespace strata::kernels
