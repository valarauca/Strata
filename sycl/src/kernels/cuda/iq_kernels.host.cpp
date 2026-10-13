// src/kernels/cuda/iq_kernels.host.cpp - the host side of iq_kernels.dp.cpp: the format tables and the launchers.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/ze/kernel_abi.hpp"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "iq_kernels.kern.inc"

namespace {

// The device code's format traits the launchers read (iq_kernels.dp.cpp): values per block, the formats with a Split
// (the decode-once kernels) and the ones that can stage their codebook grid.
template<int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256; };
template<> struct Fmt<17> { static constexpr int qk = 256; };
template<> struct Fmt<18> { static constexpr int qk = 256; };
template<> struct Fmt<20> { static constexpr int qk = 32; };
template<> struct Fmt<21> { static constexpr int qk = 256; };
template<> struct Fmt<23> { static constexpr int qk = 256; };
template<> struct Fmt<22> { static constexpr int qk = 256; };
template<> struct Fmt<29> { static constexpr int qk = 256; };
template<> struct Fmt<42> { static constexpr int qk = 64; };
template<> struct Fmt<12> { static constexpr int qk = 256; };
template<> struct Fmt<13> { static constexpr int qk = 256; };
template<> struct Fmt<14> { static constexpr int qk = 256; };
template<> struct Fmt<7> { static constexpr int qk = 32; };
template<> struct Fmt<2> { static constexpr int qk = 32; };
template<> struct Fmt<3> { static constexpr int qk = 32; };
template<> struct Fmt<6> { static constexpr int qk = 32; };
template<> struct Fmt<8> { static constexpr int qk = 32; };
template<int TY> inline constexpr bool kSplit =
    TY == 16 || TY == 17 || TY == 22 || TY == 18 || TY == 21 || TY == 29 || TY == 20 || TY == 23 || TY == 42;
template<int TY> inline constexpr bool kStageIqGrid = (TY == 16 || TY == 17 || TY == 22 || TY == 29);

// The formats of each role, one list each so a type cannot be in one switch and missing from another.  Every
// entry is a kernel template for each CUDA architecture of the build, hence two lists rather than one.
#ifdef STRATA_Q6K_EXPERTS   // opt-in build (-DSTRATA_Q6K_EXPERTS=ON): one more instance per kernel, loaded at start
#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(14) X(6) X(2) X(3) X(8)
#else
#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(6) X(2) X(3) X(8)
#endif
#define STRATA_D_FMTS(X) X(20) X(23) X(42) X(7) X(6) X(2) X(3) X(8)
#define STRATA_MMVQ_FMTS(X) X(16) X(17) X(18) X(20) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(7) X(6) X(2) X(3) X(8)

// the grouped expert kernels' lanes per row (iq_kernels.dp.cpp)
#ifndef STRATA_EXPERT_LANES
#define STRATA_EXPERT_LANES 8
#endif
constexpr int kExpertLanes = STRATA_EXPERT_LANES;
constexpr int kExpertRows = 256 / kExpertLanes;
constexpr int GU_ROWS = 8;     // rows per block (one warp each)

// the types dq_dispatch dequantizes
bool is_iq(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
#ifdef STRATA_Q6K_EXPERTS
           t == 12 || t == 13 || t == 14 || t == 7 || t == 6 || t == 2 || t == 3 || t == 8;
#else
           t == 12 || t == 13 || t == 7 || t == 6 || t == 2 || t == 3 || t == 8;
#endif
}
// values per block of the types the grouped expert kernels take (0 = none)
int gu_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_GU_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}
int d_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_D_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}
bool gu_split(int t) {
    switch (t) {
#define STRATA_SP(T) case T: return kSplit<T>;
        STRATA_GU_FMTS(STRATA_SP)
#undef STRATA_SP
        default: return false;
    }
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}
// STRATA_OLD_IQ_MMVQ=1 keeps the per-column kernels (bitwise equal to the new ones; kept for A/B timing)
bool g_old_kernels = env_on("STRATA_OLD_IQ_MMVQ");
// SYCL port (0.1.31 merge): upstream sends the split formats (the i-quants) to native_gu/down_multi_kernel, one warp
// per row, 8 rows a group. On the B70 the port's grouped kernels (8 lanes a row, 32 rows a group, the multi-entry
// dots with aligned loads and the IQ4 codebook in registers) are the measured ones, so they stay the default for
// every format; STRATA_EXPERT_SPLIT=1 takes upstream's multi kernels, launched with their own grid (GU_ROWS rows).
bool g_split_multi = env_on("STRATA_EXPERT_SPLIT");
bool g_no_sub16_gu = env_on("STRATA_NO_SUB16_GU");
// STRATA_IQ_STAGE_GRID=0 disables staging 64-bit i-quant codebook tables into shared memory
bool g_stage_grid = [] {
    const char* v = std::getenv("STRATA_IQ_STAGE_GRID");
    return v == nullptr || v[0] == '\0' || v[0] != '0';
}();
// The single-matrix mmvq (128-thread blocks, one per 4 rows) pays the 2-8 KB staging per block: on the RTX 5070 its
// IQ2_XS / IQ2_S / IQ1_M calls were 10-20% slower staged at 3-8 columns (iq_multi_parity --bench), while the grouped
// expert kernels gain.  So mmvq stages only with STRATA_IQ_STAGE_GRID_MMVQ=1 (bitwise the same either way).
bool g_stage_grid_mmvq = g_stage_grid && [] {
    const char* v = std::getenv("STRATA_IQ_STAGE_GRID_MMVQ");
    return v != nullptr && v[0] == '1';
}();

template <int TY, bool STAGE_GRID = kStageIqGrid<TY>>
void launch_mmvq_multi_nc(ze::Dim3 grid, ze::Dim3 block, ze::Stream* s,
                          const uint8_t *W, size_t rb, const block_q8_1 *X,
                          float *y, int n_in, int n_out, int ncols) {
    switch (ncols) {
    case 1: IqMmvqMulti<TY, 1, STAGE_GRID, true>::k(s, grid, block, 0, W, rb, X, y, n_in, n_out, 1); break;
    case 2: IqMmvqMulti<TY, 2, STAGE_GRID, true>::k(s, grid, block, 0, W, rb, X, y, n_in, n_out, 2); break;
    case 3: IqMmvqMulti<TY, 3, STAGE_GRID, true>::k(s, grid, block, 0, W, rb, X, y, n_in, n_out, 3); break;
    case 4: IqMmvqMulti<TY, 4, STAGE_GRID, true>::k(s, grid, block, 0, W, rb, X, y, n_in, n_out, 4); break;
    case 5: IqMmvqMulti<TY, 5, STAGE_GRID, true>::k(s, grid, block, 0, W, rb, X, y, n_in, n_out, 5); break;
    case 6: IqMmvqMulti<TY, 6, STAGE_GRID, true>::k(s, grid, block, 0, W, rb, X, y, n_in, n_out, 6); break;
        default:
        if (ncols <= 1) IqMmvqMulti<TY, 1, STAGE_GRID, false>::k(s, grid, block, 0, W, rb, X, y, n_in, n_out, ncols);
        else IqMmvqMulti<TY, 8, STAGE_GRID, false>::k(s, grid, block, 0, W, rb, X, y, n_in, n_out, ncols);
            break;
    }
}

template <int TY>
void launch_mmvq(const uint8_t *W, size_t rb, const block_q8_1 *X, float *y,
                 int n_in, int n_out, int ncols, ze::Stream* s) {
    const ze::Dim3 grid{(unsigned)((n_out + 3) / 4)}, block{32, 4};
    if constexpr (!kSplit<TY>) {
        IqMmvq<TY>::k(s, grid, block, 0, W, rb, X, y, n_in, n_out, ncols);
    }
    else if (g_old_kernels) {
        IqMmvq<TY>::k(s, grid, block, 0, W, rb, X, y, n_in, n_out, ncols);
    } else if constexpr (kStageIqGrid<TY>) {
        if (!g_stage_grid_mmvq) {
            launch_mmvq_multi_nc<TY, false>(grid, block, s, W, rb, X, y, n_in, n_out, ncols);
            return;
        }
        launch_mmvq_multi_nc<TY, true>(grid, block, s, W, rb, X, y, n_in, n_out, ncols);
    } else {
        launch_mmvq_multi_nc<TY, false>(grid, block, s, W, rb, X, y, n_in, n_out, ncols);
    }
}

// SYCL port: lanes per row of the port's grouped expert kernels at run time (STRATA_GU_LANES / STRATA_DOWN_LANES:
// 4, 8, 16 or 32; default STRATA_EXPERT_LANES), 256 / lanes rows per work-group. Gate/up rows are 80 dot calls
// (2560 wide), the IQ4_NL down rows 40 (640 wide), so the best split can differ by role.
inline int lanes_env(const char* name) {
    const char* v = std::getenv(name);
    const int n = v ? std::atoi(v) : kExpertLanes;
    return n == 4 || n == 8 || n == 16 || n == 32 ? n : kExpertLanes;
}
inline int gu_lanes() { static const int v = lanes_env("STRATA_GU_LANES"); return v; }
inline int down_lanes() { static const int v = lanes_env("STRATA_DOWN_LANES"); return v; }
template <int TG, int LN>
void launch_gu_port(unsigned groups, ze::Stream* s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                    const int32_t *n_groups, const int32_t *ent_tok, const block_q8_1 *X,
                    const NativeExpertLayout &L, float *gate, float *up) {
    constexpr int ROWS = 256 / LN;
    const unsigned gx = (unsigned) ((2 * L.n_ff + ROWS - 1) / ROWS);
    IqGu<TG, LN>::k(s, {gx, groups, 1}, {256, 1, 1}, 0, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
}
template <int TG>
void launch_gu_lanes(ze::Dim3 grid, ze::Stream* s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                     const int32_t *n_groups, const int32_t *ent_tok, const block_q8_1 *X,
                     const NativeExpertLayout &L, float *gate, float *up) {
    switch (gu_lanes()) {
        case 4: launch_gu_port<TG, 4>(grid.y, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 16: launch_gu_port<TG, 16>(grid.y, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 32: launch_gu_port<TG, 32>(grid.y, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        default: launch_gu_port<TG, 8>(grid.y, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
    }
}
template <int TD, int LN>
void launch_down_port(unsigned groups, ze::Stream* s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                      const int32_t *n_groups, const int32_t *ent_dst, const block_q8_1 *hq,
                      const NativeExpertLayout &L, float *out) {
    constexpr int ROWS = 256 / LN;
    const unsigned gx = (unsigned) ((L.n_embd + ROWS - 1) / ROWS);
    IqDown<TD, LN>::k(s, {gx, groups, 1}, {256, 1, 1}, 0, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
}
template <int TD>
void launch_down_lanes(ze::Dim3 grid, ze::Stream* s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                       const int32_t *n_groups, const int32_t *ent_dst, const block_q8_1 *hq,
                       const NativeExpertLayout &L, float *out) {
    switch (down_lanes()) {
        case 4: launch_down_port<TD, 4>(grid.y, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 16: launch_down_port<TD, 16>(grid.y, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        case 32: launch_down_port<TD, 32>(grid.y, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        default: launch_down_port<TD, 8>(grid.y, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
    }
}

// STRATA_EXPERT_SPLIT=1 (opt-in): upstream's multi-entry kernels, one warp per row, GU_ROWS rows a group, launched with
// their own grid.  The sub-warp variants (SUB16 / SUB_DOWN) are not launched here: they are bitwise equal to the
// generic path and need their own row counts; the port's lane kernels are the default for every format.
template <int TG, bool STAGE_GRID>
void launch_gu_multi(ze::Dim3 grid, ze::Stream* s, const unsigned long long *grp_ptr, const int32_t *grp_start,
                     const int32_t *n_groups, const int32_t *ent_tok, const block_q8_1 *X,
                     const NativeExpertLayout &L, float *gate, float *up) {
    const ze::Dim3 g{(unsigned) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS), grid.y};
    IqGuMulti<TG, STAGE_GRID>::k(s, g, {256, 1, 1}, 0, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
}

template <int TG>
void launch_gu(ze::Dim3 grid, ze::Stream* s,
               const unsigned long long *grp_ptr, const int32_t *grp_start,
               const int32_t *n_groups, const int32_t *ent_tok,
               const block_q8_1 *X, const NativeExpertLayout &L, float *gate,
               float *up) {
    if constexpr (!kSplit<TG>) {
        launch_gu_lanes<TG>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    } else if (g_old_kernels || !g_split_multi) {
        launch_gu_lanes<TG>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    } else {
        if constexpr (kStageIqGrid<TG>) {
            if (g_stage_grid) {
                launch_gu_multi<TG, true>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
                return;
            }
        }
        launch_gu_multi<TG, false>(grid, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    }
}

template <int TD>
void launch_down_multi(ze::Dim3 grid, ze::Stream* s, const unsigned long long *grp_ptr,
                       const int32_t *grp_start, const int32_t *n_groups, const int32_t *ent_dst,
                       const block_q8_1 *hq, const NativeExpertLayout &L, float *out) {
    const unsigned gx = (unsigned) ((L.n_embd + GU_ROWS - 1) / GU_ROWS);
    IqDownMulti<TD>::k(s, {gx, grid.y, 1}, {256, 1, 1}, 0, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
}

template <int TD>
void launch_down(ze::Dim3 grid, ze::Stream* s,
                 const unsigned long long *grp_ptr, const int32_t *grp_start,
                 const int32_t *n_groups, const int32_t *ent_dst,
                 const block_q8_1 *hq, const NativeExpertLayout &L,
                 float *out) {
    if constexpr (!kSplit<TD>) {
        launch_down_lanes<TD>(grid, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    } else if (g_old_kernels || !g_split_multi) {
        launch_down_lanes<TD>(grid, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    } else {
        launch_down_multi<TD>(grid, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    }
}

}  // namespace

void iq_set_old_kernels(bool old) { g_old_kernels = old; }
bool iq_old_kernels() { return g_old_kernels; }

bool iq_supported(int t) noexcept { return is_iq(t); }
bool embed_type_supported(int t) noexcept { return is_iq(t) || t == 30; }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 2: return (size_t) (n / 32) * sizeof(block_q4_0);
        case 3: return (size_t) (n / 32) * sizeof(block_q4_1);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 30: return (size_t) n * 2;   // BF16: the token embedding only (iq_embed_rows, iq_dequant_f32)
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    k_iq_kernels_quantize_q8_1(ze::stream_of(stream), {(unsigned) ((n + 255) / 256), 1, 1}, {256, 1, 1}, 0, x, y, n);
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const size_t rb = iq_row_bytes(t, n_in);
    ze::Stream* s = ze::stream_of(stream);
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    switch (t) {
#define STRATA_MMVQ(T) case T: launch_mmvq<T>(W, rb, X, y, n_in, n_out, ncols, s); break;
        STRATA_MMVQ_FMTS(STRATA_MMVQ)
#undef STRATA_MMVQ
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    k_iq_kernels_dequant_f16(ze::stream_of(stream), {(unsigned) (n / 256), 1, 1}, {32, 1, 1}, 0, t, src, dst);
}

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    k_iq_kernels_embed_rows(ze::stream_of(stream), {(unsigned) (n_embd / 256), (unsigned) n_tok, 1}, {32, 1, 1}, 0, t,
                            (const uint8_t*) table, row_bytes, tokens, n_embd, out);
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    k_iq_kernels_dequant_f32(ze::stream_of(stream), {(unsigned) (n / 256), 1, 1}, {32, 1, 1}, 0, t, src, dst);
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    // checked like the other entry points: an unknown type used to leave `dst` unwritten, a wrong prompt and no error
    if (n_embd % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_gu_f16: type %d / %lld\n", t, (long long) n_embd); std::exit(1); }
    const int64_t per_row = n_embd / 256;
    k_iq_kernels_dequant_gu(ze::stream_of(stream), {(unsigned) (n_ff * per_row), 2, 1}, {32, 1, 1}, 0, t, gate, up, per_row,
                            dst);
}

void iq_dequant_expert_f16(int gt, int dt, const void* gate, const void* up, const void* down, int64_t n_ff,
                           int64_t n_embd, uint16_t* dst_gu, uint16_t* dst_d, void* stream) {
    if (n_embd % 256 != 0 || (n_ff * n_embd) % 256 != 0 || !is_iq(gt) || !is_iq(dt)) {
        std::fprintf(stderr, "iq_dequant_expert_f16: types %d/%d, %lld x %lld\n", gt, dt, (long long) n_ff, (long long) n_embd);
        std::exit(1);
    }
    const int64_t per_row = n_embd / 256, gu_wgs = 2 * n_ff * per_row, d_wgs = n_ff * n_embd / 256;
    k_iq_kernels_dequant_expert(ze::stream_of(stream), {(unsigned) (gu_wgs + d_wgs), 1, 1}, {32, 1, 1}, 0, gt, dt, gate, up,
                                down, per_row, gu_wgs, dst_gu, dst_d);
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = gu_qk(gu_type), qd = d_qk(d_type);
    return qg > 0 && qd > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0 &&
           n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}


#if defined(STRATA_HIP_GFX906)
int g_exp_mode = -1;   // native_expert_set_mode (the bench); -1 = STRATA_EXP_MODE, default 7
int exp_mode() {
    static const int m = [] { const char* v = std::getenv("STRATA_EXP_MODE"); return v ? std::atoi(v) : 7; }();
    return g_exp_mode >= 0 ? g_exp_mode : m;
}
#endif
int g_exp_phase = 0;   // the bench: 0 all, 1 gate/up + swiglu + quantize only, 2 down only

void native_expert_set_mode(int mode, int phase) {
#if defined(STRATA_HIP_GFX906)
    g_exp_mode = mode;
#else
    (void) mode;
#endif
    g_exp_phase = phase;
}

namespace {
bool g_grouped_v1 = env_on("STRATA_GROUPED_V1");
}  // namespace

void native_grouped_set_v1(bool v1) { g_grouped_v1 = v1; }

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (L.n_ff % 32 != 0) { std::fprintf(stderr, "native_expert_grouped: n_ff %lld\n", (long long) L.n_ff); std::exit(1); }
    ze::Stream* s = ze::stream_of(stream);
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    // SYCL port: STRATA_EXPERT_V2 / STRATA_EXPERT_V2K (upstream's AMD-tuned S26 / S27 kernels) are not built here
    static const bool v2_note = [] {
        const char* a = std::getenv("STRATA_EXPERT_V2");
        const char* b = std::getenv("STRATA_EXPERT_V2K");
        if ((a && a[0] == '1') || (b && b[0] == '1'))
            std::fprintf(stderr, "note: STRATA_EXPERT_V2 / STRATA_EXPERT_V2K are AMD-only experiments, ignored on SYCL\n");
        return true;
    }();
    (void) v2_note;
    const bool v1 = g_grouped_v1;
    // #363: a verify window's PCIe call usually has no group: `grid_groups` block rows stride over the groups instead
    // of one row per possible group (the results do not depend on it)
    const int64_t gy = (v1 || grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups;
    const ze::Dim3 ggu{(unsigned)((2 * L.n_ff + kExpertRows - 1) / kExpertRows),
                       (unsigned)gy};
#if defined(STRATA_HIP_GFX906)
    const int em0 = exp_mode();
#endif
#if defined(STRATA_HIP_GFX906)
    const bool fused_gu = em0 == 8 && (L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 || L.gu_type == 23) &&
                          L.n_ff % 32 == 0;
    if (fused_gu && g_exp_phase != 2) {
        const dim3 gl((unsigned) (L.n_ff / 32), (unsigned) cap_groups);
        const size_t sh = (size_t) LDS_NT * (size_t) (L.n_embd / 32) * sizeof(block_q8_1);
        switch (L.gu_type) {
            case 18: native_gu_fused_kernel<18><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            case 21: native_gu_fused_kernel<21><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            case 23: native_gu_fused_kernel<23><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            default: native_gu_fused_kernel<22><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
        }
    }
    if (g_exp_phase != 2 && !fused_gu) {
#else
    if (g_exp_phase != 2) {
#endif
#if defined(STRATA_HIP_GFX906)
    const bool lds_gu = (em0 == 5 || em0 == 6 || em0 == 7 || em0 == 8) && (L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 || L.gu_type == 23);
    if (lds_gu) {
        const dim3 gl((unsigned) ((2 * L.n_ff + LDS_RB - 1) / LDS_RB), (unsigned) cap_groups);
        const size_t sh = (size_t) LDS_NT * (size_t) (L.n_embd / 32) * sizeof(block_q8_1);
        switch (L.gu_type) {
            case 18: if (em0 >= 7) native_gu_lds_kernel<18, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<18, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<18, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            case 21: if (em0 >= 7) native_gu_lds_kernel<21, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<21, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<21, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            case 23: if (em0 >= 7) native_gu_lds_kernel<23, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<23, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            default: if (em0 >= 7) native_gu_lds_kernel<22, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<22, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<22, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        }
    } else {
    const int em = exp_mode() >= 5 ? 2 : exp_mode();
    const dim3 ggu_amd((unsigned) ((2 * L.n_ff + (em == 1 ? 3 : em == 4 ? 31 : 15)) / (em == 1 ? 4 : em == 4 ? 32 : 16)), (unsigned) cap_groups);
#define STRATA_GU_AMD(T) \
    case T: if (em == 1) native_gu_amd_kernel<T, 1><<<ggu_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); \
            else if (em == 4) native_gu_amd_kernel<T, 4><<<ggu_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); \
            else native_gu_amd_kernel<T, 2><<<ggu_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
    // the AMD layouts cover the IQ packs' gate/up types; Unsloth's Q4_K / Q5_K take the CUDA layout below
    const bool amd_gu = L.gu_type == 16 || L.gu_type == 17 || L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 ||
                        L.gu_type == 23 || L.gu_type == 29 || L.gu_type == 42;
    if ((em == 1 || em == 2 || em == 4) && amd_gu) {
        switch (L.gu_type) {
            STRATA_GU_AMD(16) STRATA_GU_AMD(17) STRATA_GU_AMD(18) STRATA_GU_AMD(21) STRATA_GU_AMD(22) STRATA_GU_AMD(23)
            STRATA_GU_AMD(29) STRATA_GU_AMD(42)
        }
    } else
#undef STRATA_GU_AMD
#endif
    switch (L.gu_type) {
#define STRATA_GU(T) case T: launch_gu<T>(ggu, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        STRATA_GU_FMTS(STRATA_GU)
#undef STRATA_GU
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
    }
#if defined(STRATA_HIP_GFX906)
    }
#endif
    const long long nh = (long long) cap_entries * L.n_ff;
#if defined(STRATA_HIP_GFX906)
    // gfx906: the separate SwiGLU and q8_1 passes stay the default (the A/B baseline); the RDNA one-pass
    // kernel (bitwise the same) with STRATA_HIP_SWIGLU_FUSED=1
    static const bool sw_fused = env_on("STRATA_HIP_SWIGLU_FUSED");
    const bool sw_v1 = v1 || !sw_fused;
#else
    const bool sw_v1 = v1;
#endif
    if (sw_v1) {
            k_iq_kernels_swiglu(s, {(unsigned) ((nh + 255) / 256), 1, 1}, {256, 1, 1}, 0, gate, up, h, nh);
            k_iq_kernels_quantize_q8_1(s, {(unsigned) ((nh + 255) / 256), 1, 1}, {256, 1, 1}, 0, h, hq, nh);
    } else {
            k_iq_kernels_swiglu_q8_1(s, {(unsigned) ((nh + 255) / 256), 1, 1}, {256, 1, 1}, 0, gate, up, grp_start, n_groups,
                                     (int) L.n_ff, hq);
    }
    }
    if (g_exp_phase == 1) return;
    const int d_rows = (!g_old_kernels && !g_no_sub16_gu && L.n_ff == 640) ? (L.d_type == 20 ? 32 : (L.d_type == 42 ? 16 : 8)) : 8;
    const ze::Dim3 gd{(unsigned)((L.n_embd + d_rows - 1) / d_rows),
                      (unsigned)gy};
#if defined(STRATA_HIP_GFX906)
    if ((em0 == 7 || em0 == 8) && (L.d_type == 20 || L.d_type == 42)) {
        const dim3 gl((unsigned) ((L.n_embd + LDS_RB - 1) / LDS_RB), (unsigned) cap_groups);
        const size_t sh = (size_t) LDS_NT * (size_t) (L.n_ff / 32) * sizeof(block_q8_1);
        if (L.d_type == 20) native_down_lds_kernel<20><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        else native_down_lds_kernel<42><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        return;
    }
    const int em = exp_mode() >= 5 ? 2 : exp_mode();
    const dim3 gd_amd((unsigned) ((L.n_embd + (em == 1 ? 3 : em == 4 ? 31 : 15)) / (em == 1 ? 4 : em == 4 ? 32 : 16)), (unsigned) cap_groups);
#define STRATA_D_AMD(T) \
    case T: if (em == 1) native_down_amd_kernel<T, 1><<<gd_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); \
            else if (em == 4) native_down_amd_kernel<T, 4><<<gd_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); \
            else native_down_amd_kernel<T, 2><<<gd_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
    // the AMD layouts cover the IQ packs' down types; the others (Unsloth's Q4_K / Q5_K / Q5_1 / Q8_0) take the
    // CUDA layout below
    if ((em == 1 || em == 2 || em == 4) && (L.d_type == 20 || L.d_type == 23 || L.d_type == 42)) {
        switch (L.d_type) {
            STRATA_D_AMD(20) STRATA_D_AMD(23) STRATA_D_AMD(42)
        }
    } else
#undef STRATA_D_AMD
#endif
    switch (L.d_type) {
#define STRATA_DOWN(T) case T: launch_down<T>(gd, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        STRATA_D_FMTS(STRATA_DOWN)
#undef STRATA_DOWN
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
    }
}


// A770 port (STRATA_PF_XMX=2): a group of experts' products in ONE launch (iq_kernels.dp.cpp, xmxg::grouped_kernel).
namespace xmxg {
template <class C>
void launch(ze::Stream* q, int ty, bool gu, const uint16_t* X, float* Y, const Grp& gp, int ntiles, int N, int K) {
    constexpr int BM = C::BM, BN = C::BN, KS = C::KS, WG = C::WG;
    constexpr int LDA = KS + 8, LDB = KS + 8;
    const int nb = N / BN;
#ifndef STRATA_NO_DG2_XMX
    IqXmxg<C>::k(q, {(unsigned) nb, (unsigned) ntiles, 1}, {(unsigned) WG, 1, 1}, (size_t) (BM * LDA + BN * LDB) * sizeof(uint16_t),
                 ty, gu, X, Y, gp, N, K);
#else
    (void) q; (void) X; (void) Y; (void) gp; (void) ntiles; (void) N; (void) K;
    std::fprintf(stderr, "grouped XMX: the 8x8x16 / sub-group 8 kernel is not built for this device (STRATA_NO_DG2_XMX)\n");
    std::exit(1);
#endif
}
template <class C>
int fill(Grp& gp, const int32_t* cnt, int n) {
    int t = 0, o = 0;
    for (int i = 0; i < n; ++i) { gp.off[i] = o; gp.tstart[i] = t; o += cnt[i]; t += (cnt[i] + C::BM - 1) / C::BM; }
    gp.off[n] = o; gp.tstart[n] = t; gp.n = n;
    return t;
}
}  // namespace xmxg

void iq_xmx_grouped(int ty, const uint8_t* const* a, const uint8_t* const* b, const int32_t* cnt, int n,
                    const uint16_t* X, float* Y, int N, int K, void* stream) {
    if (n <= 0) return;
    if (n > kXmxGroupMax || N % 128 != 0 || K % 32 != 0 || !is_iq(ty)) {
        std::fprintf(stderr, "iq_xmx_grouped: bad arguments (type %d, n %d, N %d, K %d)\n", ty, n, N, K);
        std::exit(1);
    }
    xmxg::Grp gp{};
    int64_t tot = 0;
    for (int i = 0; i < n; ++i) { gp.a[i] = a[i]; gp.b[i] = b ? b[i] : nullptr; tot += cnt[i]; }
    if (tot <= 0) return;
    ze::Stream* q = ze::stream_of(stream);
    const bool gu = b != nullptr;
    if (tot / n < 96) {
        const int nt = xmxg::fill<xmxg::Small>(gp, cnt, n);
        xmxg::launch<xmxg::Small>(q, ty, gu, X, Y, gp, nt, N, K);
    } else {
        const int nt = xmxg::fill<xmxg::Big>(gp, cnt, n);
        xmxg::launch<xmxg::Big>(q, ty, gu, X, Y, gp, nt, N, K);
    }
}


// SYCL port: XMX GEMM straight from the quantized rows (iq_kernels.dp.cpp, xmx::gemm_kernel).
bool xmx_gemm_iq(int ty, const void* gate, const void* up, int64_t K, int n_out, const uint16_t* X, int M, float* Y,
                 int64_t ldy, void* stream) {
    if (!is_iq(ty) || ty == 16 || ty == 17 || ty == 29 || ty == 23 || ty == 11) return false;   // dq_dispatch covers these; the rest untested here
    if (M <= 0 || n_out <= 0 || n_out % xmx::NT != 0 || K <= 0) return false;
    const bool by_block = (K % 256) == 0;
    const bool whole = !by_block && ((int64_t) xmx::NT * K) % 256 == 0 && K % 16 == 0;
    if (!by_block && !whole) return false;
    const int slab_k = by_block ? 256 : (int) K;
    if (up != nullptr && !by_block) return false;   // gate/up rows are per_row blocks each
    const size_t stage_halves = (size_t) xmx::NT * slab_k;
    const size_t bytes = stage_halves * 2 /*stage*/ + stage_halves * 2 /*packed*/ + (size_t) xmx::NSG * 16 * (16 + xmx::APAD) * 2 +
                         (size_t) xmx::NSG * 256 * 4;
    if (bytes > 96 * 1024) return false;
    ze::Stream* q = ze::stream_of(stream);
    xmx::Src src{(const uint8_t*) gate, (const uint8_t*) up, ty, by_block ? K / 256 : 0};
    const int groups = n_out / xmx::NT;
    k_iq_kernels_xmx_gemm(q, {(unsigned) groups, 1, 1}, {(unsigned) xmx::WG, 1, 1}, bytes, src, X, (int) K, slab_k, Y, ldy, M,
                          n_out, stage_halves);
    return true;
}
}  // namespace strata::kernels
