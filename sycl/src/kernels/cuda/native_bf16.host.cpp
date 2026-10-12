// src/kernels/cuda/native_bf16.host.cpp - the host side of native_bf16.dp.cpp: block-size choice and the launchers.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <cmath>

namespace strata::kernels {

#include "native_bf16.kern.inc"

namespace {

using RowsKernel = decltype(k_native_bf16_rows_32_4);
using RowsAuxKernel = decltype(k_native_bf16_rows_aux_32_4);

// The rows entry points by block size, activation rows (4 or 8) and STRATA_TSUM.
template <int B, int NT, bool TS> constexpr const RowsKernel* rows_kernel = nullptr;
template <int B, int NT, bool TS> constexpr const RowsAuxKernel* rows_aux_kernel = nullptr;
#define NBF16_ROWS_K(B, NT)                                                                                         \
    template <> constexpr const RowsKernel* rows_kernel<B, NT, false> = &k_native_bf16_rows_##B##_##NT;            \
    template <> constexpr const RowsKernel* rows_kernel<B, NT, true> = &k_native_bf16_rows_ts_##B##_##NT;           \
    template <> constexpr const RowsAuxKernel* rows_aux_kernel<B, NT, false> = &k_native_bf16_rows_aux_##B##_##NT; \
    template <> constexpr const RowsAuxKernel* rows_aux_kernel<B, NT, true> = &k_native_bf16_rows_aux_ts_##B##_##NT;
#define NBF16_ROWS_B(B) NBF16_ROWS_K(B, 4) NBF16_ROWS_K(B, 8)
NBF16_ROWS_B(32) NBF16_ROWS_B(64) NBF16_ROWS_B(96) NBF16_ROWS_B(128)
NBF16_ROWS_B(160) NBF16_ROWS_B(192) NBF16_ROWS_B(224) NBF16_ROWS_B(256)
#undef NBF16_ROWS_B
#undef NBF16_ROWS_K

template <int B>
void launch_rows(const float *x, int64_t ldx, const uint16_t *w, float *y,
                 int64_t ldy, int n_in, int n_out, int n_tok,
                 strata::ze::Stream* st, const uint16_t *w_aux = nullptr,
                 float *y_aux = nullptr, int64_t ldy_aux = 0) {
    constexpr int RPB = 4;
    const unsigned nb = (unsigned) ((n_out + RPB - 1) / RPB);
    static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    if (w_aux != nullptr) {   // S26 STRATA_LFUSE: + one aux row block
        const RowsAuxKernel* k = n_tok <= 4 ? (ts ? rows_aux_kernel<B, 4, true> : rows_aux_kernel<B, 4, false>)
                                            : (ts ? rows_aux_kernel<B, 8, true> : rows_aux_kernel<B, 8, false>);
        (*k)(st, {nb + 1, 1, 1}, {B, 1, 1}, 0, x, ldx, w, y, ldy, n_in, n_out, n_tok, w_aux, y_aux, ldy_aux);
        return;
    }
    const RowsKernel* k = n_tok <= 4 ? (ts ? rows_kernel<B, 4, true> : rows_kernel<B, 4, false>)
                                     : (ts ? rows_kernel<B, 8, true> : rows_kernel<B, 8, false>);
    (*k)(st, {nb, 1, 1}, {B, 1, 1}, 0, x, ldx, w, y, ldy, n_in, n_out, n_tok);
}

int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

}  // namespace

bool bf16_gemv_fp32_mmvf_multi_aux(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                                   int64_t n_in, int64_t n_out, int n_tok, const uint16_t* w_aux, float* y_aux,
                                   int64_t ldy_aux, void* stream) {
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    // exactly the conditions under which both calls would take the rows / multi kernels at the same block size
    if (!rows || n_out < 64 || n_tok < 2 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || w_aux == nullptr || y_aux == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        return false;
    strata::ze::Stream* const st = strata::ze::stream_of(stream);
    const int ni = (int) n_in, no = (int) n_out;
    switch (mmvf_block_size(n_in)) {
        case 32: launch_rows<32>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 64: launch_rows<64>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 96: launch_rows<96>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 128: launch_rows<128>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 160: launch_rows<160>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 192: launch_rows<192>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        case 224: launch_rows<224>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
        default: launch_rows<256>(x, ldx, w, y, ldy, ni, no, n_tok, st, w_aux, y_aux, ldy_aux); break;
    }
    return true;
}

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) { bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream); return; }
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    strata::ze::Stream* const st = strata::ze::stream_of(stream);
    static const bool rows = [] { const char* v = std::getenv("STRATA_MMVF_ROWS"); return v && v[0] == '1'; }();
    if (rows && n_out >= 64) {
        const int ni = (int) n_in, no = (int) n_out;
        switch (mmvf_block_size(n_in)) {
            case 32: launch_rows<32>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 64: launch_rows<64>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 96: launch_rows<96>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 128: launch_rows<128>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 160: launch_rows<160>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 192: launch_rows<192>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            case 224: launch_rows<224>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
            default: launch_rows<256>(x, ldx, w, y, ldy, ni, no, n_tok, st); break;
        }
        return;
    }
    const strata::ze::Dim3 grid{(unsigned) n_out, 1, 1};
#define STRATA_MMVF_M(N)                                                                                            \
    case N:                                                                                                         \
        switch (n_tok) {                                                                                            \
        case 1: k_native_bf16_multi_##N##_1(st, grid, {N, 1, 1}, 0, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 2: k_native_bf16_multi_##N##_2(st, grid, {N, 1, 1}, 0, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 3: k_native_bf16_multi_##N##_3(st, grid, {N, 1, 1}, 0, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 4: k_native_bf16_multi_##N##_4(st, grid, {N, 1, 1}, 0, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 5: k_native_bf16_multi_##N##_5(st, grid, {N, 1, 1}, 0, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        case 6: k_native_bf16_multi_##N##_6(st, grid, {N, 1, 1}, 0, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        default: k_native_bf16_multi_##N##_8(st, grid, {N, 1, 1}, 0, x, ldx, w, y, ldy, (int) n_in, n_tok); break; \
        } break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_M(32); STRATA_MMVF_M(64); STRATA_MMVF_M(96); STRATA_MMVF_M(128);
        STRATA_MMVF_M(160); STRATA_MMVF_M(192); STRATA_MMVF_M(224); STRATA_MMVF_M(256);
    }
#undef STRATA_MMVF_M
}

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    strata::ze::Stream* const st = strata::ze::stream_of(stream);
#define STRATA_MMVF_CASE(N) \
    case N: k_native_bf16_mmvf_##N(st, {(unsigned) n_out, 1, 1}, {N, 1, 1}, 0, x, w, y, (int) n_in); break
    switch (mmvf_block_size(n_in)) {
        STRATA_MMVF_CASE(32);
        STRATA_MMVF_CASE(64);
        STRATA_MMVF_CASE(96);
        STRATA_MMVF_CASE(128);
        STRATA_MMVF_CASE(160);
        STRATA_MMVF_CASE(192);
        STRATA_MMVF_CASE(224);
        STRATA_MMVF_CASE(256);
    }
#undef STRATA_MMVF_CASE
}



void bf16_gemv_fp32_mmvf_cols(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int ncols,
                              void* stream) {
    // columns contiguous: the existing multi-row kernel (weight read once, each output bitwise its one-row call),
    // up to 8 rows per launch
    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = ncols - c0 < 8 ? ncols - c0 : 8;
        if (nc == 1) bf16_gemv_fp32_mmvf(x + (size_t) c0 * n_in, w, y + (size_t) c0 * n_out, n_in, n_out, stream);
        else bf16_gemv_fp32_mmvf_multi(x + (size_t) c0 * n_in, n_in, w, y + (size_t) c0 * n_out, n_out, n_in, n_out, nc,
                                       stream);
    }
}

}  // namespace strata::kernels
