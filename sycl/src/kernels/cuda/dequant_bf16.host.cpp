// src/kernels/cuda/dequant_bf16.host.cpp - the host side of dequant_bf16.dp.cpp: block geometry and the launchers.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace strata::kernels {

#include "dequant_bf16.kern.inc"

namespace {

struct H16 { uint16_t v; };

bool geometry(int type, int& block_elems, int& block_bytes) {
    switch (type) {
    case 2: block_elems = 32; block_bytes = 18; return true;
    case 6: block_elems = 32; block_bytes = 22; return true;
    case 7: block_elems = 32; block_bytes = 24; return true;
    case 8: block_elems = 32; block_bytes = 34; return true;
    case 20: block_elems = 32; block_bytes = 18; return true;
    case 11: block_elems = 256; block_bytes = 110; return true;
    case 12: block_elems = 256; block_bytes = 144; return true;
    case 13: block_elems = 256; block_bytes = 176; return true;
    case 14: block_elems = 256; block_bytes = 210; return true;
    case 23: block_elems = 256; block_bytes = 136; return true;
    case 42: block_elems = 64; block_bytes = 18; return true;
    default: return false;
    }
}

template <typename T>
void launch(int type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, T* out, void* stream,
            int64_t ld = 0) {
    int be = 0, bb = 0;
    if (!geometry(type, be, bb) || cols % be != 0 || rows <= 0) {
        std::fprintf(stderr, "dequant: unsupported type %d or shape %lld x %lld\n", type, (long long) rows,
                     (long long) cols);
        std::exit(1);
    }
    const int64_t row_bytes = cols / be * bb, gpr = cols / 32, total = rows * gpr;
    if (ld <= 0) ld = cols;
    const unsigned grid = (unsigned) ((total + 255) / 256);
    const uint8_t* p = (const uint8_t*) blocks;
    ze::Stream* st = ze::stream_of(stream);
#define STRATA_DQ(TY)                                                                                              \
    {                                                                                                              \
        if constexpr (std::is_same_v<T, uint16_t>)                                                                 \
            k_dequant_bf16_bf16_##TY(st, {grid, 1, 1}, {256, 1, 1}, 0, p, row_bytes, row0, rows, gpr, ld, out);    \
        else if constexpr (std::is_same_v<T, H16>)                                                                 \
            k_dequant_bf16_f16_##TY(st, {grid, 1, 1}, {256, 1, 1}, 0, p, row_bytes, row0, rows, gpr, ld,           \
                                    (uint16_t*) out);                                                              \
        else                                                                                                       \
            k_dequant_bf16_f32_##TY(st, {grid, 1, 1}, {256, 1, 1}, 0, p, row_bytes, row0, rows, gpr, ld, out);     \
    } break
    switch (type) {
    case 2: STRATA_DQ(2);
    case 6: STRATA_DQ(6);
    case 7: STRATA_DQ(7);
    case 8: STRATA_DQ(8);
    case 11: STRATA_DQ(11);
    case 12: STRATA_DQ(12);
    case 13: STRATA_DQ(13);
    case 14: STRATA_DQ(14);
    case 20: STRATA_DQ(20);
    case 23: STRATA_DQ(23);
    case 42: STRATA_DQ(42);
    }
#undef STRATA_DQ
}

}  // namespace

bool dequant_bf16_supported(int ggml_type) noexcept {
    int a, b;
    return geometry(ggml_type, a, b);
}

namespace {
// plan v0.3 P6: the i-quant formats (llama.cpp's dequantizers, iq_kernels.cu)
bool iq_only(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }
}  // namespace

void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream) {
    launch<uint16_t>(ggml_type, blocks, row0, rows, cols, out, stream);
}

void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                 void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f16(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<H16>(ggml_type, blocks, row0, rows, cols, reinterpret_cast<H16*>(out), stream);
}

bool dequant_f16_ld(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, int64_t ld,
                    uint16_t* out, void* stream) {
    int be = 0, bb = 0;
    if (iq_only(ggml_type) || !geometry(ggml_type, be, bb) || cols % be != 0 || rows <= 0 || ld < cols || ld % 8 != 0)
        return false;
    launch<H16>(ggml_type, blocks, row0, rows, cols, reinterpret_cast<H16*>(out), stream, ld);
    return true;
}

void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out, void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f32(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<float>(ggml_type, blocks, row0, rows, cols, out, stream);
}

}  // namespace strata::kernels
