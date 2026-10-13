// src/kernels/cuda/native_gr_norm.host.cpp - the host side of native_gr_norm.dp.cpp: argument checks and the launcher.
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {

#include "native_gr_norm.kern.inc"

namespace {

void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float) != 0)
        throw std::invalid_argument("native GR RMSNorm requires non-null four-byte aligned pointers");
}

} // namespace

void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                 int n_cols, int n_rows, float epsilon, void* stream) {
    native_gr_rms_norm_weighted_multi(input, gamma, output, n_cols, n_rows, 1, epsilon, stream);
}

void native_gr_rms_norm_weighted_multi(const float* input, const float* gamma, float* output,
                                       int n_cols, int n_rows, int n_tok, float epsilon, void* stream) {
    if (n_cols <= 0 || n_rows <= 0 || n_tok <= 0 || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GR RMSNorm requires positive dimensions and finite nonnegative epsilon");
    check_pointer(input);
    check_pointer(gamma);
    check_pointer(output);
    const auto cuda_stream = strata::ze::stream_of(stream);
    const strata::ze::Dim3 grid{unsigned(n_rows), unsigned(n_tok), 1u};
    if (n_cols < 1024)
        k_native_gr_norm_256(cuda_stream, grid, {256, 1, 1}, 0, input, gamma, output, n_cols, epsilon);
    else
        k_native_gr_norm_1024(cuda_stream, grid, {1024, 1, 1}, 0, input, gamma, output, n_cols, epsilon);
}

} // namespace strata::kernels
