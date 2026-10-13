// include/strata/gemmstone.hpp - Strata's dense products on gemmstone, oneDNN's GEMM kernel generator (vendored in
// third_party/gemmstone with nGEN in third_party/ngen): kernels are generated on the host as native Xe2 binaries the
// first time a case is seen and run through strata::ze.
//
// Lifetimes: gemmstone and nGEN only ever receive a problem description and a hardware description built from
// strata::ze::device_info. They return a byte image and keep nothing, and no GPU runtime handle is ever passed to them.
// strata::ze loads each image and owns its module and kernel until the process exits. A launch borrows its buffers:
// they must stay allocated until the stream has run past it (strata::ze::free waits for the device first).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::gs {

/// The weight formats of gemm().
enum class Weights : int {
    F16 = 0,    ///< W [N, K] fp16, row stride ldw elements; X fp16
    BF16 = 1,   ///< W [N, K] bf16, row stride ldw elements; X bf16
    Q4 = 2,     ///< W [N, K] signed 4-bit, two per byte (even k in the low nibble), row stride ldw elements; X fp16;
                ///< `scales` [K / 32, N] fp16 (one per 32 k of each row), as q4_repack writes them
};

/// Y[T, N] (fp32, row stride ldy) = X[T, K] (row stride ldx) . W^T, added to Y when `accumulate`, on `stream` (a
/// strata::ze::Stream*, nullptr: the current GPU's default stream). Returns false, having appended nothing, when gemmstone
/// has no kernel for the case (the caller falls back to another path).
bool gemm(Weights w, const uint16_t* X, int64_t ldx, const void* W, int64_t ldw, const uint16_t* scales, float* Y,
          int64_t ldy, int64_t T, int64_t N, int64_t K, bool accumulate, void* stream);

/// True when gemmstone generates kernels for GPU `dev` (Xe2); gemm() returns false on any other.
bool supported(int dev);

/// Generates, without any GPU, the kernel gemm() would choose for a case on an Xe2 discrete GPU with `eu_count` EUs and
/// 64-byte aligned operands, for offline checks. Returns its native binary (empty when there is none) and describes the
/// chosen strategy in `what`.
std::vector<uint8_t> probe(Weights w, int64_t T, int64_t N, int64_t K, bool accumulate, int eu_count, std::string& what);

}  // namespace strata::gs
