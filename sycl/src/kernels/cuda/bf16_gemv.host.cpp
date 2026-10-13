// src/kernels/cuda/bf16_gemv.host.cpp - the host side of bf16_gemv.dp.cpp: the launchers.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "bf16_gemv.kern.inc"

namespace {

constexpr int THREADS = 256;

inline void finish(void *stream, const char *what) {
    if (stream != nullptr) return;
    ze::device_sync();
}

}  // namespace

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // **THIS USED TO ALWAYS USE THE NAIVE KERNEL, AND THE NAIVE KERNEL IS UNCOALESCED.**
    //
    // `bf16_gemv_naive_kernel` gives one THREAD per output row and walks the row contiguously, so at a fixed
    // `i` the 32 threads of a warp read `w[(o+k)*n_in + i]` - addresses `n_in * 2` bytes apart.  Every load in
    // the kernel's inner loop costs 32 transactions instead of 1.  The comment above that kernel claims it is
    // "used directly for the shapes where the output width is already large", but nothing ever made that
    // choice: this function called it unconditionally, at every shape.
    //
    // What that costs, from the nsys per-kernel table: **4.66 ms/token across 24 calls, 194 us each** - the
    // third largest kernel in the engine at the time, moving 1.19 GiB/token at ~210 GB/s where a plain read at
    // this geometry measures 641 GB/s (`bench/micro/p32_floor.cu`).
    //
    // `bf16_gemv_warp_kernel` exists, is checked against this one by `bf16_gemv_parity`, and is coalesced -
    // consecutive lanes touch consecutive addresses - which is the property `gr_read`'s comment records as
    // having taken it from 262 ms/token to 8.9.  The threshold below is where warp-per-row has enough rows to
    // fill the machine: 8 warps per block, so 64 rows is 8 blocks and 128 is 16, against 48 SMs.
    if (n_out >= 64) {
        const int warps = THREADS / 32;
        const unsigned grid = (unsigned) ((n_out + warps - 1) / warps);
        k_bf16_gemv_warp(ze::stream_of(stream), {grid, 1, 1}, {THREADS, 1, 1}, 0, x, w, y, n_in, n_out);
        finish(stream, "bf16_gemv(warp)");
        return;
    }
    // Below the threshold the warp kernel would leave most of the machine idle, and the naive one is at least
    // not wasting warps.  It is still uncoalesced, so this is the branch to revisit if a small-output caller
    // ever shows up hot in the profile.
    const unsigned grid = (unsigned) ((n_out + THREADS - 1) / THREADS);
    k_bf16_gemv_naive(ze::stream_of(stream), {grid, 1, 1}, {THREADS, 1, 1}, 0, x, w, y, n_in, n_out);
    finish(stream, "bf16_gemv");
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // WARP-PER-ROW when the caller asks for 32, because that path needs no shared memory and no barrier - and
    // with 48 output rows it is the configuration that fills the machine.  A `tpr` that is not 32 and not a
    // power of two is refused rather than quietly rounded, because the tree reduction below needs one.
    if (threads_per_row == 32) {
        const int warps = THREADS / 32;
        const unsigned grid = (unsigned) ((n_out + warps - 1) / warps);
        k_bf16_gemv_warp(ze::stream_of(stream), {grid, 1, 1}, {THREADS, 1, 1}, 0, x, w, y, n_in, n_out);
        finish(stream, "bf16_gemv_split(warp)");
        return;
    }
    if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0) {
        std::fprintf(stderr, "bf16_gemv_split: threads_per_row %d must be a power of two (32 selects the "
                             "warp-per-row path)\n", threads_per_row);
        std::exit(1);
    }
    const unsigned grid = (unsigned) n_out;
    k_bf16_gemv_split(ze::stream_of(stream), {grid, 1, 1}, {(uint32_t) threads_per_row, 1, 1},
                      (size_t) threads_per_row * sizeof(float), x, w, y, n_in, n_out, threads_per_row);
    finish(stream, "bf16_gemv_split");
}

}  // namespace strata::kernels
