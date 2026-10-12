// src/kernels/cuda/elementwise.host.cpp - the host side of elementwise.dp.cpp: the launchers and the doorbell's host checks.
#include "strata/sycl_doorbell.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <bit>
#include <chrono>

namespace strata::kernels {

struct MappedCopies {
    MappedCopy c[8];
};

#include "elementwise.kern.inc"

namespace {

constexpr int THREADS = 256;

/// A bounded launch: a zero-length grid is illegal, and `n == 0` is a real call (an empty selection).
inline unsigned grid_for(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

void sync_if_needed(void *stream, const char *what) {
    if (stream != nullptr) return;
    ze::device_sync();
}

bool check_launch(const char* what) {
    return true;
}

}  // namespace

void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets,
                      int64_t n, int code_bits, int code_bias, int group_elems,
                      float* out, void* stream) {
    if (n <= 0) return;
    k_elementwise_embedding_gather(ze::stream_of(stream), {grid_for(n), 1, 1}, {THREADS, 1, 1}, 0, codes, scales, offsets,
                                   n, code_bits, code_bias, group_elems, out);
    check_launch("embedding_gather");
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream) {
    if (n_tokens <= 0 || h_v <= 0) return;
    const int64_t n = n_tokens * h_v;
    k_elementwise_gdn_gate(ze::stream_of(stream), {grid_for(n), 1, 1}, {THREADS, 1, 1}, 0, alpha, dt, ssm_a, gate, h_v);
    check_launch("gdn_gate");
    sync_if_needed(stream, "gdn_gate");
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    k_elementwise_scale(ze::stream_of(stream), {grid_for(n), 1, 1}, {THREADS, 1, 1}, 0, x, n, s);
    check_launch("scale_inplace");
    sync_if_needed(stream, "scale_inplace");
}

void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    k_elementwise_add(ze::stream_of(stream), {grid_for(n), 1, 1}, {THREADS, 1, 1}, 0, dst, src, n);
    check_launch("add_inplace");
    sync_if_needed(stream, "add_inplace");
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    k_elementwise_to_f16(ze::stream_of(stream), {grid_for(n), 1, 1}, {THREADS, 1, 1}, 0, x, y, n);
    check_launch("f32_to_f16_bulk");
    sync_if_needed(stream, "f32_to_f16_bulk");
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    k_elementwise_to_bf16(ze::stream_of(stream), {grid_for(n), 1, 1}, {THREADS, 1, 1}, 0, x, y, n);
    check_launch("f32_to_bf16_bulk");
    sync_if_needed(stream, "f32_to_bf16_bulk");
}

void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    k_elementwise_silu(ze::stream_of(stream), {grid_for(n), 1, 1}, {THREADS, 1, 1}, 0, x, n);
    check_launch("silu_inplace");
    sync_if_needed(stream, "silu_inplace");
}

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    if (d_flag == nullptr || d_seq == nullptr) return;
    const uint32_t spin_max = strata::spin_max(ze::stream_of(stream)->device());
    k_elementwise_doorbell_wait(ze::stream_of(stream), {1, 1, 1}, {1, 1, 1}, 0, d_flag, d_seq, spin_max);
    check_launch("doorbell_wait");
}

void scatter_rows_f32(const float* src, float* dst, const int32_t* rows, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "scatter_rows_f32: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    k_elementwise_scatter_rows(ze::stream_of(stream), {(unsigned) n, 1, 1}, {128, 1, 1}, 0, src, dst, rows, width / 4);
}
void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* stream) {
    if (rows <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_rows_from_mapped: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    k_elementwise_copy_rows_from_mapped(ze::stream_of(stream), {(unsigned) rows, 1, 1}, {128, 1, 1}, 0, dst, src,
                                        width / 4, hit_rows, count);
}
void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    if ((n & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    const int64_t n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    k_elementwise_copy_from_mapped(ze::stream_of(stream), {(unsigned) blocks, 1, 1}, {256, 1, 1}, 0, dst, src, n4);
    check_launch("copy_from_mapped");
}

namespace {
// The published payload's checksum: the wrapping sum of every 32-bit word mixed with its position (murmur3's
// finalizer).  A plain sum of the words missed a reordering ([3, 7] -> [7, 3] in the expert ids) and cancelling
// changes; mixed, a stale payload passes only with a chance near 2^-32 per read.  It detects; it does not order -
// the host keeps reading until the payload it sees matches (doorbell_payload_ready).  Positions: x 0..n-1,
// ids n..n+k-1, weights n+k..n+2k-1.
inline uint32_t dbx_payload_mix(uint32_t word, uint32_t pos) {
    uint32_t h = word ^ (pos * 0x9E3779B9u);
    h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
    return h;
}
inline uint32_t dbx_payload_sum(const float* x, int64_t n, const int32_t* ids, const float* w, int64_t k,
                                bool has_weights) {
    uint32_t s = 0;
    for (int64_t j = 0; j < n; ++j) s += dbx_payload_mix(std::bit_cast<uint32_t>(x[j]), (uint32_t) j);
    for (int64_t j = 0; j < k; ++j) s += dbx_payload_mix((uint32_t) ids[j], (uint32_t) (n + j));
    if (has_weights)
        for (int64_t j = 0; j < k; ++j) s += dbx_payload_mix(std::bit_cast<uint32_t>(w[j]), (uint32_t) (n + k + j));
    return s;
}
// The payload store (strata::sys_store_mapped(plain, ...)): uncached (sycl_doorbell.hpp) so a host that polls the ring while the kernel still runs reads a
// whole payload, or a plain store when every reader waits for the kernel to end (doorbell_plain_payload: the stepped
// verify window reads a segment's payload only after the segment has drained, and the end of a kernel makes its
// stores visible). The host checksum (doorbell_payload_ready) still guards every read.
std::atomic<bool> g_plain_payload{false};
}  // namespace

void doorbell_plain_payload(bool on) { g_plain_payload.store(on, std::memory_order_relaxed); }

void doorbell_publish_value(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k,
                            float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq, uint32_t value,
                            void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    const bool plain = strata::doorbell_plain_payload() || g_plain_payload.load(std::memory_order_relaxed);
    const bool pair = strata::a770_fast();
    k_elementwise_doorbell_publish_value(ze::stream_of(stream), {1, 1, 1}, {1024, 1, 1}, 0, x, ids, weights, (int) n,
                                         (int) k, x_out, ids_out, weights_out, d_seq, value, plain, pair);
}

void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    const bool plain = strata::doorbell_plain_payload() || g_plain_payload.load(std::memory_order_relaxed);
    const bool pair = strata::a770_fast();
    k_elementwise_doorbell_publish(ze::stream_of(stream), {1, 1, 1}, {1024, 1, 1}, 0, x, ids, weights, (int) n, (int) k,
                                   x_out, ids_out, weights_out, d_seq, plain, pair);
    check_launch("doorbell_publish");
}

void doorbell_publish_res(const float* x, const int32_t* ids, const int32_t* d_res, int n_expert, int64_t n, int64_t k,
                          float* x_out, int32_t* ids_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish_res: k too large\n"); std::exit(1); }
    static const bool full = [] { const char* v = std::getenv("STRATA_DOORBELL_CHECK"); return v != nullptr && v[0] == '1'; }();
    const bool fl = full;
    const bool plain = strata::doorbell_plain_payload() || g_plain_payload.load(std::memory_order_relaxed);
    const bool pair = strata::a770_fast();
    k_elementwise_doorbell_publish_res(ze::stream_of(stream), {1, 1, 1}, {1024, 1, 1}, 0, x, ids, d_res, n_expert, (int) n,
                                       (int) k, x_out, ids_out, d_seq, fl, plain, pair);
    check_launch("doorbell_publish_res");
}

bool doorbell_payload_ready(const uint32_t* h_seq, const float* h_x, int64_t n, const int32_t* h_ids,
                            const float* h_weights, int64_t k, uint32_t want) {
    const volatile uint32_t* w = h_seq + 1 + 2 * (want % 4);   // this ring's tag and checksum (elementwise.hpp)
    if (w[0] != want) return false;
    std::atomic_thread_fence(std::memory_order_acquire);
    // the two publish shapes: with the routing weights (doorbell_publish / _value) or without (doorbell_publish_res)
    return dbx_payload_sum(h_x, n, h_ids, h_weights, k, true) == w[1] ||
           dbx_payload_sum(h_x, n, h_ids, h_weights, k, false) == w[1];
}

bool doorbell_wait_payload(const uint32_t* h_seq, const float* h_x, int64_t n, const int32_t* h_ids,
                           const float* h_weights, int64_t k, uint32_t want, int timeout_ms) {
    // Opt-in (STRATA_DOORBELL_CHECK=1): on an Arc A750 (i915) the host-side checksum of the whole payload, read from
    // uncached host memory twice per CPU-expert layer, cost 15-20% of decode (5 interleaved rounds), and the stale
    // payload it guards against was not seen there. The ring tag is still published by the GPU.
    static const bool check = [] { const char* v = std::getenv("STRATA_DOORBELL_CHECK"); return v != nullptr && v[0] == '1'; }();
    if (!check) return true;
    if (doorbell_payload_ready(h_seq, h_x, n, h_ids, h_weights, k, want)) return true;
    const auto t0 = std::chrono::steady_clock::now();
    while (!doorbell_payload_ready(h_seq, h_x, n, h_ids, h_weights, k, want)) {
        for (int i = 0; i < 64; ++i) __builtin_ia32_pause();
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) return false;
    }
    return true;
}

#if !defined(__HIPCC__)
void copy_from_mapped_multi(const MappedCopy* copies, int n, void* stream) {
    if (n <= 0) return;
    if (n > 8) { std::fprintf(stderr, "copy_from_mapped_multi: at most 8 copies\n"); std::exit(1); }
    MappedCopies a;
    int64_t most = 0;
    for (int i = 0; i < n; ++i) {
        a.c[i] = copies[i];
        most = copies[i].words > most ? copies[i].words : most;
    }
    const int64_t units = (most + 3) / 4;
    const unsigned bx = (unsigned) ((units + 255) / 256 < 32 ? (units + 255) / 256 : 32);
    k_elementwise_copy_from_mapped_multi(ze::stream_of(stream), {bx < 1 ? 1 : bx, (unsigned) n, 1}, {256, 1, 1}, 0, a);
    check_launch("copy_from_mapped_multi");
}
#endif

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    k_elementwise_copy_i32_from_mapped(ze::stream_of(stream), {1, 1, 1}, {128, 1, 1}, 0, dst, src, (int) n);
    check_launch("copy_i32_from_mapped");
}

#if defined(__HIPCC__)   // AMD keeps one copy a launch
void copy_from_mapped_multi(const MappedCopy* copies, int n, void* stream) {
    for (int i = 0; i < n; ++i)
        copy_i32_from_mapped((int32_t*) copies[i].dst, (const int32_t*) copies[i].src, copies[i].words, stream);
}
#endif

void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;
    k_elementwise_doorbell_ring(ze::stream_of(stream), {1, 1, 1}, {1, 1, 1}, 0, d_seq);
    check_launch("doorbell_ring");
    sync_if_needed(stream, "doorbell_ring");
}

void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    // 4 warps per block, so a row count that is not a multiple of 4 wastes at most 3 warps rather than
    // launching a block per row for a 2-row call.
    const unsigned warps_per_block = 4;
    const unsigned grid = (unsigned) ((rows + warps_per_block - 1) / warps_per_block);
    k_elementwise_rms_norm_weighted(ze::stream_of(stream), {grid, 1, 1}, {warps_per_block * 32, 1, 1}, 0, x, w, rows, cols,
                                    eps);
    check_launch("rms_norm_weighted");
    sync_if_needed(stream, "rms_norm_weighted");
}

}  // namespace strata::kernels
