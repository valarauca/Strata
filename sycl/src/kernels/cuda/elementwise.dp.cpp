// src/kernels/cuda/elementwise.cu - P2.S5's glue kernels.  See the header for why each exists.
#include "strata/port.hpp"
#include "strata/sycl_doorbell.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/dp4a.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "elementwise_shared.hpp"

namespace strata::kernels {
namespace {

__dpct_inline__ void embedding_gather_kernel(const uint8_t *__restrict__ codes,
                                             const float *__restrict__ scales,
                                             const float *__restrict__ offsets,
                                             int64_t n, int code_bits,
                                             int code_bias, int group_elems,
                                             float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (codes[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = (float)(code + code_bias) * scales[group];
    out[i] = product + (offsets ? offsets[group] : 0.0f);
}

/// `ggml_compute_softplus_f32`: `log1p(exp(x))`, with the large-x branch that avoids overflow.
///
/// The branch is not cosmetic. `exp(89)` overflows f32 and `exp(20)` is already 4.85e8 where `log1p` loses
/// relative precision; above 20 the function is `x` to within f32 anyway.
__dpct_inline__ float softplus_dev(float x) {
    return x > 20.0f ? x : sycl::log1p(sycl::native::exp(x));
}

__dpct_inline__ void gdn_gate_kernel(const float *__restrict__ alpha,
                                     const float *__restrict__ dt,
                                     const float *__restrict__ ssm_a,
                                     float *__restrict__ gate, int64_t h_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= h_v) return;
    const int64_t t = i / h_v;      // `n_tokens` is the leading dim; the real call has one token
    const int64_t h = i % h_v;
    gate[i] = softplus_dev(alpha[i] + dt[h]) * ssm_a[h];
    (void) t;
}

__dpct_inline__ void scale_kernel(float *__restrict__ x, int64_t n, float s) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) x[i] *= s;
}

// `dst[i] += src[i]`.  See the header: this is what lets R4's GPU half and CPU half run at the same
// time and still add up to one `parts` buffer.
__dpct_inline__ void add_kernel(float *__restrict__ dst,
                                const float *__restrict__ src, long long n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) dst[i] += src[i];
}

__dpct_inline__ void to_f16_kernel(const float *__restrict__ x,
                                   uint16_t *__restrict__ y, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) y[i] = f16_from_f32(x[i]);
}

__dpct_inline__ void to_bf16_kernel(const float *__restrict__ x,
                                    uint16_t *__restrict__ y, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) y[i] = bf16_from_f32(x[i]);
}

__dpct_inline__ void silu_kernel(float *__restrict__ x, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    // DOUBLE then cast, matching `ref/gdn.py`'s numpy: its arrays are f32 but `np.exp` on an f32 array is
    // computed to f32 precision by a different algorithm than `expf`, and the reference is the oracle.  The
    // difference is in the last bits and this is one line.
    const double v = (double) x[i];
    x[i] = (float)(v / (1.0 + sycl::exp(-v)));
}

/// One WARP per row, reduced through shuffles.  The rows here are short and few - (24, 256), (2, 256),
/// (4, 128) - so a block-per-row tree would spend its time in `__syncthreads` for 8 warps of 32, and the whole
/// call is 30 rows.  A warp reduction with NO shared memory and NO barrier is the shape that fits.
__dpct_inline__ void rms_norm_weighted_kernel(float *__restrict__ x,
                                              const float *__restrict__ w,
                                              int64_t rows, int64_t cols,
                                              float eps) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31;
    const int64_t row =
        ((int64_t)item_ct1.get_group(2) * (item_ct1.get_local_range(2) >> 5)) +
        (item_ct1.get_local_id(2) >> 5);
    // **THE ROW GUARD IS NOT DECORATION, AND ITS ABSENCE WAS THE QSA BUG.**  The launcher rounds the grid up to
    // whole 4-warp blocks, so a call with `rows` = 2 - the QSA k-norm, the only non-multiple-of-4 row count in
    // the engine - runs EIGHT warps: rows 0 and 1 are the data, and rows 2 and 3 write 2*cols floats PAST the
    // end of `b.kcur`, which in the arena is exactly where `b.vcur` begins.  `b.vcur` was therefore silently
    // replaced by two rows of `rms_norm(...) * attn_k_norm`: a normalized vector with l2 ~20.6 against V's
    // ~5.7, which the attention then attended to.  `rope_neox_kernel` has had this guard all along.
    if (row >= rows) return;
    float* r = x + row * cols;
    float acc = 0.0f;
#pragma unroll
    for (int64_t c = lane; c < cols; c += 32) acc += r[c] * r[c];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc +=
        dpct::experimental::shift_sub_group_left(
            0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            acc, off);
    // The MEAN, not the sum: `ref/qsa.py::rms_norm` divides by `np.mean(np.square(x))`.  Broadcasting the
    // reciprocal from lane 0 keeps all 32 lanes on the same value - computing `rsqrt` per lane would be the
    // same number but a needless 32-way divergence in the last bit.
    float inv = 0.0f;
    if (lane == 0) inv = sycl::rsqrt(acc / (float)cols + eps);
    inv = dpct::experimental::select_from_sub_group(
        0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), inv,
        0);
#pragma unroll
    for (int64_t c = lane; c < cols; c += 32)
        r[c] = (w ? r[c] * w[c] : r[c]) * inv;
}

}  // namespace

/// THE DOORBELL.  One thread, one INCREMENT - the cost is the launch, and inside a graph that is paid once.
///
/// **IT INCREMENTS THE MEMORY, AND IT DOES NOT TAKE THE VALUE AS AN ARGUMENT.**  The first version did -
/// doorbell_ring(d_seq, *(h_seq) + 1u) - and that is a CLONED LITERAL: the expression is evaluated on the HOST
/// at CAPTURE time, so the graph rings 1 on every replay and a host waiting for a change waits forever.  It
/// would have looked like a working doorbell on the first token, which is the worst way for it to be wrong.
/// **THE FENCE IS NOT DECORATION, AND ITS ABSENCE WAS COSTING ~10 ms PER TOKEN ON THE HOST SIDE.**
///
/// The ring PUBLISHES three buffers the host is about to read - `h_x_f`, `h_ids`, `h_weights` - so the increment
/// must be ordered after their writes, or the host can observe the ring and then read a payload that has not
/// landed.  `__threadfence_system()` is what orders them, and it covers the HOST as well as the device, which is
/// the whole point: the reader is the CPU.
///
/// Without it the host loop was compensating with a driver call.  `session_loop` polled with `cudaEventQuery`
/// on EVERY spin iteration, because round 195 had measured that a memory-only spin never saw the datum flip -
/// and that measurement was right about the symptom and wrong about the cause.  The cause is this missing fence:
/// the write was not ordered into host-visible memory, so no amount of reading it would show it, and the driver
/// call was flushing the whole pipeline enough to make it appear.  A 10-22 us driver call per iteration is a
/// very expensive substitute for one fence instruction.
///
/// **THE STORE IS VOLATILE**, like `doorbell_publish_kernel`'s.  On RDNA4 (gfx1201) a plain store to mapped pinned
/// memory stays in the GPU's L2 until the stream is synchronized - the host never saw the ring (tests/hip/handoff:
/// 0 of 100 rings seen without a sync; volatile, a system-scope atomic store or a fence after the store: 100 of 100).
__dpct_inline__ void doorbell_ring_kernel(uint32_t *seq) {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    strata::sys_store(seq, strata::sys_load(seq) + 1u);
#if defined(__HIPCC__)  // #697: HIP only; on RDNA4 the volatile store alone can sit in L2 until the stream syncs
    __threadfence_system();
#endif
}

__dpct_inline__ void doorbell_wait_kernel(const volatile uint32_t *flag,
                                          const volatile uint32_t *seq, uint32_t spin_max) {
    const uint32_t want = strata::sys_load(seq);
    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(flag) != want; ++spin) strata_spin_pause();
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

__dpct_inline__ void copy_from_mapped_kernel(sycl::float4 *__restrict__ dst,
                                             const sycl::float4 *src,
                                             int64_t n4) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n4; i += (int64_t)item_ct1.get_group_range(2) *
                      item_ct1.get_local_range(2)) {
        const sycl::float4 v = strata::load_mapped_float4(src + i);
        dst[i] = v;
    }
}

// the CPU rows of a verify window, skipping the rows the GPU plan computes itself (the
// pool writes +0.0 into those, so this writes +0.0 too): block = row, the plan's hit rows `dst[0, *count)`.
__dpct_inline__ void copy_rows_from_mapped_kernel(
    sycl::float4 *__restrict__ dst, const sycl::float4 *src, int64_t row4,
    const int32_t *__restrict__ hit_rows, const int32_t *__restrict__ count) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2);
    auto &hit = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) {
        int h = 0;
        const int c = *count;
#pragma unroll
        for (int i = 0; i < c; ++i) h |= hit_rows[i] == row;
        hit = h;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    sycl::float4 *d = dst + (int64_t)row * row4;
    if (hit) {
#pragma unroll
        for (int64_t i = item_ct1.get_local_id(2); i < row4;
             i += item_ct1.get_local_range(2))
            d[i] = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        const sycl::float4 *sr = src + (int64_t)row * row4;
#pragma unroll
        for (int64_t i = item_ct1.get_local_id(2); i < row4;
             i += item_ct1.get_local_range(2))
            d[i] = strata::load_mapped_float4(sr + i);
    }
}
namespace {
__dpct_inline__ void scatter_rows_kernel(const sycl::float4 *__restrict__ src,
                                         sycl::float4 *dst,
                                         const int32_t *__restrict__ rows,
                                         int64_t w4) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t r = item_ct1.get_group(2);
    const sycl::float4 *s = src + r * w4;
    sycl::float4 *d = dst + (int64_t)rows[r] * w4;
#pragma unroll
    for (int64_t i = item_ct1.get_local_id(2); i < w4;
         i += item_ct1.get_local_range(2)) d[i] = s[i];
}
}  // namespace

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
// STRATA_SYCL_A770_FAST=1 (opt-in): pair the activation stores without changing the checksum's word/position
// order. Odd sizes or unaligned mapped pointers retain the scalar path.
__dpct_inline__ uint32_t dbx_publish_x(const float* x, float* out, int n, int lane, int threads, bool pair, bool plain) {
    uint32_t part = 0;
    if (pair && !plain && ((uintptr_t)out & 7u) == 0) {
        for (int j = 2 * lane; j + 1 < n; j += 2 * threads) {
            const uint32_t a = sycl::bit_cast<uint32_t>(x[j]);
            const uint32_t b = sycl::bit_cast<uint32_t>(x[j + 1]);
            strata::sys_store_mapped(reinterpret_cast<uint64_t*>(out + j),
                                     (uint64_t)a | ((uint64_t)b << 32));
            part += dbx_payload_mix(a, (uint32_t)j) + dbx_payload_mix(b, (uint32_t)(j + 1));
        }
        if ((n & 1) && lane == 0) {
            strata::sys_store_mapped(plain, out + n - 1, x[n - 1]);
            part += dbx_payload_mix(sycl::bit_cast<uint32_t>(x[n - 1]), (uint32_t)(n - 1));
        }
    } else {
        for (int j = lane; j < n; j += threads) {
            strata::sys_store_mapped(plain, out + j, x[j]);
            part += dbx_payload_mix(sycl::bit_cast<uint32_t>(x[j]), (uint32_t)j);
        }
    }
    return part;
}
// The ring, its tag and its checksum, in slot r % 4 of the sequence word (elementwise.hpp).  Written last, after a
// system fence over the uncached payload stores.
__dpct_inline__ void dbx_ring_commit(uint32_t* seq, uint32_t part, bool group_ok, uint32_t ring) {
#if defined(__HIPCC__)  // #697: HIP only; on RDNA4 the volatile store alone can sit in L2 until the stream syncs
    (void) group_ok;
#endif
    if (group_ok) {
        strata::sys_store(seq + 2 + 2 * (ring % 4), part);
        strata::sys_store(seq + 1 + 2 * (ring % 4), ring);
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        strata::sys_store(seq, ring);
    }
}
}  // namespace

__dpct_inline__ void doorbell_publish_kernel(const float *__restrict__ x,
                                             const int32_t *__restrict__ ids,
                                             const float *__restrict__ w, int n,
                                             int k, float *x_out,
                                             int32_t *ids_out, float *w_out,
                                             uint32_t *seq, bool plain, bool pair) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = (int) item_ct1.get_local_id(2);
    const int nt = (int) item_ct1.get_local_range(2);
    uint32_t part = dbx_publish_x(x, x_out, n, i, nt, pair, plain);
    if (i < k) {
        const int32_t id = ids[i];
        strata::sys_store_mapped(plain, ids_out + i, id);
        part += dbx_payload_mix((uint32_t) id, (uint32_t) (n + i));
        if (w_out != nullptr) {
            const float wv = w[i];
            strata::sys_store_mapped(plain, w_out + i, wv);
            part += dbx_payload_mix(sycl::bit_cast<uint32_t>(wv), (uint32_t) (n + k + i));
        }
    }
    const uint32_t sum = sycl::reduce_over_group(item_ct1.get_group(), part, sycl::plus<uint32_t>());
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    item_ct1.barrier();
    if (i == 0) {
        const uint32_t next = strata::sys_load(seq) + 1u;
        dbx_ring_commit(seq, sum, true, next);
    }
}

__dpct_inline__ void doorbell_publish_res_kernel(
    const float *__restrict__ x, const int32_t *__restrict__ ids,
    const int32_t *__restrict__ d_res, int n_expert, int n, int k, float *x_out,
    int32_t *ids_out, uint32_t *seq, bool full, bool plain, bool pair) {
    // full (STRATA_DOORBELL_CHECK=1): the payload is published whole and the checksum covers x and the ids (this
    // variant publishes no weights). Otherwise d_res/n_expert decide whether x is copied, as before the checksum:
    // when every routed expert is resident the host never reads x, and an uncached copy of it costs the A750 ~10%
    // of decode.
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (!full) {
        int any_miss = 0;
        if ((int) item_ct1.get_local_id(2) < k) {
            const int32_t id = ids[item_ct1.get_local_id(2)];
            strata::sys_store_mapped(plain, ids_out + item_ct1.get_local_id(2), id);
            if (d_res == nullptr || id < 0 || id >= n_expert || d_res[id] < 0) any_miss = 1;
        }
        if ((item_ct1.barrier(), sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_work_group<3>(), any_miss))) {
            for (int j = (int) item_ct1.get_local_id(2); j < n; j += (int) item_ct1.get_local_range(2))
                strata::sys_store_mapped(plain, x_out + j, x[j]);
        }
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) == 0) {
            const uint32_t next = strata::sys_load(seq) + 1u;
            dbx_ring_commit(seq, 0u, true, next);
        }
        return;
    }
    const int i = (int) item_ct1.get_local_id(2);
    const int nt = (int) item_ct1.get_local_range(2);
    uint32_t part = dbx_publish_x(x, x_out, n, i, nt, pair, plain);
    if (i < k) {
        const int32_t id = ids[i];
        strata::sys_store_mapped(plain, ids_out + i, id);
        part += dbx_payload_mix((uint32_t) id, (uint32_t) (n + i));
    }
    const uint32_t sum = sycl::reduce_over_group(item_ct1.get_group(), part, sycl::plus<uint32_t>());
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    item_ct1.barrier();
    if (i == 0) {
        const uint32_t next = strata::sys_load(seq) + 1u;
        dbx_ring_commit(seq, sum, true, next);
    }
}

// #649 (HIP, STRATA_DOORBELL_STORE=1): the same publish, but the ring is STORED (the step's own number, known at
// capture) instead of read-modify-written over PCIe - one store, no read of host memory from the GPU.
__dpct_inline__ void doorbell_publish_value_kernel(
    const float *__restrict__ x, const int32_t *__restrict__ ids,
    const float *__restrict__ w, int n, int k, float *x_out, int32_t *ids_out,
    float *w_out, uint32_t *seq, uint32_t value, bool plain, bool pair) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = (int) item_ct1.get_local_id(2);
    const int nt = (int) item_ct1.get_local_range(2);
    uint32_t part = dbx_publish_x(x, x_out, n, i, nt, pair, plain);
    if (i < k) {
        const int32_t id = ids[i];
        strata::sys_store_mapped(plain, ids_out + i, id);
        part += dbx_payload_mix((uint32_t) id, (uint32_t) (n + i));
        const float wv = w[i];
        strata::sys_store_mapped(plain, w_out + i, wv);
        part += dbx_payload_mix(sycl::bit_cast<uint32_t>(wv), (uint32_t) (n + k + i));
    }
    const uint32_t sum = sycl::reduce_over_group(item_ct1.get_group(), part, sycl::plus<uint32_t>());
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    item_ct1.barrier();
    if (i == 0) dbx_ring_commit(seq, sum, true, value);
}

__dpct_inline__ void copy_i32_from_mapped_kernel(int32_t *__restrict__ dst,
                                                 const volatile int32_t *src,
                                                 int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) dst[i] = src[i];
}

#if !defined(__HIPCC__)
__dpct_inline__ void copy_from_mapped_multi_kernel(MappedCopies a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const MappedCopy c = a.c[item_ct1.get_group(1)];
    const int64_t i0 = (int64_t)item_ct1.get_group(2) *
                           item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2),
                  st = (int64_t)item_ct1.get_group_range(2) *
                       item_ct1.get_local_range(2);
    if ((c.words & 3) == 0 && ((uintptr_t) c.dst & 15) == 0 && ((uintptr_t) c.src & 15) == 0) {
        const sycl::float4 *s = (const sycl::float4 *)c.src;
#pragma unroll
        for (int64_t i = i0; i < c.words / 4; i += st)(
            (sycl::float4 *)c.dst)[i] = const_cast<const sycl::float4 *>(s)[i];
    } else {
        const volatile int32_t* s = (const volatile int32_t*) c.src;
#pragma unroll
        for (int64_t i = i0; i < c.words; i += st)((int32_t *)c.dst)[i] = s[i];
    }
}
#endif

#include "elementwise.kern.inc"

}  // namespace strata::kernels
