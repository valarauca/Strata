// src/kernels/cuda/gr.cu - P2.S2: the gated residual / hyper-connection, `gr_read` and `gr_write`.
//
// See include/strata/kernels/gr.hpp for the semantics, for why the weights are bf16, and for the history of
// this kernel's 134x-off-the-floor first version.  The short form: it was launched `<<<1, 256>>>` so that
// `xn`/`xq` fit in shared memory, which used ONE of 48 SMs and cost 262 ms/token.
//
// THE SHAPE OF THE FIX is that each projection is an ordinary GEMV and is now mapped the way the shapes want:
//
//     norm    hc blocks, one per residual stream            (each needs its own reduction)
//     down    ONE WARP PER OUTPUT ROW, lanes stride the 10240 reduction axis
//     gate    ONE WARP PER OUTPUT ROW, lanes stride the 320 reduction axis
//     mean    a flat elementwise pass over n_embd
//
// WARP-PER-ROW RATHER THAN THREAD-PER-ROW, and that is the whole difference.  A thread walking one row is
// coalesced with ITSELF and 32 transactions away from its warp neighbours: `w_down[k*hc_dim + i]` for fixed k
// is contiguous, but two threads on rows k and k+1 are hc_dim*2 bytes apart.  With the warp on one row, lanes
// i, i+1, i+2 touch consecutive addresses in EITHER orientation - which is why this version needs no
// transposed copies and takes both matrices exactly as the manifest stores them.
//
// Layouts are the MANIFEST's: `hc_attn_down.weight` is [10240, 320] with ne0 = hc*n_embd fast, so it holds
// `w_down[k][i]` with i contiguous - `ref/gr.py`'s (hc_lr, hc*n_embd) row-major.  `hc_attn_up.weight` is
// [320, 10240] so it holds `w_up[i][k]` with k contiguous - `ref/gr.py`'s (hc*n_embd, hc_lr) row-major.  Both
// index with no permutation at all.
#include "strata/port.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/bf16_bits.hpp"

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

/// The bf16 conversions live in `strata/kernels/bf16_bits.hpp`, and this file used to carry its own copies.
/// Two files with a private copy of a conversion whose failure mode is silent wrong bits is one too many -
/// and the reason the shared header exists is that `bf16` and `fp16` are NOT two spellings of one idea, so a
/// routine written for one and reused for the other is wrong in a way that still produces numbers.
///
/// The local names are kept as thin wrappers so the body below is untouched by the swap.
__dpct_inline__ uint16_t f32_to_bf16_bits(float f) { return bf16_from_f32(f); }
__dpct_inline__ float bf16_bits_to_f32(uint16_t h) { return f32_from_bf16(h); }
__dpct_inline__ float activation_f32(float x) { return x; }
__dpct_inline__ float activation_f32(uint16_t x) { return bf16_bits_to_f32(x); }
__dpct_inline__ void store_activation(float *dst, int i, float x) {
    dst[i] = x;
}
__dpct_inline__ void store_activation(uint16_t *dst, int i, float x) {
    dst[i] = f32_to_bf16_bits(x);
}

/// silu and sigmoid exactly as `ref/gr.py` writes them - in FLOAT, not double.
///
/// This is the opposite of `shared_expert.cu`, which uses double, and the difference is not stylistic:
/// `ref/gr.py`'s arrays are float32, so `x / (1.0 + np.exp(-x))` evaluates in float32 under NEP 50, while
/// `ref/moe.py` works in float64.  Matching the reference's PRECISION is part of transcribing it.
__dpct_inline__ float silu_f(float x) {
    return x / (1.0f + sycl::native::exp(-x));
}
__dpct_inline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

__dpct_inline__ double warp_sum(double v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v +=
        dpct::experimental::shift_sub_group_left(
            0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            off);
    return dpct::experimental::select_from_sub_group(
        0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, 0);
}

__dpct_inline__ float warp_sumf(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v +=
        dpct::experimental::shift_sub_group_left(
            0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            off);
    return dpct::experimental::select_from_sub_group(
        0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, 0);
}

/// The FP32 block-wide sum, for the reason the review's G5 states: this is a GeForce part and FP64 runs at a
/// small fraction of FP32, so a `double` reduction turns a bandwidth kernel into a latency kernel.
///
/// The scratch is taken as `double*` and reinterpreted as `float*` rather than changing the callers' shared
/// arrays, because `double scratch[8]` is 64 bytes and so is at least as large as the `float[8]` this needs -
/// the cast cannot overrun, and it keeps the change to one function instead of one per caller.
inline float block_sumf(float v, double *scratch_raw) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    float *scratch = (float *)scratch_raw;
    // The leading barrier is not decoration - see the note on `block_sum` above: the result is read straight out
    // of `scratch[0]` by every thread and a later call reuses the array, so without it a fast thread can
    // overwrite `scratch[0]` before a slow one has read the previous result.
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    v = warp_sumf(v);
    if (lane == 0) scratch[warp] = v;
    item_ct1.barrier();
    const int nw = ((int)item_ct1.get_local_range(2) + 31) >> 5;
    v = (item_ct1.get_local_id(2) < nw) ? scratch[item_ct1.get_local_id(2)]
                                        : 0.0f;
    if (warp == 0) v = warp_sumf(v);
    if (item_ct1.get_local_id(2) == 0) scratch[0] = v;
    item_ct1.barrier();
    return scratch[0];
}

/// Block-wide sum in DOUBLE, broadcast to every thread.
///
/// Double because `ref/gr.py` reduces in float64 for the norm and the bottleneck, and because a block-wide
/// f32 sum of 2560 squares is a different number from the reference's; the shuffle tree here also reorders
/// the additions, so the only defence is enough precision that the order stops mattering.
///
/// The leading `__syncthreads()` is not decoration: the result is read straight out of `scratch[0]` by every
/// thread, and a later call reuses the same array.  Without a barrier on entry a fast thread can overwrite
/// `scratch[0]` before a slow one has read the previous result - a race that is invisible in most runs.
double block_sum(double v, double* scratch) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    item_ct1.barrier();
    const int nw = ((int)item_ct1.get_local_range(2) + 31) >> 5;
    v = (item_ct1.get_local_id(2) < nw) ? scratch[item_ct1.get_local_id(2)]
                                        : 0.0;
    if (warp == 0) v = warp_sum(v);
    if (item_ct1.get_local_id(2) == 0) scratch[0] = v;
    item_ct1.barrier();
    return scratch[0];
}

/// `hc` blocks, one per residual stream: per-stream RMSNorm, then the activated stream in both forms.
///
/// ggml_rms_norm reduces over ne[0], and ne[0] is n_embd, so the reduction is PER STREAM.  The `[n_embd, hc]`
/// gamma then scales each stream individually.
///
/// A per-thread accumulator is only a sum when the block is narrower than the row, which is why this is a
/// block reduction: with `n_embd` equal to the thread count every thread would hold exactly one element and
/// normalise by its own square.  That mistake is silent - it produces a plausible vector of the right shape -
/// and it is what `gr_parity`'s `mixed vs reference` caught at 2.0e+01 in round 196.
template <bool FP32_ACT>
__dpct_inline__ void gr_norm_kernel(const float *__restrict__ R,
                                    const float *__restrict__ w_norm, float eps,
                                    int n_embd, float *__restrict__ xn,
                                    uint16_t *__restrict__ xq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &scratch = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int c = item_ct1.get_group(2);
    const float* Rc = R + (size_t) c * n_embd;
    float* xnc = xn + (size_t) c * n_embd;
    uint16_t* xqc = xq + (size_t) c * n_embd;

    // FP32, for the same reason as `gr_down_kernel` and `gr_inject_kernel`: this is a GeForce part, FP64 runs
    // at a small fraction of FP32, and the sum of squares is a pure reduction.  It was the last `double` on the
    // `gr_read` path.  `gr_norm` is only 0.80 ms/token, so this is hygiene rather than a headline - but it is
    // 12 of the FP64 sites the review's G5 counted in this file, and the rule it states ("no FP64 on any decode
    // or prefill path") is worth being able to check mechanically rather than case by case.
    float ss = 0.0f;
    for (int d = item_ct1.get_local_id(2); d < n_embd;
         d += item_ct1.get_local_range(2)) {
        const float v = Rc[d];
        ss += v * v;
    }
    const float ms = block_sumf(ss, scratch) / (float) n_embd;
    const float rs = sycl::rsqrt(ms + eps);
    for (int d = item_ct1.get_local_id(2); d < n_embd;
         d += item_ct1.get_local_range(2)) {
        const float x = Rc[d] * rs * w_norm[(size_t) c * n_embd + d];
        xnc[d] = x;
        if constexpr (!FP32_ACT) xqc[d] = f32_to_bf16_bits(x);
    }
}

/// `lo = silu((bf16(xn) @ w_down.T) / hc)`, ONE BLOCK PER OUTPUT ROW with its warps splitting the reduction.
template <typename Activation>
__dpct_inline__ void gr_down_kernel(const Activation *__restrict__ xq,
                                    const uint16_t *__restrict__ w_down,
                                    int hc_dim, int hc_lr, int hc,
                                    Activation *__restrict__ lq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int k = item_ct1.get_group(2);
    if (k >= hc_lr) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const int warp = item_ct1.get_local_id(2) >> 5;
    const int nw = (int)(item_ct1.get_local_range(2) >>
                         5); // warps per row == warps per block
    const uint16_t* row = w_down + (size_t) k * hc_dim;
    // **THE OCCUPANCY HERE IS 10% AND IT IS REAL, BUT FIXING IT DOES NOT PAY.**  `hc_lr` is 320, one warp per
    // row, 8 warps per block, so this launches 40 blocks = 320 warps: **6.7 warps per SM out of the 64 a
    // Blackwell SM holds**, each lane walking 10,240 elements in 320 dependent steps.
    //
    // Four independent accumulators were tried to break the dependency chain and put four loads in flight per
    // lane.  **Measured: 39.86 ms/token of pure GPU against 39.80, i.e. no change**, with `gr_parity` green
    // either way.  Removed, on the rule this file already states about `#pragma unroll 4`: a hint that does not
    // change the timing is a claim that is not true.
    //
    // The reason is that `gr_read`'s cost is not here.  Its four kernels are the norm (4 blocks), this (40),
    // `gr_gate_kernel` (1,280) and the mean (10), and the 1,280-block gate kernel reads 6.55 MB of weights with
    // good occupancy - so `gr_read`'s ~93 us is mostly the gate kernel at its own memory rate, and the ~165 GB/s
    // whole-call figure is an average over kernels that do not share a rate.
    // **FP32, AND THE FP64 THAT WAS HERE COST MORE THAN EVERYTHING ELSE IN `gr_read`.**
    //
    // This accumulated in `double`.  GeForce Blackwell executes FP64 at a small fraction of FP32 - 1/64 on
    // recent parts - so a 320-step dependent FP64 chain per lane turned a memory-bound reduction into a
    // latency-bound one.  The `nsys` per-kernel table (round 2) measured it at **5.07 ms/token, 15.3% of all
    // GPU time**, against `gr_gate_kernel`'s 1.23 ms - which REFUTES the comment that used to sit here, that
    // "gr_read's ~93 us is mostly the gate kernel at its own memory rate".  It was not: down + inject were
    // 8.77 of gr_read's 10.8 ms, and the gate was 1.23.
    //
    // The justification for `double` was matching `ref/gr.py`'s float64 transcription.  **The ground truth is
    // llama.cpp, and llama.cpp accumulates in FP32** - so this does not move away from the oracle, it moves
    // toward it, and the review's G1 lists it first.
    //
    // Four independent accumulators were tried here BEFORE, against the FP64 version, and measured no change
    // (39.86 vs 39.80 ms/token) - because FP64 throughput was the limit, not the dependency chain.  With FP32
    // the kernel becomes memory-bound: it reads ~13 MB per call across xq and w_down, which at the measured
    // 641 GB/s is ~20 us against the 52 us this took.
    // **THE OCCUPANCY CONCLUSION ABOVE IS STALE, AND THIS IS THE RETEST IT ASKED FOR.**
    //
    // "Fixing it does not pay" was measured against the FP64 version, where FP64 THROUGHPUT was the limit and
    // more parallelism could not help.  The kernel is FP32 now (round 282: 5.07 -> 2.38 ms/token) and the limit
    // moved to memory - which is the regime where occupancy does pay, because at 6.7 warps/SM there are not
    // enough bytes in flight to cover DRAM latency.  The old result is not wrong; it is about a different kernel.
    //
    // One BLOCK per row, its 8 warps splitting the 10,240-element reduction: 320 blocks x 8 = **2,560 warps =
    // 53 warps/SM instead of 6.7**, with no second kernel and no partial buffer - the eight partials meet in
    // shared memory.  Lanes within a warp still read consecutive addresses, because warp `w` starts at
    // `w * 32` and strides by `nw * 32`.
    float acc = 0.0f;
#pragma unroll
    for (int i = warp * 32 + lane; i < hc_dim; i += nw * 32)
        acc += activation_f32(xq[i]) * bf16_bits_to_f32(row[i]);
    acc = warp_sumf(acc);

    auto &part =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (lane == 0) part[warp] = acc;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (warp == 0) {
        float t = (lane < nw) ? part[lane] : 0.0f;
        t = warp_sumf(t);
        // NOTE: the reduction order differs from the one-warp version, so this is a NUMERICS CHANGE.  C1 is the
        // gate that judges it - not this comment, and not `gr_parity`, which compares against the reference on
        // a fixture that a changed summation order can still pass.
        if (lane == 0) store_activation(lq, k, silu_f(t / (float) hc));
    }
}

/// `gated[i] = xn[i] * sigmoid(bf16(lo) @ w_up.T)`, one warp per output row, lanes striding hc_lr.
template <typename Activation>
__dpct_inline__ void gr_gate_kernel(const Activation *__restrict__ lq,
                                    const uint16_t *__restrict__ w_up,
                                    const float *__restrict__ xn, int hc_dim,
                                    int hc_lr, float *__restrict__ gated) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i =
        item_ct1.get_group(2) * WARPS + (item_ct1.get_local_id(2) >> 5);
    if (i >= hc_dim) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const uint16_t* row = w_up + (size_t) i * hc_lr;
    float acc = 0.0f;
#pragma unroll
    for (int k = lane; k < hc_lr; k += 32)
        acc += activation_f32(lq[k]) * bf16_bits_to_f32(row[k]);
    acc = warp_sumf(acc);
    // Every lane computes the sigmoid (the shuffle broadcasts acc), so there is no second sync point and no
    // branch; lane 0 stores.
    if (lane == 0) gated[i] = xn[i] * sigmoid_f(acc);
}

/// `mixed[d] = mean over c of gated[c][d]`.  Flat elementwise: the mean is over the streams, which are
/// strided by n_embd, so no cross-thread reduction is needed at all.
__dpct_inline__ void gr_mean_kernel(const float *__restrict__ gated, int n_embd,
                                    int hc, float *__restrict__ mixed) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int d = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (d >= n_embd) return;
    float m = 0.0f;
#pragma unroll
    for (int c = 0; c < hc; ++c) m += gated[(size_t)c * n_embd + d];
    mixed[d] = m / (float) hc;
}

/// `inject[c] = bf16(xn) @ w_inject[c]`, one value per stream: ONE WARP PER STREAM, lanes striding hc_dim.
///
/// This mirrors `gr_down_kernel` rather than using a block-wide reduction.  `hc` is 4 at the real geometry,
/// so a block per stream would put 256 threads on a 10240-element dot and leave the reduction to do all the
/// work; a warp per stream is the same shape as the down projection and reuses its pattern exactly.  The
/// whole kernel is one block of `32*hc` threads, which is small - and it is a 4-value output, so it is.
template <typename Activation>
__dpct_inline__ void gr_inject_kernel(const Activation *__restrict__ xq,
                                      const uint16_t *__restrict__ w_inject,
                                      int hc_dim, int hc,
                                      float *__restrict__ inject) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_local_id(2) >> 5;
    if (c >= hc) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const uint16_t* row = w_inject + (size_t) c * hc_dim;
    // FP32, for the same reason as `gr_down_kernel` - and this one is worse.  A ONE-BLOCK kernel doing four
    // 10,240-element FP64 reductions measured **3.70 ms/token, 11.1% of all GPU time, from a single SM out of
    // 48.**  It is the fourth largest consumer of GPU time in the engine and it cannot use more than 1/48th of
    // the machine.  In FP32 the same reduction is a few microseconds and the block size stops mattering.
    float acc = 0.0f;
#pragma unroll
    for (int i = lane; i < hc_dim; i += 32)
        acc += activation_f32(xq[i]) * bf16_bits_to_f32(row[i]);
    acc = warp_sumf(acc);
    if (lane == 0) inject[c] = acc;
}

__dpct_inline__ void gr_write_kernel(const float *__restrict__ R,
                                     const float *__restrict__ block_out,
                                     const float *__restrict__ inject,
                                     int n_embd, int hc,
                                     float *__restrict__ out,
                                     uint8_t *dpct_local) {
    // w = 2*sigmoid(inject/hc), computed once per stream in shared rather than per element.
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto smem_raw = (unsigned char *)dpct_local;
    float* w = (float*) smem_raw;
    if ((int)item_ct1.get_local_id(2) < hc) w[item_ct1.get_local_id(2)] =
        2.0f * sigmoid_f(inject[item_ct1.get_local_id(2)] / (float)hc);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const long long n = (long long) hc * n_embd;
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n; i += (long long)item_ct1.get_group_range(2) *
                     item_ct1.get_local_range(2)) {
        const int c = (int) (i / n_embd), d = (int) (i % n_embd);
        // every stream adds the SAME block output; only the weight differs per stream
        out[i] = R[i] + block_out[d] * w[c];
    }
}

}  // namespace

#include "gr.kern.inc"

}  // namespace strata::kernels
