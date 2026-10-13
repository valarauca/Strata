// src/kernels/cuda/ple.cu - P2.S4: the PLE block's GPU half.
//
// See include/strata/kernels/ple.hpp for the structure and for the `normalized`-is-the-conv-input finding.
//
// The legacy arithmetic below follows the captured CPU ggml graph
// (`bench/micro/ple_in.bin` / `ple_out.bin`, produced by `ple_layer_xcheck.cpp`):
//
//   * `rms_norm` accumulates `(double)(x*x)` with the PRODUCT rounded in f32 first - that is literally what
//     `ggml_compute_forward_rms_norm_f32` does (`sum += (ggml_float)(x[i00]*x[i00])` with `ggml_float` =
//     double), and it is not the same as accumulating in f32 or as widening before the multiply.
//   * `silu` is `x / (1 + expf(-x))` in f32, which is `ggml_silu_f32`.
//   * the gate's `sqrt` and `sigmoid` are f32.
//
// The opt-in native BF16 path replaces only ple_value with the pinned CUDA single-token BF16/F32 MMVF.
// A separate opt-in native postops path follows the pinned CUDA arithmetic after both projections.
#include "strata/port.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"

namespace strata::kernels {
namespace {

__dpct_inline__ void
history_advance_kernel(float *__restrict__ history,
                       const float *__restrict__ normalized) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int channel = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_local_id(2);
    if (channel >= NG_HC_DIM) return;
    float* column = history + (size_t) channel * NG_HIST;
#pragma unroll
    for (int row = 0; row + 1 < NG_HIST; ++row) column[row] = column[row + 1];
    column[NG_HIST - 1] = normalized[channel];
}

__dpct_inline__ uint16_t bf16_bits(float f) {
    uint32_t i;
    memcpy(&i, &f, 4);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (i >> 16);
}

__dpct_inline__ float bf16_float(uint16_t h) {
    const uint32_t i = (uint32_t) h << 16;
    float f;
    memcpy(&f, &i, 4);
    return f;
}

__dpct_inline__ float silu_f(float x) {
    return x / (1.0f + sycl::native::exp(-x));
}

/// Block-wide sum in double, broadcast to all threads.  The leading `__syncthreads` matters: the result is
/// read out of `scratch[0]` and a later call reuses the array, so without a barrier a fast thread can
/// overwrite it before a slow one has read it - invisible in most runs and a slightly different norm when it
/// fires.
inline double block_sum(double v, double *scratch) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    item_ct1.barrier();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v +=
        dpct::experimental::shift_sub_group_left(
            0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            off);
    if (lane == 0) scratch[warp] = v;
    item_ct1.barrier();
    const int nw = ((int)item_ct1.get_local_range(2) + 31) >> 5;
    v = (item_ct1.get_local_id(2) < nw) ? scratch[item_ct1.get_local_id(2)]
                                        : 0.0;
    if (warp == 0)
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v +=
            dpct::experimental::shift_sub_group_left(
                0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                v, off);
    if (item_ct1.get_local_id(2) == 0) scratch[0] = v;
    item_ct1.barrier();
    return scratch[0];
}

/// `grouped_norm`, ONE BLOCK PER STREAM.  The ggml form is `reshape_3d(x, n_embd, hc, T)` then
/// `rms_norm(..., eps) * w`, so ne0 = n_embd is the reduction axis and each hc stream is normalised on its
/// own.  In the flat ggml layout (ne0 fastest) stream c occupies [c*n_embd, (c+1)*n_embd).
__dpct_inline__ void gnorm_kernel(const float *__restrict__ x,
                                  const float *__restrict__ w,
                                  float *__restrict__ y, int n_embd,
                                  float eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &scratch = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int c = item_ct1.get_group(2);
    const float* xc = x + (size_t) c * n_embd;
    const float* wc = w + (size_t) c * n_embd;
    float* yc = y + (size_t) c * n_embd;

    double acc = 0.0;
    for (int d = item_ct1.get_local_id(2); d < n_embd;
         d += item_ct1.get_local_range(2)) {
        // The PRODUCT is rounded in f32 before it is widened - `(ggml_float)(x[i00]*x[i00])`.
        const float sq = xc[d] * xc[d];
        acc += (double) sq;
    }
    const float mean = (float) (block_sum(acc, scratch) / (double) n_embd);
    const float scale = 1.0f / sycl::sqrt(mean + eps);
#pragma unroll
    for (int d = item_ct1.get_local_id(2); d < n_embd;
         d += item_ct1.get_local_range(2)) yc[d] = xc[d] * scale * wc[d];
}

/// `s[c] = sum_d key[c][d]*query[c][d] / sqrt(n_embd)`, then the signed square root and the sigmoid.
///
/// `mag = sqrt(clamp(|s|, 1e-6, inf))` and `gate = sigmoid(sgn(s) * mag)`.  The clamp is a floor on |s|, so
/// it only bites near zero; the SIGN is carried separately, which is what keeps the gate symmetric about 0.5.
__dpct_inline__ void gate_kernel(const float *__restrict__ key,
                                 const float *__restrict__ query,
                                 float *__restrict__ gate, int n_embd,
                                 float inv_sqrt_n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &scratch = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int c = item_ct1.get_group(2);
    const float* kc = key + (size_t) c * n_embd;
    const float* qc = query + (size_t) c * n_embd;
    double acc = 0.0;
#pragma unroll
    for (int d = item_ct1.get_local_id(2); d < n_embd;
         d += item_ct1.get_local_range(2)) acc += (double)(kc[d] * qc[d]);
    const float s = (float) (block_sum(acc, scratch) / (double) 1.0) * inv_sqrt_n;
    const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(s), 1e-6f));
    const float sgn = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
    if (item_ct1.get_local_id(2) == 0) gate[c] =
        1.0f / (1.0f + sycl::native::exp(-(sgn * mag)));
}

/// `gated[c][d] = value[d] * gate[c]` - the value broadcast across the hc streams.
__dpct_inline__ void bcast_kernel(const float *__restrict__ value,
                                  const float *__restrict__ gate,
                                  float *__restrict__ gated, int n_embd,
                                  int hc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n_embd * hc) return;
    gated[i] = value[i % n_embd] * gate[i / n_embd];
}

/// The depthwise causal dilated conv, then SiLU.  One thread per channel.
///
/// `out[c] = sum_k kW[k][c] * terms[t - (kern-1-k)*dil][c]` with `t = hist` (the first new position), so tap k
/// reads row `hist - (kern-1-k)*dil` of `[history | new]`.  For the real geometry (kern 4, dil 3, hist 9) the
/// four taps read rows 0, 3, 6 and 9 - and row 9 is the NEW row, which is the only one beyond the history.
/// That is why no `terms` buffer is built: three taps come from the caller's history and one from `norm`.
///
/// `kW` IS GGML-NATIVE: `kW[k + kern*c]`.  The manifest's shape is [4, 10240] with ne0 = 4 fast, and
/// `ref/ngram.py`'s (4, 10240) numpy `kernel[k][c]` is the TRANSPOSE of that.  The source's own view
/// (`ggml_view_2d(model.layers[il].ple_conv1d, 1, hc_dim, nb[1], k*nb[0])`) settles which one is meant.
__dpct_inline__ void conv_kernel(const float *__restrict__ hist,
                                 const float *__restrict__ norm,
                                 const uint16_t *__restrict__ kW,
                                 float *__restrict__ out, int hc_dim, int kern,
                                 int dil, int nhist) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= hc_dim) return;
    float acc = 0.0f;
    for (int k = 0; k < kern; ++k) {
        const int row = nhist - (kern - 1 - k) * dil;      // tap 0 reads the FURTHEST back
        // ROW-FASTEST: `hist[row + nhist*c]`.  The conv state is `ggml_reshape_3d(state, d_conv-1,
        // conv_channels, n_seqs)`, so ne0 = hist varies fastest and the flat index is row + hist*channel.
        // Channel-slowest (`row*hc_dim + c`) is the natural thing to write and would read a transposed state -
        // `ple_layer_xcheck.cpp` L79-83 records getting this wrong once already.
        const float v = (row == nhist) ? norm[c] : hist[(size_t) row + (size_t) nhist * c];
        acc += f32_from_f16(kW[k + kern * c]) * v;
    }
    out[c] = silu_f(acc);
}

/// `result = hidden + gated + conv`, elementwise over hc_dim.
__dpct_inline__ void add3_kernel(const float *__restrict__ hidden,
                                 const float *__restrict__ gated,
                                 const float *__restrict__ conv,
                                 float *__restrict__ result, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) result[i] = hidden[i] + gated[i] + conv[i];
}

/// `y[o] = sum_i bf16(x[i]) * bf16(w[o*n_in + i])`.  The weight is BF16 and so is the ACTIVATION, which is
/// the contract for a BF16 tensor (`docs/activation-contract.md`); with both sides bf16 the products are
/// exact in f32 and only the summation order differs from ggml's.
__dpct_inline__ void bf16_gemv_kernel(const uint16_t *__restrict__ x,
                                      const uint16_t *__restrict__ w,
                                      float *__restrict__ y, int n_in,
                                      int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int o = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (o >= n_out) return;
    const uint16_t* row = w + (size_t) o * n_in;
    double acc = 0.0;
#pragma unroll
    for (int i = 0; i < n_in; ++i) acc +=
        (double)bf16_float(x[i]) * (double)bf16_float(row[i]);
    y[o] = (float) acc;
}

__dpct_inline__ void to_bf16_kernel(const float *__restrict__ x,
                                    uint16_t *__restrict__ y, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) y[i] = bf16_bits(x[i]);
}

}  // namespace

#include "ple.kern.inc"

}  // namespace strata::kernels
