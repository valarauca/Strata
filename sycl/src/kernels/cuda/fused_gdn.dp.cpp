// src/kernels/cuda/fused_gdn.cu - see include/strata/kernels/fused_gdn.hpp.
#include "strata/port.hpp"
#include "strata/kernels/fused_gdn.hpp"

namespace strata::kernels {
namespace {

constexpr int S = 128;          // state size (rows = cols = 128)
constexpr int RG = 4;           // row groups
constexpr int RPG = S / RG;     // 32 rows per thread

__dpct_inline__ void gdn_step_norm_kernel(
    float *__restrict__ state, const float *__restrict__ q,
    const float *__restrict__ k, const float *__restrict__ v,
    const float *__restrict__ gate, const float *__restrict__ beta,
    const float *__restrict__ z, const float *__restrict__ gamma, float eps,
    float *__restrict__ y, int h_k, int h_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red_kv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red_o =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &wsum = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[S * RG / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int col = item_ct1.get_local_id(2); // 0..127
    const int rg = item_ct1.get_local_id(1);  // 0..3
    const int tid = rg * S + col;
    const int qh = head % h_k;
    if (tid < S) { sk[tid] = k[qh * S + tid]; sq[tid] = q[qh * S + tid]; }
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    item_ct1.barrier();
    const float g = sycl::native::exp(gate[head]);
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
    red_kv[rg][col] = kv;
    item_ct1.barrier();
    const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
    const float delta = (v[head * S + col] - g * kv_col) * beta[head];
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
        o = sycl::fma(s[r], sq[rg * RPG + r], o);
        base[r * row_stride] = s[r];
    }
    red_o[rg][col] = o;
    item_ct1.barrier();
    float oc = 0.0f, sq_part = 0.0f;
    if (rg == 0) {
        oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) *
             sycl::rsqrt((float)S);
        sq_part = oc * oc;
    }
    // RMS over the head's 128 outputs: warps of row group 0 are threads 0..127.
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            sq_part, o2);
    if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
    item_ct1.barrier();
    if (rg == 0) {
        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
        const float scale = sycl::rsqrt(ss / (float)S + eps);
        const float zz = z[head * S + col];
        y[head * S + col] =
            oc * scale * gamma[col] * (1.0f / (1.0f + sycl::native::exp(-zz)));
    }
}

__dpct_inline__ void gdn_conv_l2_kernel(float *__restrict__ hist,
                                        const float *__restrict__ qkv,
                                        const float *__restrict__ w,
                                        float *__restrict__ h, int qk_heads,
                                        float eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &part =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int c = item_ct1.get_group(2) * S + item_ct1.get_local_id(2);
    const float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2], x = qkv[c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    hist[c * 3] = v1;
    hist[c * 3 + 1] = v2;
    hist[c * 3 + 2] = x;
    float y = sum / (1.0f + sycl::native::exp(-sum));
    if ((int)item_ct1.get_group(2) < qk_heads) {
        float sq = y * y;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                sq, o);
        if ((item_ct1.get_local_id(2) & 31) == 0)
            part[item_ct1.get_local_id(2) >> 5] = sq;
        item_ct1.barrier();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= sycl::rsqrt(ss + eps);
    }
    h[c] = y;
}

__dpct_inline__ void
gdn_ab_kernel(const float *__restrict__ x, const uint16_t *__restrict__ wa,
              const uint16_t *__restrict__ wb, const float *__restrict__ dt,
              const float *__restrict__ ssm_a, float *__restrict__ gate,
              float *__restrict__ beta, int n, int h_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2) * 8 + (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(
        (is_beta ? wb : wa) + (size_t)r * n);
    float acc = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const sycl::uint4 wv = *(w4 + j);
        const sycl::float4 xa =
            *reinterpret_cast<const sycl::float4 *>(x + j * 8);
        const sycl::float4 xb =
            *reinterpret_cast<const sycl::float4 *>(x + j * 8 + 4);
        acc = sycl::fma(sycl::bit_cast<float>(wv.x() << 16), (float)(xa.x()),
                        acc);
            acc = sycl::fma(sycl::bit_cast<float>(wv.x() & 0xffff0000u),
                            (float)(xa.y()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.y() << 16), (float)(xa.z()),
                        acc);
            acc = sycl::fma(sycl::bit_cast<float>(wv.y() & 0xffff0000u),
                            (float)(xa.w()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.z() << 16), (float)(xb.x()),
                        acc);
            acc = sycl::fma(sycl::bit_cast<float>(wv.z() & 0xffff0000u),
                            (float)(xb.y()), acc);
        acc = sycl::fma(sycl::bit_cast<float>(wv.w() << 16), (float)(xb.z()),
                        acc);
            acc = sycl::fma(sycl::bit_cast<float>(wv.w() & 0xffff0000u),
                            (float)(xb.w()), acc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            acc, o);
    if (lane != 0) return;
    if (is_beta) {
        beta[r] = 1.0f / (1.0f + sycl::native::exp(-acc));
    } else {
        const float v = acc + dt[r];
        const float sp = v > 20.0f ? v : sycl::log1p(sycl::native::exp(v));
        gate[r] = sp * ssm_a[r];
    }
}

}  // namespace

#include "fused_gdn.kern.inc"

}  // namespace strata::kernels
