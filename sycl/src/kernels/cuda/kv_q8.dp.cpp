// src/kernels/cuda/kv_q8.cu - see include/strata/kernels/kv_q8.hpp.
#include "strata/port.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/f16_bits.hpp"

namespace strata::kernels {
namespace {

// One block = one 64-value group of one KV head of K (blockIdx.z = 0) or V (1); 64 threads, one value each.
// MULTI (S26 STRATA_LFUSE): blockIdx.z = 2 * token + (K / V); token j reads step + j * step_stride and its rows at
// kcur / vcur + j * cur_stride - every token's code is the single launch's
template <bool MULTI = false>
__dpct_inline__ void kv_append_q8_kernel(
    int8_t *__restrict__ k_q, int8_t *__restrict__ v_q,
    uint16_t *__restrict__ k_scale, uint16_t *__restrict__ v_scale,
    const int32_t *__restrict__ table, const int32_t *__restrict__ step,
    const float *__restrict__ kcur, const float *__restrict__ vcur,
    int kv_heads, int head_dim, int page_size, KvHostPools host,
    int step_stride = 0, int cur_stride = 0) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if constexpr (MULTI) {
        const int j = item_ct1.get_group(0) >> 1;
        step += (size_t) j * step_stride; kcur += (size_t) j * cur_stride; vcur += (size_t) j * cur_stride;
    }
    const long long pos = (long long)*(step + kStepPos);
    const int h = item_ct1.get_group(2), g = item_ct1.get_group(1),
              t = item_ct1.get_local_id(2);
    const bool is_v =
        MULTI ? (item_ct1.get_group(0) & 1) == 1 : item_ct1.get_group(0) == 1;
    const int groups = head_dim / KV_Q8_GROUP;
    const float x = (is_v ? vcur : kcur)[h * head_dim + g * KV_Q8_GROUP + t];
    // max |x| over the 64 values: two warps, then combine through shared memory in a fixed order
    float a = sycl::fabs(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = sycl::fmax(
        a, dpct::experimental::permute_sub_group_by_xor(
               0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
               a, o));
    auto &warp_max =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if ((t & 31) == 0) warp_max[t >> 5] = a;
    item_ct1.barrier();
    const float amax = sycl::fmax(warp_max[0], warp_max[1]);
    const uint16_t sbits = f16_from_f32(amax / 127.0f);
    const float sf = f32_from_f16(sbits);                          // quantize against the STORED scale
    int q = 0;
    if (sf > 0.0f) {
        q = sycl::vec<float, 1>{(x / sf)}
                .convert<int, sycl::rounding_mode::rte>()[0];
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    // KV streaming: the host copy (identity layout) always, the VRAM page only if the block is resident
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? v_q : k_q)[row * head_dim + g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) (is_v ? v_scale : k_scale)[row * groups + g] = sbits;
    }
    if (host.k_q != nullptr) {
        const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
        (is_v ? host.v_q : host.k_q)[row * head_dim + g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) (is_v ? host.v_scale : host.k_scale)[row * groups + g] = sbits;
    }
}

// One thread = 4 consecutive values of one cell and head (as the FP16 gather does with uint2).
__dpct_inline__ void kv_gather_q8_kernel(
    const int8_t *__restrict__ k_q, const int8_t *__restrict__ v_q,
    const uint16_t *__restrict__ k_scale, const uint16_t *__restrict__ v_scale,
    const int32_t *__restrict__ table, const int32_t *__restrict__ ids,
    const int32_t *__restrict__ step, int kv_heads, int head_dim, int page_size,
    uint16_t *__restrict__ k_scratch, uint16_t *__restrict__ v_scratch) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long n_ids = (long long)*(step + kStepWidth);
    const int per = head_dim / 4;
    const long long total = n_ids * kv_heads * per;
    const long long i =
        item_ct1.get_group(2) * (long long)item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= total) return;
    const long long id = i / (kv_heads * (long long) per);
    const int rem = (int) (i % (kv_heads * (long long) per));
    const int h = rem / per, q4 = rem - h * per;
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);
    const int d = q4 * 4;
    const int groups = head_dim / KV_Q8_GROUP;
    const float ks = f32_from_f16(k_scale[row * groups + d / KV_Q8_GROUP]);
    const float vs = f32_from_f16(v_scale[row * groups + d / KV_Q8_GROUP]);
    const sycl::char4 kc =
        reinterpret_cast<const sycl::char4 *>(k_q + row * head_dim)[q4];
    const sycl::char4 vc =
        reinterpret_cast<const sycl::char4 *>(v_q + row * head_dim)[q4];
    sycl::ushort4 ko, vo;
    ko.x() = f16_from_f32((float)kc.x() * ks);
        ko.y() = f16_from_f32((float)kc.y() * ks);
    ko.z() = f16_from_f32((float)kc.z() * ks);
        ko.w() = f16_from_f32((float)kc.w() * ks);
    vo.x() = f16_from_f32((float)vc.x() * vs);
        vo.y() = f16_from_f32((float)vc.y() * vs);
    vo.z() = f16_from_f32((float)vc.z() * vs);
        vo.w() = f16_from_f32((float)vc.w() * vs);
    const long long dst = (id * kv_heads + h) * (long long) per + q4;
    reinterpret_cast<sycl::ushort4 *>(k_scratch)[dst] = ko;
    reinterpret_cast<sycl::ushort4 *>(v_scratch)[dst] = vo;
}

}  // namespace

#include "kv_q8.kern.inc"

}  // namespace strata::kernels
