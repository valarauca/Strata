// src/kernels/cuda/cvec.cu - see include/strata/kernels/cvec.hpp.
#include "strata/port.hpp"
#include "strata/kernels/cvec.hpp"

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

// the fused hyper-connection read's gate (fused_gr.cu), so a write done here is bitwise the one it would have folded
__dpct_inline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

// one block per (stream, token): the pending write, then h . v over the stream, then the update
__dpct_inline__ void
cvec_kernel(float *__restrict__ R, const float *__restrict__ dir,
            const float *__restrict__ s_l, const int *__restrict__ on, int mode,
            int64_t layer, int n, int hc, int64_t r_ld,
            const float *__restrict__ bo, int64_t bo_ld,
            const float *__restrict__ inj, int64_t inj_ld, int write) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2);
    const int64_t t = item_ct1.get_group(1);
    float* r = R + t * r_ld + (int64_t) c * n;
    const float s = s_l[layer];
    const bool steer = *on != 0 && s != 0.0f;   // uniform over the block
    if (!steer && !write) return;
    const float* v = dir + layer * n;
    const float w = write ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float) hc) : 0.0f;
    const float* b = write ? bo + t * bo_ld : nullptr;
    float x[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = item_ct1.get_local_id(2) + k * THREADS;
        if (d < n) {
            float xv = r[d];
            if (write) xv = sycl::fma((float)(b[d]), (float)w, xv);
            x[k] = xv;
            if (steer && mode == 0) dot = sycl::fma(xv, (float)(v[d]), dot);
        }
    }
    if (steer && mode == 0) {
        auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
            float[THREADS / 32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                dot, o);
        if ((item_ct1.get_local_id(2) & 31) == 0)
            part[item_ct1.get_local_id(2) >> 5] = dot;
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) < 32) {
            float p = item_ct1.get_local_id(2) < THREADS / 32
                          ? part[item_ct1.get_local_id(2)]
                          : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) p +=
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), p, o);
            if (item_ct1.get_local_id(2) == 0) part[0] = p;
        }
        item_ct1.barrier();
        dot = part[0] * s;   // s (h . v)
    }
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = item_ct1.get_local_id(2) + k * THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv =
                mode == 0 ? sycl::fma(-dot, (float)(v[d]), xv) : xv + v[d];
            r[d] = xv;
        }
    }
}

}  // namespace

#include "cvec.kern.inc"

}  // namespace strata::kernels
