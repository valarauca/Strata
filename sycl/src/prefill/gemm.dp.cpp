// src/prefill/gemm.dp.cpp - the prompt path's dense products on the XMX units (Intel engine).
//
// Y[M, N] (fp32, row stride ldy) = X[M, K] (fp16 or bf16, row stride ldx) . W[N, K]^T (row stride ldw), optionally
// added to Y. Each sub-group holds an (MT x 8) by (NT x 16) block of Y in joint_matrix accumulators (Xe2: 8x16x16 on
// sub-group 16) and walks K in steps of 16, loading its A tiles from X and its B tiles from W (W's rows are B's
// columns: col_major with stride ldw) straight from global memory with bounds-checked 2-D block loads, which return
// zeros past M, N and K, so any shape runs without padding. The bundle is built with the large register file.
#include "strata/port.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace strata::prefill::xgemm {
namespace {

namespace jm = sycl::ext::oneapi::experimental::matrix;
namespace jmi = sycl::ext::intel::experimental::matrix;
using half = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using sycl::access::address_space;
using sycl::access::decorated;

template <typename T, int SGM, int SGN, int MT, int NT>
inline void gemm_tile(const T* X, long long ldx, const T* W, long long ldw, float* Y, long long ldy, int M, int N, int K,
                      int accumulate) {
    constexpr int TM = 8, TN = 16, TK = 16;
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = it.get_sub_group();
    const int sgid = (int) sg.get_group_linear_id();
    const int sgm = sgid / SGN, sgn = sgid % SGN;
    const int m0 = (int) it.get_group(1) * (SGM * MT * TM) + sgm * MT * TM;
    const int n0 = (int) it.get_group(2) * (SGN * NT * TN) + sgn * NT * TN;
    if (m0 >= M || n0 >= N) return;
    auto xp = sycl::address_space_cast<address_space::global_space, decorated::no>(X);
    auto wp = sycl::address_space_cast<address_space::global_space, decorated::no>(W);
    auto yp = sycl::address_space_cast<address_space::global_space, decorated::no>(Y);
    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, TM, TN> acc[MT][NT];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (accumulate)
                jmi::joint_matrix_load_checked(sg, acc[i][j], yp, (size_t) ldy, jm::layout::row_major, (size_t) M, (size_t) N,
                                              (size_t) (m0 + i * TM), (size_t) (n0 + j * TN));
            else
                jm::joint_matrix_fill(sg, acc[i][j], 0.0f);
        }
    for (int k = 0; k < K; k += TK) {
        jm::joint_matrix<sycl::sub_group, T, jm::use::a, TM, TK, jm::layout::row_major> a[MT];
        jm::joint_matrix<sycl::sub_group, T, jm::use::b, TK, TN, jm::layout::col_major> b[NT];
#pragma unroll
        for (int i = 0; i < MT; ++i)
            jmi::joint_matrix_load_checked(sg, a[i], xp, (size_t) ldx, (size_t) M, (size_t) K, (size_t) (m0 + i * TM), (size_t) k);
#pragma unroll
        for (int j = 0; j < NT; ++j)
            jmi::joint_matrix_load_checked(sg, b[j], wp, (size_t) ldw, (size_t) K, (size_t) N, (size_t) k, (size_t) (n0 + j * TN));
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) jm::joint_matrix_mad(sg, acc[i][j], a[i], b[j], acc[i][j]);
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
            jmi::joint_matrix_store_checked(sg, acc[i][j], yp, (size_t) ldy, jm::layout::row_major, (size_t) M, (size_t) N,
                                           (size_t) (m0 + i * TM), (size_t) (n0 + j * TN));
}

// N below a tile (the shared expert's one-column gate): one sub-group per output, fp32 lane sums over K, then the
// sub-group's tree reduce.
template <typename T>
inline void gemm_narrow(const T* X, long long ldx, const T* W, long long ldw, float* Y, long long ldy, int M, int N, int K,
                        int accumulate) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = it.get_sub_group();
    const int m = (int) it.get_group(1);
    const int n = (int) it.get_group(2);
    if (m >= M || n >= N) return;
    const int lane = (int) sg.get_local_linear_id();
    const T* x = X + (size_t) m * ldx;
    const T* w = W + (size_t) n * ldw;
    float s = 0.0f;
    for (int k = lane; k < K; k += 16) s += (float) x[k] * (float) w[k];
    s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
    if (lane == 0) {
        float* y = Y + (size_t) m * ldy + n;
        *y = accumulate ? *y + s : s;
    }
}

}  // namespace

#include "gemm.kern.inc"

}  // namespace strata::prefill::xgemm
