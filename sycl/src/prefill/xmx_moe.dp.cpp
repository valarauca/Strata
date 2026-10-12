// A770 port: grouped FP16 XMX GEMM for the prompt path's experts (STRATA_PF_XMX=1).
// Y[r][n] = sum_k X[r][k] * W_e[n][k] for each expert e of a group, its rows r in [row0 + off[e], row0 + off[e+1]).
// One launch per group instead of one oneMKL call per expert.  joint_matrix 8x8x16 (DG2), SLM-tiled, 256 GRF.
// Measured standalone on the A770 (64 experts, gamma-distributed rows, vs per-expert oneMKL): down 6.1x at mean 40
// rows, 2.8x at 160, 1.75x at 320; gate/up 3.0x / 1.46x / 0.79x (oneMKL wins above ~250 rows: the caller picks).
#include "strata/port.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include "strata/prefill/xmx_moe.hpp"
#include "xmx_moe_shapes.hpp"

namespace strata::prefill::xmx {
namespace {
using half = sycl::half;
namespace jm = sycl::ext::oneapi::experimental::matrix;

template <class C>
void grouped_kernel(const half* X, float* Y, const Grp& gp, int N, int K) {
    constexpr int BM = C::BM, BN = C::BN, KS = C::KS, WG = C::WG, SG = C::SG;
    constexpr int LDA = KS + 8, LDB = KS + 8;
#ifndef STRATA_NO_DG2_XMX
    constexpr int TN = kSgW;
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = it.get_sub_group();
    const int t = (int) it.get_group(1);
    int e = 0;
    while (e + 1 < gp.n && t >= gp.tstart[e + 1]) ++e;
    const int m0 = (t - gp.tstart[e]) * BM;
    const int rows = sycl::min(BM, gp.off[e + 1] - gp.off[e] - m0);
    const half* Xe = X + ((size_t) gp.off[e] + m0) * K;
    float* Ye = Y + ((size_t) gp.off[e] + m0) * N;
    const half* We = reinterpret_cast<const half*>(gp.w[e]);
    const int n0 = (int) it.get_group(2) * BN;
    const int lid = (int) it.get_local_id(2), sgid = lid / SG;
    const int sgm = sgid / C::SGN, sgn = sgid % C::SGN;
    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, TN> acc[C::MT][C::NT];
    for (int i = 0; i < C::MT; ++i)
        for (int j = 0; j < C::NT; ++j) jm::joint_matrix_fill(sg, acc[i][j], 0.0f);
    half* pa = strata_slm<half>();
    half* pb = pa + BM * LDA;
    auto sa = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(pa);
    auto sb = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(pb);
    for (int k0 = 0; k0 < K; k0 += KS) {
        constexpr int VA = BM * KS / 8, VB = BN * KS / 8;
        for (int v = lid; v < VA; v += WG) {
            const int r = v / (KS / 8), c = (v % (KS / 8)) * 8;
            sycl::vec<half, 8> x(0);
            if (r < rows) x = *reinterpret_cast<const sycl::vec<half, 8>*>(Xe + (size_t) r * K + k0 + c);
            *reinterpret_cast<sycl::vec<half, 8>*>(pa + r * LDA + c) = x;
        }
        for (int v = lid; v < VB; v += WG) {
            const int r = v / (KS / 8), c = (v % (KS / 8)) * 8;
            *reinterpret_cast<sycl::vec<half, 8>*>(pb + r * LDB + c) =
                *reinterpret_cast<const sycl::vec<half, 8>*>(We + (size_t) (n0 + r) * K + k0 + c);
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (int kk = 0; kk < KS; kk += 16) {
            jm::joint_matrix<sycl::sub_group, half, jm::use::a, 8, 16, jm::layout::row_major> a[C::MT];
            jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, TN, jm::layout::col_major> b[C::NT];
            for (int i = 0; i < C::MT; ++i)
                jm::joint_matrix_load(sg, a[i], sa + (sgm * C::MT * 8 + i * 8) * LDA + kk, LDA);
            for (int j = 0; j < C::NT; ++j)
                jm::joint_matrix_load(sg, b[j], sb + (sgn * C::NT * TN + j * TN) * LDB + kk, LDB);
            for (int i = 0; i < C::MT; ++i)
                for (int j = 0; j < C::NT; ++j) jm::joint_matrix_mad(sg, acc[i][j], a[i], b[j], acc[i][j]);
        }
        it.barrier(sycl::access::fence_space::local_space);
    }
    for (int i = 0; i < C::MT; ++i) {
        const int r = sgm * C::MT * 8 + i * 8;
        if (r >= rows) continue;
        for (int j = 0; j < C::NT; ++j) {
            const int c = n0 + sgn * C::NT * TN + j * TN;
            if (r + 8 <= rows) {
                jm::joint_matrix_store(sg, acc[i][j],
                    sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        Ye + (size_t) r * N + c), N, jm::layout::row_major);
            } else {   // partial rows: through this sub-group's own SLM slice
                float* st = reinterpret_cast<float*>(pa) + sgid * 8 * TN;
                jm::joint_matrix_store(sg, acc[i][j],
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(st),
                    TN, jm::layout::row_major);
                sycl::group_barrier(sg);
                const int lim = rows - r, ln = (int) sg.get_local_linear_id();
                for (int rr = 0; rr < lim; ++rr) Ye[(size_t) (r + rr) * N + c + ln] = st[rr * TN + ln];
                sycl::group_barrier(sg);
            }
        }
    }
#endif
}
}  // namespace

#include "xmx_moe.kern.inc"

}  // namespace strata::prefill::xmx
