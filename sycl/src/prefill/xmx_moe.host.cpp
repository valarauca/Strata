// src/prefill/xmx_moe.host.cpp - the host side of xmx_moe.dp.cpp: the mode switches, the group table and the launch.
#include "strata/prefill/xmx_moe.hpp"
#include "strata/intel_gpu.hpp"
#include "strata/ze/kernel_abi.hpp"
#include "xmx_moe_shapes.hpp"

#include <cstdlib>
#include <cstdio>

namespace strata::prefill::xmx {

// Grp's weight pointers for the STRATA_ZE_CHECK_DEVICES launch check.
inline void launch_ptrs(const Grp& g, const strata::ze::LaunchPtrs& c) {
    for (int i = 0; i < g.n && i < kMaxGroup; ++i) c(g.w[i]);
}

#include "xmx_moe.kern.inc"

namespace {

template <class C>
void launch(const strata::ze::Kernel<void(const uint16_t*, float*, Grp, int, int)>& k, strata::ze::Stream* q,
            const uint16_t* X, float* Y, const Grp& gp, int ntiles, int N, int K) {
    constexpr int BM = C::BM, BN = C::BN, KS = C::KS, WG = C::WG;
    constexpr int LDA = KS + 8, LDB = KS + 8;
    const int nb = N / BN;
#ifndef STRATA_NO_DG2_XMX
    k(q, {(uint32_t) nb, (uint32_t) ntiles, 1}, {(uint32_t) WG, 1, 1}, (size_t) (BM * LDA + BN * LDB) * sizeof(uint16_t), X,
      Y, gp, N, K);
#else
    (void) k; (void) q; (void) X; (void) Y; (void) gp; (void) ntiles; (void) N; (void) K;
    std::fprintf(stderr, "grouped XMX: the 8x8x16 / sub-group 8 kernel is not built for this device (STRATA_NO_DG2_XMX)\n");
    std::exit(1);
#endif
}

template <class C>
int fill(Grp& gp, const int32_t* cnt, int n) {
    int t = 0, o = 0;
    for (int i = 0; i < n; ++i) {
        gp.off[i] = o; gp.tstart[i] = t;
        o += cnt[i]; t += (cnt[i] + C::BM - 1) / C::BM;
    }
    gp.off[n] = o; gp.tstart[n] = t; gp.n = n;
    return t;
}
}  // namespace

int mode() {
    // The kernels are joint_matrix 8x8x16 on sub-group 8 (DG2, Arc A-series) or, in a STRATA_XMX_SG16 build, 8x16x16 on
    // sub-group 16 (Xe2, Battlemage).
    static const int v = [] {
        const char* e = std::getenv("STRATA_PF_XMX");
        const int want = e ? std::atoi(e) : 0;
        if (want == 0) return 0;
#ifdef STRATA_NO_DG2_XMX
        std::fprintf(stderr, "STRATA_PF_XMX ignored: this build is AOT for a device without the sub-group 8 XMX kernels\n");
        return 0;
#else
#ifndef STRATA_XMX_SG16
        if (strata::intel_gpu_gen(strata::ze::stream_of(nullptr)->device()) == strata::IntelGpuGen::Battlemage) {
            std::fprintf(stderr, "STRATA_PF_XMX ignored: this build has the sub-group 8 / 8x8x16 kernels (Arc A-series); Battlemage needs the AOT bmg build\n");
            return 0;
        }
#endif
        return want;
#endif
    }();
    return v;
}
bool enabled() { return mode() != 0; }

int gu_max_mean() {
    static const int v = [] { const char* e = std::getenv("STRATA_PF_XMX_GU_MAX"); return e ? std::atoi(e) : 250; }();
    return v;
}

void grouped_f16(const uint16_t* X, const uint16_t* const* W, const int32_t* cnt, int n, float* Y, int N, int K,
                 void* stream) {
    if (n <= 0) return;
    strata::ze::Stream* q = strata::ze::stream_of(stream);
    Grp gp{};
    int64_t tot = 0;
    for (int i = 0; i < n; ++i) { gp.w[i] = W[i]; tot += cnt[i]; }
    if (tot <= 0) return;
    if (tot / n < 96) {
        const int nt = fill<Small>(gp, cnt, n);
        launch<Small>(k_xmx_moe_small, q, X, Y, gp, nt, N, K);
    } else {
        const int nt = fill<Big>(gp, cnt, n);
        launch<Big>(k_xmx_moe_big, q, X, Y, gp, nt, N, K);
    }
}
}  // namespace strata::prefill::xmx
