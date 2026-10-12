// src/kernels/cuda/qsa_prompt_attn.host.cpp - the host side of qsa_prompt_attn.dp.cpp: kernel choice and launches.
#include "strata/core/emulate.hpp"
#include <cstdlib>
#include <cstring>
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_prompt_attn_xmx.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cmath>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace strata::kernels {

#include "qsa_prompt_attn.kern.inc"

namespace {

constexpr int HD = 256;           // head_dim
constexpr int G = 12;             // query heads per KV head
#ifndef D1_CH
#define D1_CH 32
#endif
constexpr int CH = D1_CH;         // cells per chunk
constexpr int THREADS = 128;      // 4 warps: scores by cell (8 each), p.v by dimension (64 each = one int8 scale group)
constexpr int QS = HD + 8;        // q row stride in halves (bank-conflict-free fragment loads)

// The work-group local memory of prompt_attn_kernel (Smem) and prompt_attn_i8_kernel (Smem2) in qsa_prompt_attn.dp.cpp,
// for the launch size: the same members, with each half stored as uint16_t (same size and alignment).
template <int KV_MODE>
struct Smem {
    using KElem = typename std::conditional<KV_MODE == 0, uint16_t, int8_t>::type;
    using VElem = typename std::conditional<KV_MODE == 1 || KV_MODE == 4, int8_t, uint16_t>::type;
    static constexpr int KROW = KV_MODE == 0 ? HD + 8 : HD + 16;   // elements; 16-byte aligned rows, banks spread
    static constexpr int VROW = (KV_MODE == 1 || KV_MODE == 4) ? HD + 16 : HD + 8;
    static constexpr int NG = KV_MODE == 4 ? HD / QK4_0 : 4;      // scale groups per row (q4_0: 8 of 32, else 4 of 64)
    uint16_t qh[16][QS];
    uint16_t ql[16][QS];
    KElem k[CH][KROW];
    VElem v[CH][VROW];
    float ks[CH][NG];
    float vs[CH][NG];
    float s[16][CH + 1];
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    long long row[CH];
};

constexpr int CH2 = 32;

struct Smem2 {
    int8_t kv[2][4][2][CH2][64];   // stage, warp, K/V, cell, 64 dims in 16-byte pieces XOR-swizzled by the cell
    float sc[2][4][2][CH2];        // stage, warp, K/V scale of the cell for the warp's group
    float part[4][16][CH2 + 1];    // q.k per dim group
    float p[16][CH2 + 1];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

bool launch_i8(const float *q, const QsaAttnPools &pools, const int32_t *ids,
               const int32_t *steps, int64_t cap, const QsaShapes &s,
               float *attn, int64_t n_q, strata::ze::Stream* st) {
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs this on several)
    int dev = 0;
    dev = strata::ze::current_device();
    const int bytes = (int)sizeof(Smem2);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (0 != 0) {
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        k_qsa_prompt_attn_i8(st, {(unsigned) nb, (unsigned) s.n_head_kv, 1}, {THREADS, 1, 1}, bytes, q + q0 * s.n_head * HD,
                             pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv, (int) s.page_size,
                             scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    return true;
}

template <int KV_MODE>
const auto& prompt_kernel() {
    if constexpr (KV_MODE == 0) return k_qsa_prompt_attn_m0;
    else if constexpr (KV_MODE == 1) return k_qsa_prompt_attn_m1;
    else if constexpr (KV_MODE == 3) return k_qsa_prompt_attn_m3;
    else return k_qsa_prompt_attn_m4;
}

template <int KV_MODE>
bool launch(const float *q, const QsaAttnPools &pools, const int32_t *ids,
            const int32_t *steps, int64_t cap, const QsaShapes &s, float *attn,
            int64_t n_q, strata::ze::Stream* st) {
    static bool attr[64] = {};   // per device, as above
    int dev = 0;
    dev = strata::ze::current_device();
    const int bytes = (int)sizeof(Smem<KV_MODE>);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (0 != 0) {
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_kernel<KV_MODE>()(st, {(unsigned) nb, (unsigned) s.n_head_kv, 1}, {THREADS, 1, 1}, bytes,
                                 q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount,
                                 (int) s.n_head_kv, (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    return true;
}

#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
template <int KV_MODE>
bool launch70(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
              const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem70<KV_MODE>);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_v70_kernel<KV_MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_v70_kernel<KV_MODE><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (sm_70): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
#else   // AMD, or a build without STRATA_EXPERIMENTAL_SM60: no m8n8k4 kernel (the caller keeps the old one)
template <int KV_MODE>
bool launch70(const float *, const QsaAttnPools &, const int32_t *,
              const int32_t *, int64_t, const QsaShapes &, float *, int64_t,
              strata::ze::Stream*) {
    return false;
}
#endif  // !__HIPCC__ && STRATA_EXPERIMENTAL_SM60

#if defined(__HIPCC__)
// gfx12 (RDNA4) and gfx11 (RDNA3 / RDNA3.5), only on request: the output differs from the default kernel's in its
// last bits
bool hip_wmma_usable() {
    static const bool want = [] {
        const char* e = std::getenv("STRATA_HIP_WMMA");
        return e != nullptr && e[0] == '1';
    }();
    if (!want) return false;
    static int arch[64] = {};   // 0 unknown, 1 gfx11 / gfx12, 2 other
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) { (void) hipGetLastError(); return false; }
    if (arch[dev] == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
        arch[dev] = std::strncmp(prop.gcnArchName, "gfx12", 5) == 0 || strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName)
                        ? 1 : 2;
        static bool told = false;
        if (!told) {
            told = true;
            std::fprintf(stderr, "strata: STRATA_HIP_WMMA: the prompt attention on matrix cores %s (%s)\n",
                         arch[dev] == 1 ? "on" : "unavailable", prop.gcnArchName);
        }
    }
    return arch[dev] == 1;
}

bool launch_wmma(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                 const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        static const bool fast = [] { const char* e = std::getenv("STRATA_PA_FAST"); return e != nullptr && e[0] == '1'; }();
        static const int64_t min_t = [] { const char* e = std::getenv("STRATA_PF_SWITCH_MIN_T"); return e ? (int64_t) std::atoll(e) : (int64_t) 0; }();
        if (fast && n_q >= min_t)
            prompt_attn_wmma_kernel<false><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, 0, st>>>(
                q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
                (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
        else
            prompt_attn_wmma_kernel<true><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, 0, st>>>(
                q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
                (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch (wmma): %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
#endif

}  // namespace

bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    bool volta = false, turing = false;   // per call, from the CURRENT device (a layer split can mix Turing with newer cards)
    {   // sm_75 or newer: the MMA above compiles for both.  sm_80+ runs the cp.async kernel (launch_i8); Turing has
        // no cp.async, so it runs the v1 kernel (launch<1>, same accuracy, another summation order).  An older card
        // keeps the old kernel.
#if defined(STRATA_HIP_GFX906)
        return false;   // gfx906: the tensor-core and WMMA kernels are not for it; the FP32 kernel runs
#endif
        // SYCL: the mma.sync kernel below is not ported (upstream's Turing/sm_80 split - `turing` - does not
        // apply); its XMX port takes the same arguments
        if (qsa_prompt_attn_xmx(q, pools, ids, steps, cap, s, attn, n_q, stream)) return true;
        return false;   // the caller takes the older kernel
    }
#if defined(__HIPCC__)
    // the tensor-core kernels are compiled out on AMD (its major version is not a CUDA sm); RDNA4 has its own int8-KV
    // matrix-core kernel, opt-in (STRATA_HIP_WMMA=1); everything else keeps the old kernel
    (void) turing; (void) volta;
    if (pools.k_q != nullptr && pools.v_q != nullptr && pools.k_scale != nullptr && pools.v_scale != nullptr &&
        pools.k_q4 == nullptr && pools.v_q4 == nullptr && s.head_dim == HD && s.n_head == (int64_t) G * s.n_head_kv &&
        cap > 0 && ids && steps && pools.page_table && hip_wmma_usable())
        return launch_wmma(q, pools, ids, steps, cap, s, attn, n_q, (cudaStream_t) stream);
    return false;
#endif
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps || !pools.page_table)
        return false;
    strata::ze::Stream* st = strata::ze::stream_of(stream);
    if (pools.k_q4 != nullptr) {   // Q4_0 K and V (--kv q4_0): mode 4.  STRATA_PROMPT_ATTN_Q4=0: the old kernel (A/B)
        static const bool q4_off = [] {
            const char* v = std::getenv("STRATA_PROMPT_ATTN_Q4");
            return v != nullptr && v[0] == '0';
        }();
        // sm_80+ only: on Turing mode 4 would run as pairs of m16n8k8 MMAs, which no parity run has checked yet
        if (q4_off || turing || pools.v_q4 == nullptr) return false;
        return launch<4>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {   // hybrid K8V4: int8 K + dequantized-q4 V
        if (!pools.k_scale) return false;
        if (volta) return launch70<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
        return launch<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        // STRATA_PROMPT_ATTN_V1=1 (debug): the first version, same accuracy, another summation order - the control
        // for how far the model amplifies an FP32-level change.  Turing always takes it: v2's cp.async does not
        // exist before sm_80.
        static const bool v1 = std::getenv("STRATA_PROMPT_ATTN_V1") != nullptr;
        if (volta) return launch70<1>(q, pools, ids, steps, cap, s, attn, n_q, st);
        if (v1 || turing) return launch<1>(q, pools, ids, steps, cap, s, attn, n_q, st);
        return launch_i8(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    if (volta) return launch70<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
    return launch<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
}

}  // namespace strata::kernels