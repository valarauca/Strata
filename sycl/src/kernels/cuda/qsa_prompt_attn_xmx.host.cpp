// src/kernels/cuda/qsa_prompt_attn_xmx.host.cpp - the host side of qsa_prompt_attn_xmx.dp.cpp: the dispatch and the launches.
#include "strata/kernels/qsa_prompt_attn_xmx.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/ze/kernel_abi.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "qsa_prompt_attn_xmx.kern.inc"

namespace {
using qsa_xmx::HD;
using qsa_xmx::G;
using qsa_xmx::THREADS;
using qsa_xmx::Layout;

template <int KV_MODE, int CH>
const auto& xmx_kernel() {
    if constexpr (CH == 64)
        return KV_MODE == 0 ? k_qsa_prompt_attn_xmx_m0_c64 : KV_MODE == 1 ? k_qsa_prompt_attn_xmx_m1_c64 : k_qsa_prompt_attn_xmx_m3_c64;
    else
        return KV_MODE == 0 ? k_qsa_prompt_attn_xmx_m0_c32 : KV_MODE == 1 ? k_qsa_prompt_attn_xmx_m1_c32 : k_qsa_prompt_attn_xmx_m3_c32;
}

template <int KV_MODE, int CH>
bool launch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
            const QsaShapes& s, float* attn, int64_t n_q, strata::ze::Stream* st) {
    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    constexpr size_t bytes = Layout<CH>::bytes;
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        const float* qq = q + q0 * s.n_head * HD;
        const int32_t* idq = ids + q0 * cap;
        const int32_t* stq = steps + q0 * kStepCount;
        float* aq = attn + q0 * s.n_head * HD;
        const int nkv = (int) s.n_head_kv, ps = (int) s.page_size, capi = (int) cap;
        xmx_kernel<KV_MODE, CH>()(st, {(uint32_t) nb, (uint32_t) nkv, 1}, {THREADS, 1, 1}, bytes, qq, pools, idq, stq, nkv,
                                  ps, scale_log2, aq, capi);
    }
    return true;
}

template <int CH>
bool dispatch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
              const QsaShapes& s, float* attn, int64_t n_q, strata::ze::Stream* st) {
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {
        if (!pools.k_scale) return false;
        return launch<3, CH>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
        return launch<1, CH>(q, pools, ids, steps, cap, s, attn, n_q, st);
    }
    if (!pools.k_pool || !pools.v_pool) return false;
    return launch<0, CH>(q, pools, ids, steps, cap, s, attn, n_q, st);
}
}  // namespace

bool qsa_prompt_attn_xmx(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                         const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    // STRATA_PROMPT_ATTN_XMX=1 (64-cell chunks) or =32 (32-cell chunks); unset or 0: the caller's fallback
    static const int mode = [] { const char* v = std::getenv("STRATA_PROMPT_ATTN_XMX"); return v ? std::atoi(v) : 0; }();
    if (mode == 0 || n_q <= 0) return false;
    if (pools.k_q4 != nullptr || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps ||
        !pools.page_table)
        return false;
    strata::ze::Stream* st = strata::ze::stream_of(stream);
    const size_t have = strata::ze::device_info(st->device()).max_slm;
    const bool ch64 = mode != 32 && have >= Layout<64>::bytes;
    static bool told = false;
    if (!told) {
        told = true;
        std::fprintf(stderr, "strata: prompt attention on XMX, %d-cell chunks (%zu of %zu bytes of local memory)\n",
                     ch64 ? 64 : 32, ch64 ? Layout<64>::bytes : Layout<32>::bytes, have);
    }
    if (!ch64 && have < Layout<32>::bytes) return false;
    return ch64 ? dispatch<64>(q, pools, ids, steps, cap, s, attn, n_q, st)
                : dispatch<32>(q, pools, ids, steps, cap, s, attn, n_q, st);
}
}  // namespace strata::kernels
