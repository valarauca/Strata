// src/kernels/cuda/router_top10.host.cpp - the host side of router_top10.dp.cpp: the launchers.
#include "strata/sycl_doorbell.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "router_top10.kern.inc"

namespace {

constexpr int RT_MAX_THREADS = 512;

template <bool FORCE_SERIAL>
void launch_sycl_fast(ze::Stream& q, const float* logits, int n_tokens, int n_expert, int k,
                      int* ids, float* weights) {
    const int threads = (std::min(n_expert, RT_MAX_THREADS) + 31) & ~31;
    const size_t slm = (size_t) n_expert * sizeof(double) + (RT_MAX_THREADS / 32) * (sizeof(double) + sizeof(float)) +
                       sizeof(float);
    if constexpr (FORCE_SERIAL)
        k_router_top10_fast_serial(&q, {(unsigned) n_tokens, 1, 1}, {(unsigned) threads, 1, 1}, slm, logits, n_expert, k,
                                   ids, weights);
    else
        k_router_top10_fast(&q, {(unsigned) n_tokens, 1, 1}, {(unsigned) threads, 1, 1}, slm, logits, n_expert, k, ids,
                            weights);
}

#if defined(__HIPCC__)
void launch_generic(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights, void* stream,
                    int mode) {   // mode 0: router_top10_kernel, 1: the HIP fast kernel, 2: the fast kernel's serial path
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + 31) & ~31;
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    const size_t smem = taken_bytes + (size_t) n_expert * sizeof(double) + (size_t) n_expert * sizeof(float);
    if (mode == 1) {
        router_top10_fast_kernel<false><<<(unsigned) n_tokens, threads, smem, (cudaStream_t) stream>>>(
            logits, n_tokens, n_expert, k, ids, weights);
        return;
    }
    if (mode == 2) {
        router_top10_fast_kernel<true><<<(unsigned) n_tokens, threads, smem, (cudaStream_t) stream>>>(
            logits, n_tokens, n_expert, k, ids, weights);
        return;
    }
    router_top10_kernel<<<(unsigned) n_tokens, threads, smem, (cudaStream_t) stream>>>(
        logits, n_tokens, n_expert, k, ids, weights);
}
#endif

}  // namespace

bool router_top10_variant(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                          void* stream, int variant) {
#if defined(__HIPCC__)
    if (n_tokens <= 0 || n_expert <= 64 || n_expert > 512 || k <= 0 || k > 32) return false;
    if (variant < 0 || variant > 2) return false;
    launch_generic(logits, n_tokens, n_expert, k, ids, weights, stream, variant);
    return cudaGetLastError() == cudaSuccess;
#else
    if (n_tokens <= 0 || n_expert <= 64 || n_expert > 512 || k <= 0 || k > 32 || variant < 0 || variant > 2)
        return false;
    auto& q = *ze::stream_of(stream);
    if (variant == 1) launch_sycl_fast<false>(q, logits, n_tokens, n_expert, k, ids, weights);
    else if (variant == 2) launch_sycl_fast<true>(q, logits, n_tokens, n_expert, k, ids, weights);
    else {
        const int threads = (std::min(n_expert, RT_MAX_THREADS) + 31) & ~31;
        const size_t taken_bytes = ((size_t)n_expert + 15u) & ~size_t(15);
        const size_t smem = taken_bytes + n_expert * (sizeof(double) + sizeof(float));
        k_router_top10(&q, {(unsigned) n_tokens, 1, 1}, {(unsigned) threads, 1, 1}, smem, logits, n_tokens, n_expert, k,
                       ids, weights);
    }
    if (stream == nullptr) q.sync();
    return true;
#endif
}

void router_top10(const float *logits, int n_tokens, int n_expert, int k,
                  int *ids, float *weights, void *stream) {
    if (strata::a770_fast() && n_tokens > 0 && n_expert > 64 && n_expert <= 512 && k > 0 && k <= 32) {
        auto& q = *ze::stream_of(stream);
        launch_sycl_fast<false>(q, logits, n_tokens, n_expert, k, ids, weights);
        if (stream == nullptr) q.sync();
        return;
    }
#if defined(__HIPCC__)
    {
#if defined(STRATA_HIP_GFX906)
        // gfx906 (wave64): the one-wavefront kernel below stays the default; the S6 kernel is wave32-shaped and only
        // runs here on request (STRATA_HIP_ROUTER_FAST=1) until it is measured on this card
        static const bool old = std::getenv("STRATA_HIP_ROUTER_FAST") == nullptr;
#else
        static const bool old = std::getenv("STRATA_HIP_ROUTER_OLD") != nullptr;
#endif
        if (!old && n_tokens > 0 && n_expert > 64 && n_expert <= 512 && k > 0 && k <= 32) {
            launch_generic(logits, n_tokens, n_expert, k, ids, weights, stream, 1);
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                std::fprintf(stderr, "router_top10 launch: %s\n", cudaGetErrorString(e));
                std::exit(1);
            }
            if (stream == nullptr) {
                const cudaError_t s = cudaDeviceSynchronize();
                if (s != cudaSuccess) {
                    std::fprintf(stderr, "router_top10: %s\n", cudaGetErrorString(s));
                    std::exit(1);
                }
            }
            return;
        }
    }
#endif
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 64) {
        std::fprintf(stderr, "router_top10: k %d exceeds the kernel's 64\n", k);
        std::exit(1);
    }
    if (n_expert > RT_MAX_THREADS * 64) {
        std::fprintf(stderr, "router_top10: n_expert %d is past the kernel's %d\n", n_expert,
                     RT_MAX_THREADS * 64);
        std::exit(1);
    }
#if defined(STRATA_HIP_GFX906)
    if (n_expert <= 64 * RW_PER && !std::getenv("STRATA_ROUTER_BLOCK")) {
        router_top10_wave_kernel<<<(unsigned) n_tokens, 64, 0, (cudaStream_t) stream>>>(logits, n_tokens, n_expert, k,
                                                                                       ids, weights);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "router_top10 launch: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
        if (stream == nullptr && cudaDeviceSynchronize() != cudaSuccess) std::exit(1);
        return;
    }
#endif
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + 31) & ~31;                  // at least one full warp, for the reductions
    // the selection's taken-mask, then n_expert doubles for the exponentials, then n_expert floats for the
    // probabilities.  The mask is first so the two aligned arrays need no offset parameter.
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    const size_t smem =
        taken_bytes + (size_t)n_expert * sizeof(double) +
        (size_t)n_expert * sizeof(float);
    k_router_top10(ze::stream_of(stream), {(unsigned) n_tokens, 1, 1}, {(unsigned) threads, 1, 1}, smem, logits, n_tokens,
                   n_expert, k, ids, weights);

    if (stream == nullptr) ze::default_stream().sync();
}

}  // namespace strata::kernels
