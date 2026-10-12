// src/kernels/gr_multi_parity.cpp - the multi-token hyper-connection read and write (#783 PR-g: gr_read_multi,
// gr_write_multi and the exact-T BF16 projections under them) against the single-token calls, memcmp.
//
// Real geometry (n_embd 2560, hc 4, hc_lr 320), the pinned native MMVF path, every T = 1..8, with and without an
// injection weight (a layer read vs the final mixer), and the write both out of place and in place (R_out == R).
// Each output of the multi call must equal the token-by-token one bit for bit. GPU, synthetic, no model.
#include "strata/ze/ze.hpp"
#include "strata/kernels/gr.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
/// cudaMemcpy: after the device's queued work, complete on return.
void memcpy_sync(void* dst, const void* src, size_t bytes) {
    strata::ze::device_sync();
    strata::ze::default_stream().memcpy(dst, src, bytes);
    strata::ze::default_stream().sync();
}

/// cudaMemset: after the device's queued work, complete on return.
void memset_sync(void* dst, int value, size_t bytes) {
    strata::ze::device_sync();
    strata::ze::default_stream().memset(dst, value, bytes);
    strata::ze::default_stream().sync();
}

template <typename T>
T* up(const std::vector<T>& h) {
    void* p = nullptr;
    p = strata::ze::malloc_device(h.size() * sizeof(T) + 64);
    memcpy_sync(p, h.data(), h.size() * sizeof(T));
    return (T*) p;
}
template <typename T>
T* dev(size_t n) {
    void* p = nullptr;
    p = strata::ze::malloc_device(n * sizeof(T) + 64);
    memset_sync(p, 0, n * sizeof(T) + 64);
    return (T*) p;
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    memcpy_sync(h.data(), d, n * sizeof(T));
    return h;
}
uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}
}  // namespace

int main() {
    const long long n_embd = 2560, hc = 4, hc_lr = 320, hc_dim = hc * n_embd;
    const float eps = 1e-6f;
    std::mt19937 rng(783);
    std::normal_distribution<float> gauss(0.f, 1.f);
    k::gr_set_native_mmvf(true);

    std::vector<float> w_norm((size_t) hc_dim);
    for (auto& x : w_norm) x = 1.0f + 0.1f * gauss(rng);
    std::vector<uint16_t> w_down((size_t) (hc_lr * hc_dim)), w_up((size_t) (hc_dim * hc_lr)), w_inj((size_t) (hc * hc_dim));
    for (auto& x : w_down) x = bf16_of(gauss(rng) * 0.05f);
    for (auto& x : w_up) x = bf16_of(gauss(rng) * 0.05f);
    for (auto& x : w_inj) x = bf16_of(gauss(rng) * 0.02f);
    float* d_norm = up(w_norm);
    uint16_t *d_down = up(w_down), *d_up = up(w_up), *d_inj = up(w_inj);

    const k::GrShapes sh{n_embd, hc, hc_lr};
    void* ws_raw = nullptr;
    ws_raw = strata::ze::malloc_device(k::gr_workspace_bytes(sh));
    k::GrWorkspace ws;
    k::gr_workspace_init(sh, ws_raw, ws);

    int cases = 0, bad = 0;
    for (int T = 1; T <= 8; ++T) {
        std::vector<float> R((size_t) T * hc_dim);
        for (int t = 0; t < T; ++t)
            for (long long c = 0; c < hc; ++c)
                for (long long d = 0; d < n_embd; ++d) R[(size_t) t * hc_dim + c * n_embd + d] = gauss(rng) * (float) (1 << (2 * c));
        float* d_R = up(R);
        float* d_xn = dev<float>((size_t) T * hc_dim);
        float* d_lo = dev<float>((size_t) T * hc_lr);
        float* d_g = dev<float>((size_t) T * hc_dim);
        for (int with_inj = 0; with_inj < 2; ++with_inj) {
            float* d_mixed_m = dev<float>((size_t) T * n_embd);
            float* d_mixed_s = dev<float>((size_t) T * n_embd);
            float* d_inject_m = dev<float>((size_t) T * hc);
            float* d_inject_s = dev<float>((size_t) T * hc);
            const uint16_t* wi = with_inj ? d_inj : nullptr;
            k::gr_read_multi(d_R, d_norm, d_down, d_up, wi, eps, sh, ws, d_xn, d_lo, d_g, d_mixed_m,
                             with_inj ? d_inject_m : nullptr, T, nullptr);
            for (int t = 0; t < T; ++t)
                k::gr_read(d_R + (size_t) t * hc_dim, d_norm, d_down, d_up, wi, eps, sh, ws, d_mixed_s + (size_t) t * n_embd,
                           with_inj ? d_inject_s + (size_t) t * hc : nullptr, nullptr);
            strata::ze::device_sync();
            bool ok = std::memcmp(down(d_mixed_m, (size_t) T * n_embd).data(), down(d_mixed_s, (size_t) T * n_embd).data(),
                                  (size_t) T * n_embd * 4) == 0;
            if (with_inj)
                ok = ok && std::memcmp(down(d_inject_m, (size_t) T * hc).data(), down(d_inject_s, (size_t) T * hc).data(),
                                       (size_t) T * hc * 4) == 0;
            ++cases;
            if (!ok) {
                std::printf("    *** gr_read_multi T=%d %s differs from gr_read ***\n", T, with_inj ? "with injection" : "final mixer");
                ++bad;
            }
            strata::ze::free(d_mixed_m);
            strata::ze::free(d_mixed_s);
            strata::ze::free(d_inject_m);
            strata::ze::free(d_inject_s);
        }

        // the write: R_out[t] = R[t] + block_out[t] * 2 sigmoid(inject[t] / hc)
        std::vector<float> bo((size_t) T * n_embd), inj((size_t) T * hc);
        for (auto& x : bo) x = gauss(rng);
        for (auto& x : inj) x = gauss(rng) * 3.0f;
        float* d_bo = up(bo);
        float* d_in = up(inj);
        for (int inplace = 0; inplace < 2; ++inplace) {
            float* d_Rm = up(R);
            float* d_Rs = up(R);
            float* d_om = inplace ? d_Rm : dev<float>((size_t) T * hc_dim);
            float* d_os = inplace ? d_Rs : dev<float>((size_t) T * hc_dim);
            k::gr_write_multi(d_Rm, d_bo, d_in, sh, d_om, T, nullptr);
            for (int t = 0; t < T; ++t)
                k::gr_write(d_Rs + (size_t) t * hc_dim, d_bo + (size_t) t * n_embd, d_in + (size_t) t * hc, sh,
                            d_os + (size_t) t * hc_dim, nullptr);
            strata::ze::device_sync();
            const bool ok = std::memcmp(down(d_om, (size_t) T * hc_dim).data(), down(d_os, (size_t) T * hc_dim).data(),
                                        (size_t) T * hc_dim * 4) == 0;
            ++cases;
            if (!ok) {
                std::printf("    *** gr_write_multi T=%d %s differs from gr_write ***\n", T, inplace ? "in place" : "out of place");
                ++bad;
            }
            strata::ze::free(d_Rm);
            strata::ze::free(d_Rs);
            if (!inplace) {
                strata::ze::free(d_om);
                strata::ze::free(d_os);
            }
        }
        strata::ze::free(d_R);
        strata::ze::free(d_xn);
        strata::ze::free(d_lo);
        strata::ze::free(d_g);
        strata::ze::free(d_bo);
        strata::ze::free(d_in);
    }
    std::printf("gr_multi: %d cases, %d failures\n", cases, bad);
    if (bad) return 1;
    std::printf("gr_multi_parity OK\n");
    return 0;
}
