// src/prefill/xgemm_parity.cpp - the prompt path's XMX GEMM (gemm.dp.cpp) against a double-precision CPU product, at
// the shapes the model's prompt path runs, and its speed there.
//
//   xgemm_parity            check every shape (exit 1 on a miss)
//   xgemm_parity --bench    also time every shape (TFLOP/s, weight GB/s)
//
// A product of K terms in fp32 is held to |y - ref| <= 4 K 2^-24 sum_k |x_k w_k| + 1e-30 (the fp32 accumulation bound
// with room for any summation order); inputs are rounded to fp16 / bf16 first, so the reference sees the same operands.
#include "strata/prefill/gemm.hpp"
#include "strata/ze/ze.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - exp);
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t) (sign | h);
    }
    if (exp >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = ((uint32_t) exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (uint16_t) (sign | h);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ffu, x;
    if (exp == 0) {
        if (mant == 0) x = sign;
        else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; --exp; }
            mant &= 0x3ffu;
            x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) x = sign | 0x7f800000u | (mant << 13);
    else x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    x += 0x7fffu + ((x >> 16) & 1);
    return (uint16_t) (x >> 16);
}

float bf16_to_f32(uint16_t b) {
    const uint32_t x = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

struct Case {
    const char* what;
    bool bf;
    int64_t M, N, K, ldx, ldy;
    float beta;
};

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const Case cases[] = {
        {"expert gate/up, 1 row", false, 1, 1280, 2560, 2560, 1280, 0.0f},
        {"expert gate/up, 8 rows", false, 8, 1280, 2560, 2560, 1280, 0.0f},
        {"expert gate/up, 33 rows", false, 33, 1280, 2560, 2560, 1280, 0.0f},
        {"expert gate/up, 80 rows", false, 80, 1280, 2560, 2560, 1280, 0.0f},
        {"expert gate/up, 161 rows", false, 161, 1280, 2560, 2560, 1280, 0.0f},
        {"expert gate/up, 420 rows", false, 420, 1280, 2560, 2560, 1280, 0.0f},
        {"expert down, 1 row", false, 1, 2560, 640, 640, 2560, 0.0f},
        {"expert down, 33 rows", false, 33, 2560, 640, 640, 2560, 0.0f},
        {"expert down, 80 rows", false, 80, 2560, 640, 640, 2560, 0.0f},
        {"expert down, 420 rows", false, 420, 2560, 640, 640, 2560, 0.0f},
        {"dense f16 4096 x 2560 x 2560", false, 4096, 2560, 2560, 2560, 2560, 0.0f},
        {"dense f16 4096 x 10240 x 2560", false, 4096, 10240, 2560, 2560, 10240, 0.0f},
        {"dense f16 4096 x 2560 x 10240", false, 4096, 2560, 10240, 10240, 2560, 0.0f},
        {"dense bf16 4096 x 2560 x 2560", true, 4096, 2560, 2560, 2560, 2560, 0.0f},
        {"bf16, padded x, strided y, accumulate", true, 777, 640, 2560, 2624, 1000, 1.0f},
        {"bf16 one column (shared gate)", true, 4096, 1, 2560, 2560, 1, 0.0f},
        {"bf16 narrow 5 columns", true, 300, 5, 640, 640, 8, 1.0f},
        {"f16 odd sizes", false, 37, 200, 72, 72, 200, 0.0f},
    };
    strata::prefill::Gemm gemm;
    std::string err;
    strata::ze::Stream& s = strata::ze::default_stream();
    if (!gemm.init(&s, 0, err)) {
        std::fprintf(stderr, "xgemm_parity: %s\n", err.c_str());
        return 2;
    }
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    int bad_cases = 0;
    for (const Case& c : cases) {
        std::vector<uint16_t> X((size_t) c.M * c.ldx, 0), W((size_t) c.N * c.K);
        std::vector<float> Y0((size_t) c.M * c.ldy, 0.0f);
        for (int64_t m = 0; m < c.M; ++m)
            for (int64_t k = 0; k < c.K; ++k) {
                const float v = uni(rng);
                X[(size_t) m * c.ldx + k] = c.bf ? f32_to_bf16(v) : f32_to_f16(v);
            }
        for (auto& w : W) {
            const float v = uni(rng) * 0.05f;
            w = c.bf ? f32_to_bf16(v) : f32_to_f16(v);
        }
        for (auto& y : Y0) y = c.beta != 0.0f ? uni(rng) : -7.0f;
        auto* dX = (uint16_t*) strata::ze::malloc_device(X.size() * 2);
        auto* dW = (uint16_t*) strata::ze::malloc_device(W.size() * 2);
        auto* dY = (float*) strata::ze::malloc_device(Y0.size() * 4);
        s.memcpy(dX, X.data(), X.size() * 2);
        s.memcpy(dW, W.data(), W.size() * 2);
        s.memcpy(dY, Y0.data(), Y0.size() * 4);
        if (c.bf) gemm.bf16(dX, dW, dY, c.M, c.N, c.K, c.ldy, c.beta, c.ldx);
        else gemm.f16(dX, dW, dY, c.M, c.N, c.K, c.ldy, c.beta);
        std::vector<float> Y(Y0.size());
        s.memcpy(Y.data(), dY, Y.size() * 4);
        s.sync();
        auto cvt = [&](uint16_t v) { return c.bf ? bf16_to_f32(v) : f16_to_f32(v); };
        int64_t bad = 0, nonfinite = 0, untouched = 0;
        double worst = 0.0;
        for (int64_t m = 0; m < c.M; ++m)
            for (int64_t n = 0; n < c.ldy; ++n) {
                const float got = Y[(size_t) m * c.ldy + n];
                if (n >= c.N) {
                    untouched += got != Y0[(size_t) m * c.ldy + n];
                    continue;
                }
                double ref = c.beta != 0.0f ? (double) Y0[(size_t) m * c.ldy + n] : 0.0, mag = std::fabs(ref);
                for (int64_t k = 0; k < c.K; ++k) {
                    const double p = (double) cvt(X[(size_t) m * c.ldx + k]) * (double) cvt(W[(size_t) n * c.K + k]);
                    ref += p;
                    mag += std::fabs(p);
                }
                if (!std::isfinite(got)) { ++nonfinite; continue; }
                const double tol = 4.0 * (double) (c.K + 1) * std::ldexp(1.0, -24) * mag + 1e-30;
                const double e = std::fabs((double) got - ref);
                if (e > tol) ++bad;
                if (mag > 0) worst = std::max(worst, e / mag);
            }
        const bool ok = bad == 0 && nonfinite == 0 && untouched == 0;
        bad_cases += !ok;
        std::printf("%-40s M %5lld N %6lld K %6lld  %s  (worst |err|/sum|xw| %.2e, out of tolerance %lld, non-finite %lld, "
                    "written past N %lld)",
                    c.what, (long long) c.M, (long long) c.N, (long long) c.K, ok ? "PASS" : "FAIL", worst, (long long) bad,
                    (long long) nonfinite, (long long) untouched);
        if (bench) {
            strata::ze::Event e0(true), e1(true);
            const int reps = c.M * c.N * c.K < 100000000ll ? 200 : 20;
            for (int r = 0; r < 3; ++r) {
                if (c.bf) gemm.bf16(dX, dW, dY, c.M, c.N, c.K, c.ldy, c.beta, c.ldx);
                else gemm.f16(dX, dW, dY, c.M, c.N, c.K, c.ldy, c.beta);
            }
            s.record(e0);
            for (int r = 0; r < reps; ++r) {
                if (c.bf) gemm.bf16(dX, dW, dY, c.M, c.N, c.K, c.ldy, c.beta, c.ldx);
                else gemm.f16(dX, dW, dY, c.M, c.N, c.K, c.ldy, c.beta);
            }
            s.record(e1);
            e1.sync();
            const double us = 1000.0 * e1.elapsed_ms_since(e0) / reps;
            std::printf("  %9.1f us  %7.2f TFLOP/s  %7.1f GB/s of W", us, 2.0 * c.M * c.N * c.K / us / 1e6,
                        2.0 * c.N * c.K / us / 1e3);
        }
        std::printf("\n");
        strata::ze::free(dX);
        strata::ze::free(dW);
        strata::ze::free(dY);
    }
    std::printf("xgemm_parity: %d of %zu cases failed\n", bad_cases, sizeof cases / sizeof cases[0]);
    return bad_cases == 0 ? 0 : 1;
}
