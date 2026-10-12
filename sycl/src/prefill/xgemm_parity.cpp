// src/prefill/xgemm_parity.cpp - the prompt path's GEMMs against a double-precision CPU product, at the shapes the
// model's prompt path runs, and their speed there: the XMX GEMM of gemm.dp.cpp and the gemmstone kernels
// (strata/gemmstone.hpp) on the same fp16 / bf16 cases, and gemmstone on Q4_0 weights repacked by q4_repack.
//
//   xgemm_parity            check every shape (exit 1 on a miss)
//   xgemm_parity --bench    also time every shape (TFLOP/s, weight GB/s)
//
// A product of K terms in fp32 is held to |y - ref| <= 4 K 2^-24 sum_k |x_k w_k| + 1e-30 (the fp32 accumulation bound
// with room for any summation order); inputs are rounded to fp16 / bf16 first, so the reference sees the same operands.
// Q4_0 weights get 2^-10 sum_k |x_k w_k| more, for a (q - 8) d product rounded to fp16 before the multiply.
#include "strata/gemmstone.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/q4pack.hpp"
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

struct Q4Case {
    const char* what;
    int64_t M, N, K;
    bool interleave;
};

struct Check {
    int64_t bad = 0, nonfinite = 0, untouched = 0;
    double worst = 0.0;
    bool ok() const { return bad == 0 && nonfinite == 0 && untouched == 0; }
};

void report(const char* what, const char* path, int64_t M, int64_t N, int64_t K, const Check& c) {
    std::printf("%-40s %-4s M %5lld N %6lld K %6lld  %s  (worst |err|/sum|xw| %.2e, out of tolerance %lld, non-finite %lld, "
                "written past N %lld)",
                what, path, (long long) M, (long long) N, (long long) K, c.ok() ? "PASS" : "FAIL", c.worst, (long long) c.bad,
                (long long) c.nonfinite, (long long) c.untouched);
}

template <class F>
double time_us(strata::ze::Stream& s, int reps, F&& f) {
    strata::ze::Event e0(true), e1(true);
    for (int r = 0; r < 3; ++r) f();
    s.record(e0);
    for (int r = 0; r < reps; ++r) f();
    s.record(e1);
    e1.sync();
    return 1000.0 * e1.elapsed_ms_since(e0) / reps;
}

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
    int bad_cases = 0, n_checks = 0;
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
        auto cvt = [&](uint16_t v) { return c.bf ? bf16_to_f32(v) : f16_to_f32(v); };
        auto xmx = [&] {
            if (c.bf) gemm.bf16(dX, dW, dY, c.M, c.N, c.K, c.ldy, c.beta, c.ldx);
            else gemm.f16(dX, dW, dY, c.M, c.N, c.K, c.ldy, c.beta);
            return true;
        };
        auto gs = [&] {
            return strata::gs::gemm(c.bf ? strata::gs::Weights::BF16 : strata::gs::Weights::F16, dX, c.ldx, dW, c.K, nullptr,
                                    dY, c.ldy, c.M, c.N, c.K, c.beta != 0.0f, &s);
        };
        for (int path = 0; path < 2; ++path) {
            s.memcpy(dY, Y0.data(), Y0.size() * 4);
            const bool ran = path == 0 ? xmx() : gs();
            std::vector<float> Y(Y0.size());
            s.memcpy(Y.data(), dY, Y.size() * 4);
            s.sync();
            Check ck;
            if (!ran) ck.bad = 1;
            for (int64_t m = 0; m < c.M && ran; ++m)
                for (int64_t n = 0; n < c.ldy; ++n) {
                    const float got = Y[(size_t) m * c.ldy + n];
                    if (n >= c.N) {
                        ck.untouched += got != Y0[(size_t) m * c.ldy + n];
                        continue;
                    }
                    double ref = c.beta != 0.0f ? (double) Y0[(size_t) m * c.ldy + n] : 0.0, mag = std::fabs(ref);
                    for (int64_t k = 0; k < c.K; ++k) {
                        const double p = (double) cvt(X[(size_t) m * c.ldx + k]) * (double) cvt(W[(size_t) n * c.K + k]);
                        ref += p;
                        mag += std::fabs(p);
                    }
                    if (!std::isfinite(got)) { ++ck.nonfinite; continue; }
                    const double tol = 4.0 * (double) (c.K + 1) * std::ldexp(1.0, -24) * mag + 1e-30;
                    const double e = std::fabs((double) got - ref);
                    if (e > tol) ++ck.bad;
                    if (mag > 0) ck.worst = std::max(ck.worst, e / mag);
                }
            bad_cases += !ck.ok();
            ++n_checks;
            report(c.what, path == 0 ? "xmx" : "gs", c.M, c.N, c.K, ck);
            if (!ran) std::printf("  no gemmstone kernel");
            if (bench && ran) {
                const int reps = c.M * c.N * c.K < 100000000ll ? 200 : 20;
                const double us = path == 0 ? time_us(s, reps, xmx) : time_us(s, reps, gs);
                std::printf("  %9.1f us  %7.2f TFLOP/s  %7.1f GB/s of W", us, 2.0 * c.M * c.N * c.K / us / 1e6,
                            2.0 * c.N * c.K / us / 1e3);
            }
            std::printf("\n");
        }
        strata::ze::free(dX);
        strata::ze::free(dW);
        strata::ze::free(dY);
    }
    const Q4Case q4cases[] = {
        {"q4 expert gate/up, 1 row", 1, 1280, 2560, false},
        {"q4 expert gate/up, 33 rows", 33, 1280, 2560, false},
        {"q4 expert gate/up, 161 rows", 161, 1280, 2560, false},
        {"q4 expert gate/up, 420 rows", 420, 1280, 2560, false},
        {"q4 expert down, 161 rows", 161, 2560, 640, false},
        {"q4 expert down, 420 rows", 420, 2560, 640, false},
        {"q4 gate/up interleaved, 80 rows", 80, 1280, 2560, true},
    };
    for (const Q4Case& c : q4cases) {
        const int64_t nb = c.K / 32, rows_src = c.interleave ? c.N / 2 : c.N, nsrc = c.interleave ? 2 : 1;
        std::vector<std::vector<uint8_t>> B(nsrc, std::vector<uint8_t>((size_t) rows_src * nb * 18));
        for (auto& blk : B)
            for (int64_t i = 0; i < rows_src * nb; ++i) {
                const uint16_t d = f32_to_f16(0.002f + 0.01f * (0.5f + 0.5f * uni(rng)));
                blk[(size_t) i * 18] = (uint8_t) (d & 0xff);
                blk[(size_t) i * 18 + 1] = (uint8_t) (d >> 8);
                for (int j = 0; j < 16; ++j) blk[(size_t) i * 18 + 2 + j] = (uint8_t) (rng() & 0xff);
            }
        auto weight = [&](int64_t n, int64_t k) {
            const int src = c.interleave ? (int) (n & 1) : 0;
            const int64_t r = c.interleave ? n / 2 : n;
            const uint8_t* b = &B[(size_t) src][(size_t) (r * nb + k / 32) * 18];
            const int e = (int) (k % 32);
            const int q = e < 16 ? (b[2 + e] & 0xF) : (b[2 + e - 16] >> 4);
            return (double) (q - 8) * (double) f16_to_f32((uint16_t) (b[0] | (b[1] << 8)));
        };
        std::vector<uint16_t> X((size_t) c.M * c.K);
        for (auto& x : X) x = f32_to_f16(uni(rng));
        auto* dX = (uint16_t*) strata::ze::malloc_device(X.size() * 2);
        auto* dQ = (uint8_t*) strata::ze::malloc_device((size_t) c.N * c.K / 2);
        auto* dS = (uint16_t*) strata::ze::malloc_device((size_t) c.N * nb * 2);
        auto* dY = (float*) strata::ze::malloc_device((size_t) c.M * c.N * 4);
        std::vector<uint8_t*> dB(nsrc);
        for (int64_t i = 0; i < nsrc; ++i) {
            dB[(size_t) i] = (uint8_t*) strata::ze::malloc_device(B[(size_t) i].size());
            s.memcpy(dB[(size_t) i], B[(size_t) i].data(), B[(size_t) i].size());
        }
        s.memcpy(dX, X.data(), X.size() * 2);
        for (int64_t i = 0; i < nsrc; ++i)
            strata::prefill::q4_repack(dB[(size_t) i], rows_src, c.K, dQ, dS, c.N, nsrc, i, &s);
        auto run = [&] { return strata::gs::gemm(strata::gs::Weights::Q4, dX, c.K, dQ, c.K, dS, dY, c.N, c.M, c.N, c.K, false, &s); };
        s.memset(dY, 0xff, (size_t) c.M * c.N * 4);
        const bool ran = run();
        std::vector<float> Y((size_t) c.M * c.N);
        s.memcpy(Y.data(), dY, Y.size() * 4);
        s.sync();
        Check ck;
        if (!ran) ck.bad = 1;
        for (int64_t m = 0; m < c.M && ran; ++m)
            for (int64_t n = 0; n < c.N; ++n) {
                double ref = 0.0, mag = 0.0;
                for (int64_t k = 0; k < c.K; ++k) {
                    const double p = (double) f16_to_f32(X[(size_t) m * c.K + k]) * weight(n, k);
                    ref += p;
                    mag += std::fabs(p);
                }
                const float got = Y[(size_t) m * c.N + n];
                if (!std::isfinite(got)) { ++ck.nonfinite; continue; }
                const double tol = (4.0 * (double) (c.K + 1) * std::ldexp(1.0, -24) + std::ldexp(1.0, -10)) * mag + 1e-30;
                const double e = std::fabs((double) got - ref);
                if (e > tol) ++ck.bad;
                if (mag > 0) ck.worst = std::max(ck.worst, e / mag);
            }
        bad_cases += !ck.ok();
        ++n_checks;
        report(c.what, "gs", c.M, c.N, c.K, ck);
        if (!ran) std::printf("  no gemmstone kernel");
        if (bench && ran) {
            const int reps = c.M * c.N * c.K < 100000000ll ? 200 : 20;
            const double us = time_us(s, reps, run);
            std::printf("  %9.1f us  %7.2f TFLOP/s  %7.1f GB/s of W", us, 2.0 * c.M * c.N * c.K / us / 1e6,
                        (double) c.N * c.K * 0.5625 / us / 1e3);
        }
        std::printf("\n");
        for (uint8_t* p : dB) strata::ze::free(p);
        strata::ze::free(dX);
        strata::ze::free(dQ);
        strata::ze::free(dS);
        strata::ze::free(dY);
    }
    std::printf("xgemm_parity: %d of %d checks failed\n", bad_cases, n_checks);
    return bad_cases == 0 ? 0 : 1;
}
