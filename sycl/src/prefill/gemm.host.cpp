// src/prefill/gemm.host.cpp - see include/strata/prefill/gemm.hpp: the Intel engine's dense products, on gemmstone's
// generated kernels (strata/gemmstone.hpp), else the XMX GEMM of gemm.dp.cpp (no BLAS library). STRATA_XGEMM=xmx keeps
// every product on the XMX GEMM.
#include "strata/gemmstone.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::prefill {

namespace xgemm {
#include "gemm.kern.inc"
}  // namespace xgemm

namespace {

using GemmKernel = ze::Kernel<void(const uint16_t*, long long, const uint16_t*, long long, float*, long long, int, int, int, int)>;

struct Shape {
    const GemmKernel* f16;
    const GemmKernel* bf16;
    int bm, bn, threads;
};

const Shape kLarge{&xgemm::k_xgemm_f16_4x4_4x4, &xgemm::k_xgemm_bf16_4x4_4x4, 128, 256, 256};
const Shape kMedium{&xgemm::k_xgemm_f16_2x4_4x2, &xgemm::k_xgemm_bf16_2x4_4x2, 64, 128, 128};
const Shape kSmall{&xgemm::k_xgemm_f16_1x8_4x1, &xgemm::k_xgemm_bf16_1x8_4x1, 32, 128, 128};

const Shape& pick(int64_t M) {
    static const char* forced = std::getenv("STRATA_XGEMM_SHAPE");
    if (forced != nullptr && forced[0] == 'L') return kLarge;
    if (forced != nullptr && forced[0] == 'M') return kMedium;
    if (forced != nullptr && forced[0] == 'S') return kSmall;
    if (M >= 256) return kLarge;
    if (M >= 48) return kMedium;
    return kSmall;
}

void run(bool bf, const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw, float* Y, int64_t ldy, int64_t M,
         int64_t N, int64_t K, float beta, void* stream) {
    if (beta != 0.0f && beta != 1.0f) {
        std::fprintf(stderr, "prefill gemm: beta %g (only 0 and 1 are used)\n", (double) beta);
        std::exit(1);
    }
    const int acc = beta != 0.0f ? 1 : 0;
    static const bool xmx_only = [] { const char* v = std::getenv("STRATA_XGEMM"); return v && std::strcmp(v, "xmx") == 0; }();
    if (!xmx_only && gs::gemm(bf ? gs::Weights::BF16 : gs::Weights::F16, X, ldx, W, ldw, nullptr, Y, ldy, M, N, K, acc != 0, stream))
        return;
    ze::Stream* s = ze::stream_of(stream);
    const bool blockable = ldx % 4 == 0 && ldw % 4 == 0 && K % 2 == 0 && ldy % 2 == 0 && ((uintptr_t) X & 3) == 0 &&
                           ((uintptr_t) W & 3) == 0 && ((uintptr_t) Y & 3) == 0;
    if (N < 16 || !blockable) {
        const GemmKernel& k = bf ? xgemm::k_xgemm_narrow_bf16 : xgemm::k_xgemm_narrow_f16;
        k(s, {(uint32_t) N, (uint32_t) M, 1}, {16, 1, 1}, 0, X, ldx, W, ldw, Y, ldy, (int) M, (int) N, (int) K, acc);
        return;
    }
    const Shape& sh = pick(M);
    const GemmKernel& k = bf ? *sh.bf16 : *sh.f16;
    k(s, {(uint32_t) ((N + sh.bn - 1) / sh.bn), (uint32_t) ((M + sh.bm - 1) / sh.bm), 1}, {(uint32_t) sh.threads, 1, 1}, 0, X,
      ldx, W, ldw, Y, ldy, (int) M, (int) N, (int) K, acc);
}

}  // namespace

bool prompt_f16() { return false; }

Gemm::~Gemm() {
    if (!external_) {
        if (scratch_) ze::free(scratch_);
        if (workspace_) ze::free(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    (void) ws_bytes;
    (void) err;
    stream_ = stream;
    external_ = true;
    workspace_ = workspace;
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    (void) ws_bytes;
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    stream_ = stream;
    if (scratch_elems > 0) {
        scratch_ = (uint16_t*) ze::malloc_device((size_t) scratch_elems * 2, ze::stream_of(stream)->device());
        if (scratch_ == nullptr) {
            err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB: out of device memory";
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}

bool Gemm::bf16_hcd_exact(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K) {
    (void) X; (void) ldx; (void) W; (void) Y; (void) T; (void) N; (void) K;
    return false;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
                int64_t ldx) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (ldx <= K) ldx = K;
    run(true, X, ldx, W, K, Y, ldy, T, N, K, beta, stream_);
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    run(false, X, K, W, K, Y, ldy, T, N, K, beta, stream_);
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta, int64_t ldx) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (ldx <= K) ldx = K;
    const int64_t rows = scratch_elems_ / K;
    if (rows <= 0) {
        std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K);
        std::exit(1);
    }
    for (int64_t r0 = 0; r0 < N; r0 += rows) {
        const int64_t n = (N - r0 < rows) ? N - r0 : rows;
        strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
        run(false, X, ldx, scratch_, K, Y + r0, ldy, T, n, K, beta, stream_);
    }
}

}  // namespace strata::prefill
