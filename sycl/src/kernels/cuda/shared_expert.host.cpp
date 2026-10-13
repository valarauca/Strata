// src/kernels/cuda/shared_expert.host.cpp - the host side of shared_expert.dp.cpp: the shared expert's launchers.
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cmath>
#include <climits>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {

#include "shared_expert.kern.inc"

namespace {

constexpr int THREADS = 128;
bool native_bf16 = false;

}  // namespace

void shared_expert_set_native_bf16(bool enabled) { native_bf16 = enabled; }
bool shared_expert_native_bf16_enabled() { return native_bf16; }

namespace {
void launch_sigmoid_scale_rows(float *out, const float *g, int n_embd,
                               int n_tok, ze::Stream* cs) {
    if ((n_embd & 3) == 0 && ((uintptr_t) out & 15u) == 0) {
        const int n4 = n_embd >> 2;
        k_shared_expert_sigmoid_scale_rows_vec4(cs, {(unsigned) ((n4 + THREADS - 1) / THREADS), (unsigned) n_tok, 1},
                                                {THREADS, 1, 1}, 0, out, g, n4);
    } else {
        k_shared_expert_sigmoid_scale_rows(cs, {(unsigned) ((n_embd + THREADS - 1) / THREADS), (unsigned) n_tok, 1},
                                           {THREADS, 1, 1}, 0, out, g, n_embd);
    }
}

bool fused_swiglu_q81_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_FUSED_SWIGLU_Q81");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}
}  // namespace

void shared_expert_multi(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd,
                         int64_t n_ff, void* stream, const void* x_q8_1_ready, int lfuse) {
    const bool gate_deferred = (lfuse & 1) != 0, pair = (lfuse & 2) != 0;
    if (n_tok < 1 || n_tok > 8 || !nw.q8_1 || !nw.gate_data || !nw.up_data || !nw.down_data || !stream)
        throw std::invalid_argument("shared_expert_multi: needs 1..8 tokens, native weights, scratch and a stream");
    ze::Stream* cs = ze::stream_of(stream);
    // the activation's q8_1: the caller's (STRATA_VERIFY_QDEDUP, x_q8_1_ready) or the descriptor's, else quantized here
    const void* x_q8_1 = x_q8_1_ready ? x_q8_1_ready : nw.x_q8_1;
    if (!x_q8_1) {
        native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
        x_q8_1 = nw.q8_1;
    }
    // S26 STRATA_LFUSE (gate_deferred): gate and up in one launch where the pair applies (bitwise the two calls)
    if (!(pair && nw.gate_type == nw.up_type &&
          native_mmvq_pair(nw.gate_type, nw.gate_data, nw.up_data, x_q8_1, gate, up, (int) n_embd, (int) n_ff, n_tok, stream))) {
        native_mmvq(nw.gate_type, nw.gate_data, x_q8_1, gate, (int) n_embd, (int) n_ff, n_tok, stream);
        native_mmvq(nw.up_type, nw.up_data, x_q8_1, up, (int) n_embd, (int) n_ff, n_tok, stream);
    }
    const int n = (int) (n_ff * n_tok);
    if (fused_swiglu_q81_enabled()) {
        native_swiglu_quantize_q8_1(gate, up, nw.q8_1, (int) n_ff, n_tok, stream);
    } else {
        k_shared_expert_native_swiglu(cs, {(unsigned) ((n + THREADS - 1) / THREADS), 1, 1}, {THREADS, 1, 1}, 0, gate, up,
                                      gate, n);
        native_quantize_q8_1(gate, nw.q8_1, (int) n_ff, n_tok, stream);
    }
    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
    if (gate_deferred) {   // the caller computed g (raw) with the router and scales `out` in the combine
        return;
    }
    static const bool batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    if (native_bf16 && batch && n_tok > 1) {   // one gemv for all rows (outputs identical), fused sigmoid+scale
        bf16_gemv_fp32_mmvf_multi(x, n_embd, gate_inp_bf16, g, 1, n_embd, 1, n_tok, stream);
        launch_sigmoid_scale_rows(out, g, (int) n_embd, n_tok, cs);
    } else if (native_bf16 && n_tok == 1) {
        bf16_gemv_fp32_mmvf(x, gate_inp_bf16, g, n_embd, 1, stream);
        launch_sigmoid_scale_rows(out, g, (int) n_embd, 1, cs);
    } else {
        for (int t = 0; t < n_tok; ++t) {
            if (native_bf16) {
                bf16_gemv_fp32_mmvf(x + (size_t) t * n_embd, gate_inp_bf16, g + t, n_embd, 1, stream);
                k_shared_expert_native_scalar_sigmoid(cs, {1, 1, 1}, {1, 1, 1}, 0, g + t);
            } else {
                k_shared_expert_scalar_gate(cs, {1, 1, 1}, {256, 1, 1}, 0, x_bf16 + (size_t) t * n_embd, gate_inp_bf16,
                                            g + t, (int) n_embd);
            }
        }
        k_shared_expert_scale_rows(cs, {(unsigned) ((n_embd + THREADS - 1) / THREADS), (unsigned) n_tok, 1},
                                   {THREADS, 1, 1}, 0, out, g, (int) n_embd);
    }
}

uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    // gate (n_ff f32) | up (n_ff f32) | q8_0 (n_ff/32*34) | q8k (n_ff/256*292) | g (1 f32), 16-byte aligned
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}

void shared_expert(const uint8_t *x_q8_0, const uint8_t *x_q8k,
                   const uint16_t *x_bf16, const SForm &gate_form,
                   const uint8_t *gate_codes, const float *gate_scales,
                   const float *gate_off, const SForm &up_form,
                   const uint8_t *up_codes, const float *up_scales,
                   const float *up_off, const SForm &down_form,
                   const uint8_t *down_codes, const float *down_scales,
                   const float *down_off, const uint16_t *gate_inp_bf16,
                   float *scratch, float *out, int64_t n_embd, int64_t n_ff,
                   int tpr, void *stream, const float *x_f32,
                   const NativeSharedWeights *native) {
    if (n_embd <= 0 || n_ff <= 0) return;
    const bool use_native = native_bf16;
    const bool native_gate = native && native->gate_data && native_mmvq_supported(native->gate_type);
    const bool native_up = native && native->up_data && native_mmvq_supported(native->up_type);
    const bool native_down = native && native->down_data && native_mmvq_supported(native->down_type);
    const bool native_projection = native_gate || native_up || native_down;
    if ((use_native || native_gate || native_up) && !x_f32)
        throw std::invalid_argument("shared_expert native input projection requires unrounded x_f32");
    if (native_projection) {
        if (!native->q8_1 || !stream || n_embd > INT_MAX || n_ff > INT_MAX)
            throw std::invalid_argument("shared_expert native projections require scratch, stream and int32 dimensions");
        // Validate every active shape before any kernel is enqueued.
        if (native_gate) native_mmvq_weight_bytes(native->gate_type, (int) n_embd, (int) n_ff);
        if (native_up) native_mmvq_weight_bytes(native->up_type, (int) n_embd, (int) n_ff);
        if (native_down) native_mmvq_weight_bytes(native->down_type, (int) n_ff, (int) n_embd);
    }
    if (scratch == nullptr) {
        std::fprintf(stderr, "shared_expert: scratch is null; the caller owns it "
                             "(see shared_expert_scratch_bytes)\n");
        std::exit(1);
    }
    // CARVED FROM THE CALLER'S SCRATCH.  This used to be four `cudaMalloc`s and four `cudaFree`s per call, which
    // is illegal during stream capture AND a token-path allocation - P2.T10 forbids both.
    uint8_t* p = (uint8_t*) scratch;
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    float* gate = (float*) p;
    float* up = (float*) (p + a);
    uint8_t* h_q8_0 = (uint8_t*) (p + a * 2);
    uint8_t* h_q8k = (uint8_t*) (p + a * 2 + q0);
    float* g = (float*) (p + a * 2 + q0 + qk);

    // WHICH ACTIVATION THIS PROJECTION WANTS, READ FROM ITS OWN FORM.  See `SForm::act_kind`: the three
    // families cannot be told apart by the other fields, so the kind is carried rather than derived.
    auto gemv = [&](const SForm& f, const uint8_t* codes, const float* scales, const float* off,
                    const uint8_t* act80, const uint8_t* actq8k, float* y, int64_t nin, int64_t nout) {
        if (f.code_bits == 2) {
            s2_gemv_q8(act80, codes, scales, y, nin, nout, tpr, stream);
        } else if (f.act_kind == 1) {
            s_gemv_q8k_split(actq8k, codes, scales, off, y, nin, nout, f, stream);
        } else {
            s_gemv_q8_0_split(act80, codes, scales, off, y, nin, nout, f, stream);
        }
    };

    const unsigned g_ff = (unsigned) ((n_ff + THREADS - 1) / THREADS);
    const unsigned g_embd = (unsigned) ((n_embd + THREADS - 1) / THREADS);

    // gate and up projections, then silu(gate) * up in place in `gate`
    if (native_gate || native_up)
        native_quantize_q8_1(x_f32, native->q8_1, (int) n_embd, 1, stream);
    if (native_gate)
        native_mmvq(native->gate_type, native->gate_data, native->q8_1, gate, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(gate_form, gate_codes, gate_scales, gate_off, x_q8_0, x_q8k, gate, n_embd, n_ff);
    if (native_up)
        native_mmvq(native->up_type, native->up_data, native->q8_1, up, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(up_form, up_codes, up_scales, up_off, x_q8_0, x_q8k, up, n_embd, n_ff);
    if (native_projection && native_down && fused_swiglu_q81_enabled()) {
        native_swiglu_quantize_q8_1(gate, up, native->q8_1, (int) n_ff, 1, stream);
        native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, stream);
    } else {
        if (native_projection)
        {
            k_shared_expert_native_swiglu(ze::stream_of(stream), {g_ff, 1, 1}, {THREADS, 1, 1}, 0, gate, up, gate,
                                          (int) n_ff);
        } else {
            k_shared_expert_swiglu(ze::stream_of(stream), {g_ff, 1, 1}, {THREADS, 1, 1}, 0, gate, up, gate, (int) n_ff);
        }

        // down: (n_ff) -> (n_embd), and THE INTERMEDIATE IS QUANTIZED TO THE DOWN WEIGHT'S OWN CONTRACT - which is
        // what `ggml_mul_mat` does for every matmul in the model.  It used to be rounded to fp16 with no
        // justification beyond "the kernel takes fp16".
        if (native_down) {
            native_quantize_q8_1(gate, native->q8_1, (int) n_ff, 1, stream);
            native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, stream);
        } else if (down_form.act_kind == 1) {
            if (n_ff % 256 != 0) {
                std::fprintf(stderr, "shared_expert: the down weight wants Q8_K but n_ff %lld is not a multiple "
                                     "of 256; Q8_K is structurally impossible here\n", (long long) n_ff);
                std::exit(1);
            }
            quantize_q8_K(gate, h_q8k, n_ff, stream);
            gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
        } else if (down_form.code_bits == 2) {
            quantize_q8_0(gate, h_q8_0, n_ff, stream);
            gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
        } else {
            quantize_q8_0(gate, h_q8_0, n_ff, stream);
            gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
        }
    }

    // the per-token scalar gate, then the multiply.  Note the gate is computed from `x`, the ORIGINAL hidden
    // state, not from anything the expert produced. The historical branch uses BF16-rounded inputs; the
    // native branch uses the pinned CUDA FP32 activation contract.
    // `<<<1, 256>>>`: one block, because the output is ONE scalar and a second block would only add a global
    // round trip.  256 threads is the reduction's width, not the problem's size.
    if (use_native) {
        bf16_gemv_fp32_mmvf(x_f32, gate_inp_bf16, g, n_embd, 1, stream);
        launch_sigmoid_scale_rows(out, g, (int)n_embd, 1,
                                  ze::stream_of(stream));
    } else {
        k_shared_expert_scalar_gate(ze::stream_of(stream), {1, 1, 1}, {256, 1, 1}, 0, x_bf16, gate_inp_bf16, g,
                                    (int) n_embd);
        k_shared_expert_scale(ze::stream_of(stream), {g_embd, 1, 1}, {THREADS, 1, 1}, 0, out, g, (int) n_embd);
    }

    if (stream == nullptr) ze::default_stream().sync();
}

void moe_combine(const float *parts, const float *weights, const float *shared,
                 float *y, int64_t n_embd, int64_t k, void *stream) {
    if (n_embd <= 0 || k <= 0) return;
    // k > 64 is refused rather than truncated: silently summing the first 64 of a longer list would be a
    // wrong answer that looks like a right one, and no geometry in this artifact comes close to it.
    if (k > 64) {
        std::fprintf(stderr, "moe_combine: k = %lld exceeds 64\n", (long long) k);
        std::exit(1);
    }
    const unsigned grid = (unsigned) ((n_embd + THREADS - 1) / THREADS);
    k_shared_expert_moe_combine(ze::stream_of(stream), {grid, 1, 1}, {THREADS, 1, 1}, 0, parts, weights, shared, y,
                                (int) n_embd, (int) k, shared != nullptr);
    if (stream == nullptr) ze::default_stream().sync();
}

}  // namespace strata::kernels
