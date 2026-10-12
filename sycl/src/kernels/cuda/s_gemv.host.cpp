// src/kernels/cuda/s_gemv.host.cpp - the host side of s_gemv.dp.cpp: the launchers.
#include "strata/kernels/s_gemv.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "s_gemv.kern.inc"

namespace {

constexpr int Q8K_BLOCK_ELEMS = 256;
constexpr int Q8_0_BLOCK_ELEMS = 32;

}  // namespace

void s_gemv(const uint16_t *x, const uint8_t *codes, const float *scales,
            const float *offset, float *y, int64_t n_in, int64_t n_out,
            const SForm &form) {
    if (n_in <= 0 || n_out <= 0) return;
    if (form.group_elems <= 0 || n_in % form.group_elems != 0) {
        std::fprintf(stderr, "s_gemv: n_in %lld is not a multiple of group_elems %d\n", (long long) n_in,
                     form.group_elems);
        std::exit(1);
    }
    if (form.has_offset && offset == nullptr) {
        std::fprintf(stderr, "s_gemv: form says has_offset but offset is null\n");
        std::exit(1);
    }
    const int threads = 128;
    const long long blocks = (n_out + threads - 1) / threads;
    const int cb = (int) form.codebook;
    ze::Stream* q = &ze::default_stream();
    const ze::Dim3 grid{(unsigned) blocks, 1, 1}, block{threads, 1, 1};
    switch (form.code_bits) {
    case 2:
        k_s_gemv_2(q, grid, block, 0, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems,
                   form.has_offset ? 1 : 0);
        break;
    case 4:
        k_s_gemv_4(q, grid, block, 0, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems,
                   form.has_offset ? 1 : 0);
        break;
    case 8:
        k_s_gemv_8(q, grid, block, 0, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb, form.group_elems,
                   form.has_offset ? 1 : 0);
        break;
        default:
            std::fprintf(stderr, "s_gemv: unsupported code_bits %d\n", form.code_bits);
            std::exit(1);
    }
    ze::device_sync();
}

static void s_gemv_split_impl(const uint16_t *x, const uint8_t *codes,
                              const float *scales, const float *offset,
                              float *y, int64_t n_in, int64_t n_out,
                              const SForm &form, int threads_per_row,
                              ze::Stream* stream, bool sync) {
    if (n_in <= 0 || n_out <= 0) return;
    if (threads_per_row < 1 || (threads_per_row & (threads_per_row - 1)) != 0 || threads_per_row > 1024) {
        std::fprintf(stderr, "s_gemv_split: threads_per_row must be a power of two in 1..1024, got %d\n",
                     threads_per_row);
        std::exit(1);
    }
    const int cb = (int) form.codebook;
    // Every group size this format defines is a power of two, which is what lets the per-element group index
    // be a shift.  Refuse anything else rather than compute a wrong index quietly.
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0) {
        std::fprintf(stderr, "s_gemv_split: group_elems must be a power of two, got %d\n", form.group_elems);
        std::exit(1);
    }
    // AND THE QUAD MUST FIT INSIDE ONE GROUP: four consecutive elements share a scale and the kernel reads
    // exactly one.  Every group size this format defines is 16, 32 or 64, so this cannot fire on a real pack.
    if (form.group_elems % 4 != 0) {
        std::fprintf(stderr, "s_gemv_split: group_elems %d is not a multiple of 4\n", form.group_elems);
        std::exit(1);
    }
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    const size_t smem = (size_t)threads_per_row * sizeof(float);
    const unsigned grid = (unsigned) n_out;
    const ze::Dim3 g{grid, 1, 1}, b{(unsigned) threads_per_row, 1, 1};
    switch (form.code_bits) {
        case 2:
            k_s_gemv_split_2(stream, g, b, smem, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                             form.group_elems, group_shift, form.has_offset ? 1 : 0, threads_per_row);
            break;
        case 4:
            k_s_gemv_split_4(stream, g, b, smem, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                             form.group_elems, group_shift, form.has_offset ? 1 : 0, threads_per_row);
            break;
        case 8:
            k_s_gemv_split_8(stream, g, b, smem, x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                             form.group_elems, group_shift, form.has_offset ? 1 : 0, threads_per_row);
            break;
        default:
            std::fprintf(stderr, "s_gemv_split: unsupported code_bits %d\n", form.code_bits);
            std::exit(1);
    }
    if (sync) {
        ze::device_sync();
    }
}

void s_gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                  float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row) {
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form,
                      threads_per_row,
                      &ze::default_stream(), true);
}

void s_gemv_split_async(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                        float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row,
                        void* stream) {
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form,
                      threads_per_row, ze::stream_of(stream), false);
}

// ---- the Q8_K entry points -----------------------------------------------------------------------------

namespace {

bool q8k_form_ok(const SForm& form, int64_t n_in, const char* who) {
    if (form.group_elems <= 0 || n_in % form.group_elems != 0) {
        std::fprintf(stderr, "%s: n_in %lld is not a multiple of group_elems %d\n", who, (long long) n_in,
                     form.group_elems);
        return false;
    }
    if (n_in % Q8K_BLOCK_ELEMS != 0) {
        // `quantize_q8_K` requires this too, and a partial block would read past the end of the activation.
        std::fprintf(stderr, "%s: n_in %lld is not a multiple of the Q8_K block %d\n", who, (long long) n_in,
                     Q8K_BLOCK_ELEMS);
        return false;
    }
    return true;
}

void finish(const char *who, void *stream) {
    if (stream != nullptr) return;
    ze::device_sync();
}

}  // namespace

void s_gemv_q8k(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset, float* y,
                int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (!q8k_form_ok(form, n_in, "s_gemv_q8k")) std::exit(1);
    const int threads = 128;
    const unsigned blocks = (unsigned) ((n_out + threads - 1) / threads);
    const int cb = (int) form.codebook;
    const int ho = form.has_offset ? 1 : 0;
    switch (form.code_bits) {
    case 4:
        k_s_gemv_q8k_4(ze::stream_of(stream), {blocks, 1, 1}, {threads, 1, 1}, 0, x_q8k, codes, scales, offset, y, n_in,
                       n_out, form.code_bias, cb, form.group_elems, ho);
        break;
    case 8:
        k_s_gemv_q8k_8(ze::stream_of(stream), {blocks, 1, 1}, {threads, 1, 1}, 0, x_q8k, codes, scales, offset, y, n_in,
                       n_out, form.code_bias, cb, form.group_elems, ho);
        break;
        default:
            // 2-bit S2 never has a Q8_K activation: its `vec_dot_type` is Q8_0.  Refusing is better than
            // running a kernel that would be numerically wrong in a way the caller cannot see.
            std::fprintf(stderr, "s_gemv_q8k: code_bits %d has no Q8_K contract "
                                 "(S2 uses Q8_0; see docs/activation-contract.md)\n", form.code_bits);
            std::exit(1);
    }
    finish("s_gemv_q8k", stream);
}

void s_gemv_q8k_split(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset,
                      float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (!q8k_form_ok(form, n_in, "s_gemv_q8k_split")) std::exit(1);
    // THE GROUP SIZE IS PASSED AS ITS LOGARITHM: see the note on the kernel.  A non-power-of-two group would
    // make the shift a WRONG INDEX rather than a slow one, so it is refused here.
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    if ((1 << group_shift) != form.group_elems) {
        std::fprintf(stderr, "s_gemv_q8k_split: group_elems %d is not a power of two\n", form.group_elems);
        std::exit(1);
    }
    // AND THE QUAD MUST FIT INSIDE ONE GROUP.  Four consecutive elements share a scale, so a group smaller
    // than four would need two of them and the kernel reads exactly one.  Every group size this format
    // defines is 16, 32 or 64, so this cannot fire on a real pack - it is here because a future one could.
    if (form.group_elems % 16 != 0) {
        std::fprintf(stderr, "s_gemv_q8k_split: group_elems %d is not a multiple of 16\n", form.group_elems);
        std::exit(1);
    }
    const int threads = 256;
    const int warps = threads / 32;
    const unsigned blocks = (unsigned) ((n_out + warps - 1) / warps);
    const int cb = (int) form.codebook;
    const int ho = form.has_offset ? 1 : 0;
    switch (form.code_bits) {
    case 4:
        k_s_gemv_q8k_split_4(ze::stream_of(stream), {blocks, 1, 1}, {threads, 1, 1}, 0, x_q8k, codes, scales, offset, y,
                             n_in, n_out, form.code_bias, cb, group_shift, ho);
        break;
    case 8:
        k_s_gemv_q8k_split_8(ze::stream_of(stream), {blocks, 1, 1}, {threads, 1, 1}, 0, x_q8k, codes, scales, offset, y,
                             n_in, n_out, form.code_bias, cb, group_shift, ho);
        break;
        default:
            std::fprintf(stderr, "s_gemv_q8k_split: code_bits %d has no Q8_K contract\n", form.code_bits);
            std::exit(1);
    }
    finish("s_gemv_q8k_split", stream);
}

void s_gemv_q8_0_split(const uint8_t* x_q8_0, const uint8_t* codes, const float* scales, const float* offset,
                       float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // THE ACTIVATION IS `block_q8_0`, 32 elements per block, so `n_in` must be a multiple of 32 - which is a
    // WEAKER requirement than Q8_K's 256 and is exactly why this kernel has to exist: `ffn_down_shexp` has
    // n_in = 640, a multiple of 32 that is NOT a multiple of 256.
    if (n_in % Q8_0_BLOCK_ELEMS != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: n_in %lld is not a multiple of %d\n", (long long) n_in,
                     Q8_0_BLOCK_ELEMS);
        std::exit(1);
    }
    // the same two checks `s_gemv_q8k_split` makes, for the same reasons - see the notes there
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: group_elems must be a power of two, got %d\n",
                     form.group_elems);
        std::exit(1);
    }
    // SIXTEEN, NOT FOUR: a lane-iteration takes QE = 16 consecutive elements under ONE scale (see the kernel),
    // so a group of 4 or 8 would read the wrong scale for most of them.  The Q8_K launcher already said 16.
    if (form.group_elems % 16 != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: group_elems %d is not a multiple of 16\n", form.group_elems);
        std::exit(1);
    }
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;

    const int threads = 256;
    const int warps = threads / 32;
    const unsigned blocks = (unsigned) ((n_out + warps - 1) / warps);
    const int cb = (int) form.codebook;
    const int ho = form.has_offset ? 1 : 0;
    switch (form.code_bits) {
    case 4:
        k_s_gemv_q8_0_split_4(ze::stream_of(stream), {blocks, 1, 1}, {threads, 1, 1}, 0, x_q8_0, codes, scales, offset,
                              y, n_in, n_out, form.code_bias, cb, group_shift, ho);
        break;
    case 8:
        k_s_gemv_q8_0_split_8(ze::stream_of(stream), {blocks, 1, 1}, {threads, 1, 1}, 0, x_q8_0, codes, scales, offset,
                              y, n_in, n_out, form.code_bias, cb, group_shift, ho);
        break;
        default:
            std::fprintf(stderr, "s_gemv_q8_0_split: code_bits %d has no Q8_0 contract\n", form.code_bits);
            std::exit(1);
    }
    finish("s_gemv_q8_0_split", stream);
}

}  // namespace strata::kernels
