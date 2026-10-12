// src/kernels/cuda/kv_q8.host.cpp - the host side of kv_q8.dp.cpp: shape checks and the launchers.
#include "strata/kernels/kv_q8.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "kv_q8.kern.inc"

namespace {

void validate(const QsaShapes& s, const char* what) {
    if (s.head_dim % KV_Q8_GROUP != 0 || s.n_head_kv <= 0 || s.page_size <= 0) {
        std::fprintf(stderr, "kv_q8: %s: head_dim %lld must be a multiple of %d\n", what, (long long) s.head_dim,
                     KV_Q8_GROUP);
        std::exit(1);
    }
}

}  // namespace

void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    validate(s, "kv_append_q8");
    const ze::Dim3 grid{(unsigned)s.n_head_kv, (unsigned)(s.head_dim / KV_Q8_GROUP), 2};
    k_kv_q8_append(ze::stream_of(stream), grid, {KV_Q8_GROUP, 1, 1}, 0, k_q, v_q, k_scale, v_scale, page_table, step,
                   kcur, vcur, (int)s.n_head_kv, (int)s.head_dim, (int)s.page_size, host ? *host : KvHostPools{});
}

void kv_append_q8_steps(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                        const int32_t* step, int step_stride, const float* kcur, const float* vcur, int cur_stride,
                        int n_tok, const QsaShapes& s, void* stream, const KvHostPools* host) {
    validate(s, "kv_append_q8 (steps)");
    if (n_tok < 1) return;
    const ze::Dim3 grid{(unsigned)s.n_head_kv, (unsigned)(s.head_dim / KV_Q8_GROUP), (unsigned)(2 * n_tok)};
    k_kv_q8_append_multi(ze::stream_of(stream), grid, {KV_Q8_GROUP, 1, 1}, 0, k_q, v_q, k_scale, v_scale, page_table,
                         step, kcur, vcur, (int)s.n_head_kv, (int)s.head_dim, (int)s.page_size,
                         host ? *host : KvHostPools{}, step_stride, cur_stride);
}

void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather_q8");
    if (max_ids <= 0) return;
    const long long total = max_ids * s.n_head_kv * (s.head_dim / 4);
    const unsigned blocks = (unsigned) ((total + 255) / 256);
    k_kv_q8_gather(ze::stream_of(stream), {blocks, 1, 1}, {256, 1, 1}, 0, k_q, v_q, k_scale, v_scale, page_table, ids,
                   step, (int)s.n_head_kv, (int)s.head_dim, (int)s.page_size, k_scratch, v_scratch);
}

}  // namespace strata::kernels
