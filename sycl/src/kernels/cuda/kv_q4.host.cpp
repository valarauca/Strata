// src/kernels/cuda/kv_q4.host.cpp - the host side of kv_q4.dp.cpp: shape checks and the launchers.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

#include "kv_q4.kern.inc"

namespace {

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    const int rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    k_kv_q4_fwht256(ze::stream_of(stream), {(unsigned)num_blocks, 1, 1}, {32, rows_per_block, 1}, 0, src, dst, n_rows,
                    1.0f / 16.0f);
}

void kv_append_q4_steps(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_q4");
    const int planes = (v_q4 == nullptr || (v_q4 == k_q4 && vcur == kcur)) ? 1 : 2;
    const ze::Dim3 grid{(unsigned)s.n_head_kv, (unsigned)(s.head_dim / QK4_0), (unsigned)(n_steps * planes)};
    k_kv_q4_append(ze::stream_of(stream), grid, {32, 1, 1}, 0, k_q4, v_q4, page_table, step, step_stride, planes, kcur,
                   vcur, (int)s.n_head_kv, (int)s.head_dim, (int)s.page_size, host ? *host : KvHostPools{});
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    kv_append_q4_steps(k_q4, v_q4, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const ze::Dim3 grid{(unsigned)T, (unsigned)s.n_head_kv, (unsigned)(s.head_dim / QK4_0)};
    ze::Stream* cs = ze::stream_of(stream);
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
        k_kv_q4_append_batch(cs, grid, {32, 1, 1}, 0, k_q4, v_q4, page_table, pos0, K, V, (int)s.n_head_kv,
                             (int)s.head_dim, (int)s.page_size, is_v, h, st);
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / QK4_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    k_kv_q4_gather(ze::stream_of(stream), {num_blocks, 1, 1}, {32, rows_per_block, 1}, 0, k_q4, v_q4, page_table, ids,
                   step, (int)s.n_head_kv, (int)s.head_dim, (int)s.page_size, k_scratch, v_scratch);
}

}  // namespace strata::kernels
