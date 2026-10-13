// verify_kernels.kern.inc - entry points of verify_kernels.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(verify_kernels_gdn_conv_l2_multi,
              (float* hist, const float* qkv, const float* w, float* h, int C, int qk_heads, float eps, int t_begin,
               bool commit),
              gdn_conv_l2_multi_kernel(hist, qkv, w, h, C, qk_heads, eps, t_begin, commit))
STRATA_KERNEL(verify_kernels_gdn_conv_commit, (float* hist, const float* qkv, int C, const int32_t* n_keep),
              gdn_conv_commit_kernel(hist, qkv, C, n_keep))

#define VK_AB(NAME, MAX_T, EXACT_T)                                                                                   \
    STRATA_KERNEL(verify_kernels_gdn_ab_multi_##NAME,                                                                 \
                  (const float* x, const uint16_t* wa, const uint16_t* wb, const float* dt, const float* ssm_a,         \
                   float* gate, float* beta, int n, int h_v, int T),                                                  \
                  gdn_ab_multi_kernel<MAX_T, EXACT_T>(x, wa, wb, dt, ssm_a, gate, beta, n, h_v, T))
VK_AB(t1, 1, true)
VK_AB(t2, 2, true)
VK_AB(t3, 3, true)
VK_AB(t4, 4, true)
VK_AB(t5, 5, true)
VK_AB(t6, 6, true)
VK_AB(tmax, kVerifyMaxT, false)
#undef VK_AB

STRATA_KERNEL(verify_kernels_gdn_step_commit,
              (float* state, const float* hbuf, int C, const float* gate, const float* beta, int h_k, int h_v,
               const int32_t* n_keep),
              gdn_step_commit_kernel(state, hbuf, C, gate, beta, h_k, h_v, n_keep))

#define VK_NORM(NAME, ALL_OUT, Q)                                                                                     \
    STRATA_KERNEL(verify_kernels_gdn_step_norm_multi_##NAME,                                                          \
                  (float* state, const float* hbuf, int C, const float* gate, const float* beta, const float* z,       \
                   const float* gamma, float eps, float* y, int h_k, int h_v, int T, const int32_t* n_keep,            \
                   int t_out_begin, void* xq),                                                                        \
                  gdn_step_norm_multi_kernel<ALL_OUT, Q>(state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, T,   \
                                                         n_keep, t_out_begin, (GdnQ81*) xq))
VK_NORM(all_q, true, true)
VK_NORM(keep_q, false, true)
VK_NORM(all, true, false)
VK_NORM(keep, false, false)
#undef VK_NORM

STRATA_KERNEL(verify_kernels_gdn_step_split_out,
              (float* state, const float* hbuf, int C, const float* gate, const float* beta, float* y, int h_k,
               int h_v, int T, const int32_t* n_keep, int t_out_begin),
              gdn_step_split_kernel<true>(state, hbuf, C, gate, beta, y, h_k, h_v, T, n_keep, t_out_begin))
STRATA_KERNEL(verify_kernels_gdn_state_replay,
              (float* state, const float* hbuf, int C, const float* gate, const float* beta, int h_k, int h_v, int T,
               const int32_t* n_keep),
              gdn_state_replay_kernel<kVerifyMaxT>(state, hbuf, C, gate, beta, h_k, h_v, T, n_keep))
STRATA_KERNEL(verify_kernels_gdn_out_norm_q,
              (const float* z, const float* gamma, float eps, float* y, int h_v, int T, const int32_t* n_keep,
               int t_out_begin, void* xq),
              gdn_out_norm_kernel<true>(z, gamma, eps, y, h_v, T, n_keep, t_out_begin, (GdnQ81*) xq))
STRATA_KERNEL(verify_kernels_gdn_out_norm,
              (const float* z, const float* gamma, float eps, float* y, int h_v, int T, const int32_t* n_keep,
               int t_out_begin, void* xq),
              gdn_out_norm_kernel<false>(z, gamma, eps, y, h_v, T, n_keep, t_out_begin, (GdnQ81*) xq))

STRATA_KERNEL(verify_kernels_embedding_gather_dev,
              (const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens, int64_t n,
               int code_bits, int code_bias, int group_elems, unsigned long long row_codes,
               unsigned long long row_groups, float* out),
              embedding_gather_dev_kernel(codes, scales, offsets, tokens, n, code_bits, code_bias, group_elems,
                                          row_codes, row_groups, out))
STRATA_KERNEL(verify_kernels_broadcast_streams, (const float* x, float* R, int64_t n, int hc),
              broadcast_streams_kernel(x, R, n, hc))
STRATA_KERNEL(verify_kernels_copy_indexed,
              (float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n),
              copy_indexed_kernel(dst, src, stride, index, n))
STRATA_KERNEL(verify_kernels_fetch_blobs, (const unsigned long long* src, const int32_t* n, void* dst, long long per),
              fetch_blobs_kernel(src, n, (sycl::uint4*) dst, per))
STRATA_KERNEL(verify_kernels_rebase_ptrs,
              (unsigned long long* ptr, const int32_t* n, unsigned long long base, long long bytes),
              rebase_ptrs_kernel(ptr, n, base, bytes))
STRATA_KERNEL(verify_kernels_add_streams_broadcast, (const float* h, const float* e, float* R, int64_t n, int hc),
              add_streams_broadcast_kernel(h, e, R, n, hc))
STRATA_KERNEL(verify_kernels_ident_hits, (const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count),
              ident_hits_kernel(ids, n, slot, dst, count))
STRATA_KERNEL(verify_kernels_gather_rows_16,
              (const void* src, long long row_e, const int32_t* ids, long long n, void* dst),
              gather_rows_kernel((const sycl::uint4*) src, row_e, ids, n, (sycl::uint4*) dst))
STRATA_KERNEL(verify_kernels_gather_rows_4,
              (const void* src, long long row_e, const int32_t* ids, long long n, void* dst),
              gather_rows_kernel((const uint32_t*) src, row_e, ids, n, (uint32_t*) dst))
STRATA_KERNEL(verify_kernels_gather_rows_1,
              (const void* src, long long row_e, const int32_t* ids, long long n, void* dst),
              gather_rows_kernel((const uint8_t*) src, row_e, ids, n, (uint8_t*) dst))
STRATA_KERNEL(verify_kernels_map_ids, (int32_t* ids, const int32_t* table, int n), map_ids_kernel(ids, table, n))
STRATA_KERNEL(verify_kernels_row_top_prob, (const float* logits, int n_vocab, const int32_t* ids, float* probs),
              row_top_prob_kernel(logits, n_vocab, ids, probs))
STRATA_KERNEL(verify_kernels_mtp_select,
              (const float* R_src, int64_t stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
               int32_t* tok_dst, int32_t* out, int j, const float* probs, float* out_p),
              mtp_select_kernel(R_src, stride, ids, row_dev, R_dst, tok_dst, out, j, probs, out_p))
STRATA_KERNEL(verify_kernels_dense_steps, (const int32_t* cells, int n, int32_t* steps),
              dense_steps_kernel(cells, n, steps))
STRATA_KERNEL(verify_kernels_force_token, (int32_t* tok, const int32_t* force, int j), force_token_kernel(tok, force, j))
STRATA_KERNEL(verify_kernels_copy_row_to_first,
              (const int32_t* row_dev, float* a, int64_t a_n, float* b, int64_t b_n, float* c, int64_t c_n),
              copy_row_to_first_kernel(row_dev, a, a_n, b, b_n, c, c_n))
STRATA_KERNEL(verify_kernels_argmax_rows, (const float* logits, int n, int per_block, void* rows, int32_t* out),
              argmax_rows_kernel(logits, n, per_block, (ArgRow*) rows, out))
STRATA_KERNEL(verify_kernels_row_top_prob_split,
              (const float* logits, int n_vocab, const int32_t* ids, float* probs, void* rows),
              row_top_prob_split_kernel(logits, n_vocab, ids, probs, (TopRow*) rows))
STRATA_KERNEL(verify_kernels_window_ids, (int32_t* steps, int window, int32_t* ids, long long stride),
              window_ids_kernel(steps, window, ids, stride))
STRATA_KERNEL(verify_kernels_wait_flag_ge, (const uint32_t* flag, uint32_t value, uint32_t spin_max),
              wait_flag_ge_kernel(flag, value, spin_max))
STRATA_KERNEL(verify_kernels_resident_plan,
              (const int32_t* ids, int n, int k, const int32_t* res, int n_expert, const uint8_t* cache_base,
               const unsigned long long* slot_off, long long blob, int32_t* pl, long long capx, uint32_t* skip,
               uint32_t ring, const unsigned long long* mir, uint32_t* plan_err, int32_t* usage,
               unsigned long long* stats, int32_t* lru_res, int32_t* lru_owner, uint32_t* lru_last, uint32_t* lru_state,
               unsigned long long* lru_jobs, uint32_t* lru_njobs, int lru_layer, uint32_t lru_guard, uint32_t lru_scan,
               uint32_t lru_sample),
              resident_plan_kernel(ids, n, k, res, n_expert, cache_base, slot_off, blob, pl, capx, skip, ring, mir,
                                   plan_err, usage, stats, lru_res, lru_owner, lru_last, lru_state, lru_jobs, lru_njobs,
                                   lru_layer, lru_guard, lru_scan, lru_sample))
STRATA_KERNEL(verify_kernels_lru_copy, (const unsigned long long* jobs, const uint32_t* njobs, long long chunks),
              lru_copy_kernel(jobs, njobs, chunks))
STRATA_KERNEL(verify_kernels_wait_flag_ge_or,
              (const uint32_t* flag, uint32_t value, const uint32_t* skip, uint32_t spin_max),
              wait_flag_ge_or_kernel(flag, value, skip, spin_max))
STRATA_KERNEL(verify_kernels_copy_i32_unless,
              (int32_t* dst, const int32_t* src, int n, const uint32_t* skip, uint32_t value),
              copy_i32_unless_kernel(dst, src, n, skip, value))
STRATA_KERNEL(verify_kernels_copy_or_zero,
              (float* dst, const float* src, long long n4, const uint32_t* skip, uint32_t value, bool pair),
              copy_or_zero_kernel((sycl::float4*) dst, (const sycl::float4*) src, n4, skip, value, pair))
STRATA_KERNEL(verify_kernels_gpu_stamp, (unsigned long long* buf, int i, const uint64_t* word),
              gpu_stamp_kernel(buf, i, word))
