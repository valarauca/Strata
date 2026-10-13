// sampler.kern.inc - entry points of sampler.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(sampler_greedy, (const float* logits, int n_vocab, const int* history, int history_len, SamplerParams p,
                               int* out),
              sampler_greedy_kernel(logits, n_vocab, history, history_len, p, p.penalty_last_n, p.penalty_last_n, out,
                                    strata_slm<uint8_t>()))
STRATA_KERNEL(sampler_old, (const float* logits, int n_vocab, int n_tokens, const int* history, int history_len,
                            SamplerParams p, int* out),
              sampler_kernel(logits, n_vocab, n_tokens, history, history_len, p, out, strata_slm<uint8_t>()))
STRATA_KERNEL(sampler_split_part, (const float* logits, int n_vocab, const int* history, int history_len,
                                   SamplerParams p, int k, int n_blocks, int32_t* cand),
              sampler_split_part_kernel(logits, n_vocab, history, history_len, p, k, n_blocks, (sycl::int2*) cand))
STRATA_KERNEL(sampler_split_merge, (const int32_t* cand, int n_blocks, int n_vocab, SamplerParams p, int k, int* out),
              sampler_split_merge_kernel((const sycl::int2*) cand, n_blocks, n_vocab, p, k, out))
STRATA_KERNEL(sampler_one_block, (const float* logits, int n_vocab, const int* history, int history_len,
                                  SamplerParams p, int* out),
              sampler_one_block_kernel(logits, n_vocab, history, history_len, p, out, strata_slm<uint8_t>()))
STRATA_KERNEL(sampler_coupled_stage, (const SamplerParams* mp, const int* mh, SamplerParams* dp, int* ring, int cap),
              coupled_stage_kernel(mp, mh, dp, ring, cap))
STRATA_KERNEL(sampler_coupled_penalize, (float* logits, int nv, const int* id_to_sub, int id_vocab,
                                         const SamplerParams* dp, const int* ring, int cap, int j),
              coupled_penalize_kernel(logits, nv, id_to_sub, id_vocab, dp, ring, cap, j, strata_slm<uint8_t>()))
STRATA_KERNEL(sampler_coupled_split_part, (const float* logits, int nv, int kpart, int n_blocks, int32_t* cand),
              sampler_split_part_kernel(logits, nv, nullptr, 0, SamplerParams{}, kpart, n_blocks, (sycl::int2*) cand))
STRATA_KERNEL(sampler_coupled_merge, (const int32_t* cand, int n_blocks, int nv, int kpart, const SamplerParams* dp,
                                      const int* step_rec, const int* sub_to_id, int* ring, int cap, int j,
                                      int* out_id, float* out_prob),
              coupled_merge_kernel((const sycl::int2*) cand, n_blocks, nv, kpart, dp, step_rec, sub_to_id, ring, cap, j,
                                   out_id, out_prob))
