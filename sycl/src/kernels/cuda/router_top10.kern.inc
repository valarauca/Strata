// router_top10.kern.inc - entry points of router_top10.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(router_top10, (const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights),
              router_top10_kernel(logits, n_tokens, n_expert, k, ids, weights, strata_slm<uint8_t>()))
STRATA_KERNEL(router_top10_fast, (const float* logits, int n_expert, int k, int* ids, float* weights),
              router_top10_sycl_fast_kernel<false>(logits, n_expert, k, ids, weights, strata_slm<unsigned char>()))
STRATA_KERNEL(router_top10_fast_serial, (const float* logits, int n_expert, int k, int* ids, float* weights),
              router_top10_sycl_fast_kernel<true>(logits, n_expert, k, ids, weights, strata_slm<unsigned char>()))
