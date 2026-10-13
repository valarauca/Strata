// quantize_act.kern.inc - entry points of quantize_act.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(quantize_act_q8_0, (const float* x, uint8_t* blocks, long long n_blocks),
              quantize_q8_0_kernel(x, blocks, n_blocks))
STRATA_KERNEL(quantize_act_q8_0_scaled, (const float* x, uint8_t* blocks, float* scales, long long n_blocks),
              quantize_q8_0_scaled_kernel(x, blocks, scales, n_blocks))
STRATA_KERNEL(quantize_act_dequant_q8_0, (const uint8_t* blocks, float* x, long long n_blocks),
              dequant_q8_0_kernel(blocks, x, n_blocks))
STRATA_KERNEL(quantize_act_q8_K, (const float* x, uint8_t* blocks, long long n_blocks),
              quantize_q8_K_kernel(x, blocks, n_blocks))
STRATA_KERNEL(quantize_act_dequant_q8_K, (const uint8_t* blocks, float* x, long long n_blocks),
              dequant_q8_K_kernel(blocks, x, n_blocks))
