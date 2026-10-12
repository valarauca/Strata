// bf16_gemv.kern.inc - entry points of bf16_gemv.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(bf16_gemv_naive, (const uint16_t* x, const uint16_t* w, float* y, long long n_in, long long n_out),
              bf16_gemv_naive_kernel(x, w, y, n_in, n_out))
STRATA_KERNEL(bf16_gemv_warp, (const uint16_t* x, const uint16_t* w, float* y, long long n_in, long long n_out),
              bf16_gemv_warp_kernel(x, w, y, n_in, n_out))
STRATA_KERNEL(bf16_gemv_split, (const uint16_t* x, const uint16_t* w, float* y, long long n_in, long long n_out, int tpr),
              bf16_gemv_split_kernel(x, w, y, n_in, n_out, tpr, strata_slm<uint8_t>()))
