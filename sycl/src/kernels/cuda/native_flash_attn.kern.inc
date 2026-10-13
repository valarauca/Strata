// native_flash_attn.kern.inc - entry points of native_flash_attn.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(native_flash_attn_attend, (const float* q, const uint16_t* k, const uint16_t* v, const int32_t* step,
                                         int max_context, int padded_length, float scale, float* out,
                                         int32_t* status, const uint16_t* mask),
              attend(q, (const sycl::half*) k, (const sycl::half*) v, step, max_context, padded_length, scale, out,
                     status, (const sycl::half*) mask))
