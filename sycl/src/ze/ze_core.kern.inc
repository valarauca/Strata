// ze_core.kern.inc - entry points of ze_core.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(ze_tag_write, (unsigned char* base, unsigned long long c0, unsigned long long n, unsigned long long block),
              tag_write(base, c0, n, block))
STRATA_KERNEL(ze_tag_check, (const unsigned char* base, unsigned long long c0, unsigned long long n, unsigned long long block,
                             unsigned long long* res),
              tag_check(base, c0, n, block, res))
STRATA_KERNEL(ze_fill_u64, (unsigned long long* d, unsigned long long n), fill_u64(d, n))
