// src/ze/ze_core.dp.cpp - the runtime's own kernels: the arena alias check and the chunked zero fill (src/ze/ze_util.cpp).
#include "strata/port.hpp"

#include <cstdint>

namespace strata::ze::device {

namespace {

inline void tag_write(unsigned char* base, unsigned long long c0, unsigned long long n, unsigned long long block) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned long long i = (unsigned long long) it.get_global_id(2);
    if (i >= n) return;
    const unsigned long long blk = c0 + i;
    *(volatile uint64_t*) (base + blk * block) = (blk + 1) * 0x9E3779B97F4A7C15ull;
}

inline void tag_check(const unsigned char* base, unsigned long long c0, unsigned long long n, unsigned long long block,
                      unsigned long long* res) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned long long i = (unsigned long long) it.get_global_id(2);
    if (i >= n) return;
    const unsigned long long blk = c0 + i;
    if (*(const volatile uint64_t*) (base + blk * block) != (blk + 1) * 0x9E3779B97F4A7C15ull) {
        sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device> bad(res[0]);
        sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device> lo(res[1]);
        bad.fetch_add(1ull);
        lo.fetch_min(blk * block);
    }
}

inline void fill_u64(unsigned long long* d, unsigned long long n) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned long long i = (unsigned long long) it.get_global_id(2);
    if (i < n) d[i] = 0;
}

}  // namespace

#include "ze_core.kern.inc"

}  // namespace strata::ze::device
