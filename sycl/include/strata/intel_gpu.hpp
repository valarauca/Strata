// include/strata/intel_gpu.hpp - host-side facts about the Intel GPU the engine runs on, and the memory helpers built
// on the Level Zero runtime (strata/ze/ze.hpp). Host code only.
#pragma once

#include "strata/ze/ze.hpp"

#include <cstdlib>
#include <string>

namespace strata {

/// The kernel driver of the Intel GPU ("xe", "i915", or "" when unknown), from sysfs. The driver decides what the GPU
/// may read: an Arc Pro B70 (xe) page-faults on ordinary host memory a kernel is handed.
const std::string& intel_gpu_driver();

/// Which Intel GPU generation GPU `ordinal` is (Battlemage = Arc B-series / Pro B-series).
enum class IntelGpuGen { Unknown, Alchemist, Battlemage };
IntelGpuGen intel_gpu_gen(int ordinal = -1);

/// Host memory a running kernel polls for the host's stores (doorbell flags, payload rows): pinned and uncached unless
/// STRATA_HOST_UNCACHED=0.
inline void* host_malloc_polled(size_t bytes) {
    static const bool on = [] { const char* v = std::getenv("STRATA_HOST_UNCACHED"); return !(v && v[0] == '0'); }();
    return on ? ze::malloc_host_uncached(bytes) : ze::malloc_host(bytes);
}
inline void host_free_polled(void* p) { ze::free(p); }

/// STRATA_ARENA_ALIAS_CHECK (default on): large device arenas are checked for the xe aliasing fault (ze::malloc_device_guarded).
inline bool arena_alias_guard_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_ARENA_ALIAS_CHECK"); return v == nullptr || v[0] != '0'; }();
    return on;
}

}  // namespace strata
