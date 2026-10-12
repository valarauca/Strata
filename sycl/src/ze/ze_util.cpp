// src/ze/ze_util.cpp - runtime services built on the core (strata/ze/ze.hpp): the arena alias guard, the chunked
// zero fill and host functions.
#include "strata/ze/ze.hpp"
#include "strata/ze/kernel_abi.hpp"
#include "strata/intel_gpu.hpp"

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::ze {

namespace device {
#include "../../src/ze/ze_core.kern.inc"
}  // namespace device

namespace {

uint64_t alias_check(int dev, void* base, size_t bytes, size_t* first_bad) {
    constexpr unsigned long long kBlock = 64 << 10;
    const unsigned long long n = bytes / kBlock;
    if (n == 0) return 0;
    Stream& s = default_stream(dev);
    auto* res = (unsigned long long*) malloc_device(2 * sizeof(unsigned long long), dev);
    unsigned long long init[2] = {0, ~0ull};
    s.memcpy(res, init, sizeof init);
    s.sync();
    constexpr unsigned long long chunk = 1ull << 20;
    for (unsigned long long c0 = 0; c0 < n; c0 += chunk) {
        const unsigned long long cn = n - c0 < chunk ? n - c0 : chunk;
        device::k_ze_tag_write(&s, {(uint32_t) ((cn + 255) / 256), 1, 1}, {256, 1, 1}, 0, (unsigned char*) base, c0, cn, kBlock);
        s.sync();
    }
    for (unsigned long long c0 = 0; c0 < n; c0 += chunk) {
        const unsigned long long cn = n - c0 < chunk ? n - c0 : chunk;
        device::k_ze_tag_check(&s, {(uint32_t) ((cn + 255) / 256), 1, 1}, {256, 1, 1}, 0, (const unsigned char*) base, c0, cn,
                               kBlock, res);
        s.sync();
    }
    unsigned long long out[2] = {0, 0};
    s.memcpy(out, res, sizeof out);
    s.sync();
    free(res);
    if (first_bad) *first_bad = (size_t) out[1];
    return out[0];
}

}  // namespace

void* malloc_device_guarded(size_t bytes, int ordinal, const char* what) {
    if (ordinal < 0) ordinal = current_device();
    void* p = malloc_device(bytes, ordinal);
    if (p == nullptr || bytes < ((size_t) 32 << 20) || !strata::arena_alias_guard_enabled()) return p;
    std::vector<void*> spacers;
    bool clean = false;
    for (int attempt = 0; attempt < 8 && p != nullptr; ++attempt) {
        size_t first = 0;
        const uint64_t bad = alias_check(ordinal, p, bytes, &first);
        if (bad == 0) {
            clean = true;
            if (attempt > 0)
                std::fprintf(stderr, "strata: %s (%.2f GiB) is clean after %d move(s)\n", what, (double) bytes / 1073741824.0, attempt);
            break;
        }
        std::fprintf(stderr,
                     "strata: WARNING: %llu 64-KiB blocks of the %s (%.2f GiB) do not keep their own bytes (first at +%zu MiB): the GPU's "
                     "page table maps them onto other pages; allocating it again behind a %d MiB spacer\n",
                     (unsigned long long) bad, what, (double) bytes / 1073741824.0, first >> 20, 2 * (attempt + 1));
        free(p);
        spacers.push_back(malloc_device((size_t) (2 * (attempt + 1)) << 20, ordinal));
        p = malloc_device(bytes, ordinal);
    }
    for (void* sp : spacers) free(sp);
    if (p != nullptr && !clean)
        std::fprintf(stderr, "strata: WARNING: the %s still has aliased pages after 8 tries (STRATA_ARENA_ALIAS_CHECK=0 skips the check)\n", what);
    return p;
}

void fill_zero(Stream& s, void* p, size_t bytes, size_t chunk) {
    static const bool plain = [] { const char* v = std::getenv("STRATA_CHUNKED_FILL"); return v && v[0] == '0'; }();
    if (plain) {
        s.memset(p, 0, bytes);
        s.sync();
        return;
    }
    uint8_t* b = (uint8_t*) p;
    size_t off = 0;
    while (off < bytes) {
        const size_t n = bytes - off < chunk ? bytes - off : chunk;
        if (n >= (1u << 20) && ((uintptr_t) (b + off) & 7) == 0) {
            const unsigned long long w = n / 8;
            device::k_ze_fill_u64(&s, {(uint32_t) ((w + 255) / 256), 1, 1}, {256, 1, 1}, 0, (unsigned long long*) (b + off), w);
            if (n & 7) s.memset(b + off + w * 8, 0, n & 7);
        } else {
            s.memset(b + off, 0, n);
        }
        s.sync();
        off += n;
    }
}

namespace {

struct HostFuncWorker {
    struct Job {
        std::unique_ptr<Event> ev;
        void (*fn)(void*);
        void* arg;
        Stream* s;
    };
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Job> jobs;
    HostFuncWorker() {
        std::thread([this] {
            for (;;) {
                Job j;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return !jobs.empty(); });
                    j = std::move(jobs.front());
                    jobs.pop_front();
                }
                j.ev->sync();
                j.fn(j.arg);
                j.ev.reset();
                host_func_finished(*j.s);
            }
        }).detach();
    }
};

HostFuncWorker& host_worker(int dev) {
    static std::mutex mu;
    static HostFuncWorker* w[kMaxDevices] = {};
    std::lock_guard<std::mutex> lk(mu);
    if (w[dev] == nullptr) w[dev] = new HostFuncWorker();
    return *w[dev];
}

}  // namespace

void host_func_finished(Stream& s) { s.host_func_done(); }

void host_func(Stream& s, void (*fn)(void*), void* arg) {
    if (s.capturing()) {
        std::fprintf(stderr, "strata ze: host_func on a capturing stream\n");
        std::exit(1);
    }
    auto ev = std::make_unique<Event>();
    s.record(*ev);
    {
        std::lock_guard<std::mutex> lk(s.hf_mu_);
        ++s.hf_pending_;
    }
    HostFuncWorker& w = host_worker(s.device());
    {
        std::lock_guard<std::mutex> lk(w.mu);
        w.jobs.push_back({std::move(ev), fn, arg, &s});
    }
    w.cv.notify_one();
}

}  // namespace strata::ze
