// src/ze/modules.cpp - see include/strata/ze/modules.hpp.
//
// Bundle layout (little endian): "SZEB", u32 module count, then per module u64 binary size, u32 kernel-name count,
// the names as NUL-terminated strings, padding to 8 bytes, the binary, padding to 8 bytes.
#include "strata/ze/modules.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
extern const int strata_ze_nbundles;
extern const unsigned char* const strata_ze_bundle_data[];
extern const unsigned long long strata_ze_bundle_size[];
extern const char* const strata_ze_bundle_name[];
}

namespace strata::ze {

namespace {

std::vector<BundleModule> parse() {
    std::vector<BundleModule> out;
    for (int b = 0; b < strata_ze_nbundles; ++b) {
        const uint8_t* p = strata_ze_bundle_data[b];
        const uint8_t* end = p + strata_ze_bundle_size[b];
        auto bad = [&](const char* why) {
            std::fprintf(stderr, "strata ze: device bundle %s is damaged (%s)\n", strata_ze_bundle_name[b], why);
            std::exit(1);
        };
        if (end - p < 8 || std::memcmp(p, "SZEB", 4) != 0) bad("magic");
        uint32_t n = 0;
        std::memcpy(&n, p + 4, 4);
        p += 8;
        for (uint32_t m = 0; m < n; ++m) {
            if (end - p < 12) bad("header");
            uint64_t size = 0;
            uint32_t nk = 0;
            std::memcpy(&size, p, 8);
            std::memcpy(&nk, p + 8, 4);
            p += 12;
            BundleModule bm;
            bm.name = std::string(strata_ze_bundle_name[b]) + "#" + std::to_string(m);
            for (uint32_t k = 0; k < nk; ++k) {
                const uint8_t* z = (const uint8_t*) std::memchr(p, 0, (size_t) (end - p));
                if (z == nullptr) bad("names");
                bm.kernels.emplace_back((const char*) p, (size_t) (z - p));
                p = z + 1;
            }
            p += (8 - ((uintptr_t) (p - strata_ze_bundle_data[b]) & 7)) & 7;
            if ((uint64_t) (end - p) < size) bad("binary");
            bm.data = p;
            bm.size = (size_t) size;
            p += size;
            p += (8 - ((uintptr_t) (p - strata_ze_bundle_data[b]) & 7)) & 7;
            out.push_back(std::move(bm));
        }
    }
    return out;
}

}  // namespace

const std::vector<BundleModule>& bundle_modules() {
    static const std::vector<BundleModule> mods = parse();
    return mods;
}

}  // namespace strata::ze
