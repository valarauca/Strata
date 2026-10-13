// kv_stream_shared.hpp - the by-value kernel argument kv_stream.dp.cpp and kv_stream.host.cpp share.
#pragma once

#include <cstdint>

namespace strata::kernels {
namespace {

// The per-block byte runs of the (up to four) pool arrays: block b of array i is bytes [b * len, (b + 1) * len).
struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
};

}  // namespace
}  // namespace strata::kernels
