#pragma once
// dma_batch.hpp - independent host-to-device blob copies: copy_blobs.
//
// The CUDA engine can submit a group of independent pinned uploads as one cudaMemcpyBatchAsync (PR #807,
// STRATA_DMA_BATCH=1/2). The Intel engine appends them one by one to the stream it is given, which on a copy-engine
// stream is one Level Zero copy command each: STRATA_DMA_BATCH is read, and every mode takes the loop.
#include "strata/ze/ze.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace strata::core {

/// The mode: STRATA_DMA_BATCH (0, 1, 2); unset = the engine's default (`default_mode`, set at the call site's first
/// use; 0 until a measured win makes it 1).
inline int dma_batch_mode(int default_mode = 0) {
    static const int mode = [default_mode] {
        const char* e = std::getenv("STRATA_DMA_BATCH");
        return e != nullptr && e[0] != '\0' ? std::atoi(e) : default_mode;
    }();
    return mode;
}

/// `n` independent copies, dst[i] <- src[i] (bytes[i] each), on `stream`.  The sources are immutable pinned host
/// blobs and the destinations disjoint; the caller keeps them alive until the stream completes.  Ordering with
/// earlier and later work on the stream is that of the copies one by one.  Returns 0.
inline int copy_blobs(void* const* dst, const void* const* src, const size_t* bytes, size_t n, strata::ze::Stream* stream,
                      int batch_mode) {
    (void) batch_mode;
    for (size_t i = 0; i < n; ++i) stream->memcpy(dst[i], src[i], bytes[i]);
    return 0;
}

/// The verifier's form: `n` blobs of `bytes` each into consecutive slots of `dst`.
inline int copy_expert_blobs(uint8_t* dst, const uint8_t* const* src, int n, size_t bytes, strata::ze::Stream* stream,
                             int batch_mode) {
    (void) batch_mode;
    for (int i = 0; i < n; ++i) stream->memcpy(dst + (size_t) i * bytes, src[i], bytes);
    return 0;
}

}  // namespace strata::core
