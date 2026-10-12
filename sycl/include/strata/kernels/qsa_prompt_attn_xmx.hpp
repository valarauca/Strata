// SYCL port: the prompt attention (QSA, selected cells) on XMX joint_matrix - see qsa_prompt_attn_xmx.dp.cpp.
#pragma once
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include <cstddef>
#include <cstdint>
namespace strata::kernels {
/// The same contract as qsa_prompt_attn_batch; returns false (nothing launched) when the shape, pools or device are not
/// handled, so the caller keeps its fallback. Off by default (slower than the fallback): STRATA_PROMPT_ATTN_XMX=1 enables it.
bool qsa_prompt_attn_xmx(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                         const QsaShapes& s, float* attn, int64_t n_q, void* stream);

/// The XMX kernel's geometry and work-group local memory layout, shared by its device code and its launcher.
namespace qsa_xmx {
constexpr int HD = 256, G = 12, SG = 16, NSG = 8, THREADS = SG * NSG, QS = HD + 8;

// local memory layout for CH cells per chunk; the P tiles reuse the K^T region (K is done once the scores are)
template <int CH>
struct Layout {
    static constexpr int PS = CH + 8;                        // P / score row stride
    static constexpr int NT = CH / 16;                       // cell tiles
    static constexpr int UNITS = NT * 4;                     // (cell tile, scale group) score units
    static constexpr size_t q = 2 * 16 * QS * 2;             // Q hi, Q lo (halves)
    static constexpr size_t kp = (size_t) CH * HD * 2;       // packed K^T, then the P tiles
    static constexpr size_t pv = (size_t) 4 * 2 * 16 * PS * 2;
    static constexpr size_t kp_or_pv = kp > pv ? kp : pv;
    static constexpr size_t vp = (size_t) CH * HD * 2;       // packed V
    static constexpr size_t parts = (size_t) (UNITS > 16 ? UNITS : 16) * 256 * 4;   // score units, then P.V tiles
    static constexpr size_t s = (size_t) 16 * PS * 4;        // probabilities
    static constexpr size_t acc = (size_t) 16 * HD * 4;
    static constexpr size_t scal = (size_t) CH * 4 * 4 * 2;  // ks, vs
    static constexpr size_t misc = 80 * 4;
    static constexpr size_t row = (size_t) CH * 8;
    static constexpr size_t bytes = q + kp_or_pv + vp + parts + s + acc + scal + misc + row + 16 * 12;
};
}  // namespace qsa_xmx
}
