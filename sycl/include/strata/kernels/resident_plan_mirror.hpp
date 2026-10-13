// SYCL port: experts mirrored in pinned host memory (GgufExpertSource::mirror) in the device-built verify plan.
#pragma once
#include <cstdint>
namespace strata::kernels {
/// From now on (and in every graph captured after this call) the device-built plan (resident_plan) treats an expert
/// that is not in the VRAM cache but has a non-zero entry in `mirror_table` ([n_layers][n_expert] device-readable
/// addresses, 0 = none) as planned: the expert kernels read it from that address, over PCIe. `d_res` is the residency
/// table of `entries` (n_layers * n_expert) whose per-layer slices resident_plan receives, so the layer of a call is
/// found from its pointer. Each stage of a layer split registers its own residency table with a mirror table on its
/// own GPU (a later call for the same `d_res` replaces it); a slice of an unregistered table plans without a mirror.
/// `stats`, if not null, is three device counters the plans add to: experts the LRU admitted, routed entries whose
/// expert was in VRAM when the plan started, and routed entries.
void resident_plan_set_mirror(const int32_t* d_res, int64_t entries, const unsigned long long* mirror_table,
                              int32_t* usage = nullptr, unsigned long long* stats = nullptr);
/// STRATA_EXPERT_LRU: the device-side expert cache policy. Each plan call (one layer of a window) refreshes the slots
/// it hits and admits its misses: the hand takes the next `sample` slots of [0, lru_end) and evicts the least recently
/// used of those not used in the last `guard` calls; if none qualifies it walks on to the first that does (at most
/// `scan` steps in all; none found = the miss is read from the host mirror, unadmitted). An admitted expert's blob is
/// copied from its mirror address into the slot by a kernel right after the plan, so the window's expert kernels read
/// it from VRAM. Every slot holds one blob of the same size, and every expert has a mirror address.
struct ExpertLru {
    int32_t* res_all = nullptr;          ///< the [n_layers][n_expert] residency table the plans read and update
    int64_t entries = 0;                 ///< n_layers * n_expert
    int32_t* owner = nullptr;            ///< per slot: the res_all index of the expert it holds, -1 = none
    uint32_t* last = nullptr;            ///< per slot: the tick of its last use
    uint32_t* state = nullptr;           ///< [0] the tick (one per plan call), [1] the hand, [2] lru_end
    unsigned long long* jobs = nullptr;  ///< per admitted expert of the last plan: mirror address, slot address
    uint32_t* njobs = nullptr;           ///< the last plan's admitted experts
    uint32_t guard = 150;
    uint32_t scan = 512;
    uint32_t sample = 16;
};
/// From now on (and in every graph captured after this call) the plans on `lru->res_all` run `lru` and copy the experts
/// they admit (a second call for the same table replaces it); nullptr turns every one off.
void resident_plan_set_lru(const ExpertLru* lru);
}
