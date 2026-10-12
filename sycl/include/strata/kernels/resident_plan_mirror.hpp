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
void resident_plan_set_mirror(const int32_t* d_res, int64_t entries, const unsigned long long* mirror_table,
                              int32_t* usage = nullptr);
}
