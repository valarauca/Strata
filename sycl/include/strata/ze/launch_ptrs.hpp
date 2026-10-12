// include/strata/ze/launch_ptrs.hpp - the pointer members of the shared by-value kernel argument types, for the
// STRATA_ZE_CHECK_DEVICES launch check (strata::ze::LaunchPtrs). A type local to one launcher declares its own
// launch_ptrs beside it, in the type's namespace.
#pragma once

#include "strata/ze/ze.hpp"

namespace strata::kernels {

struct FusedGrArgs;
struct KvHostPools;
struct KvStreamMap;
struct QsaAttnPools;
struct RopeTab;

/// Hands each pointer member of `a` to `c`.
void launch_ptrs(const FusedGrArgs& a, const ze::LaunchPtrs& c);
/// Hands each pointer member of `p` to `c`.
void launch_ptrs(const KvHostPools& p, const ze::LaunchPtrs& c);
/// Hands each pointer member of `m` to `c`.
void launch_ptrs(const KvStreamMap& m, const ze::LaunchPtrs& c);
/// Hands each pointer member of `p` to `c`.
void launch_ptrs(const QsaAttnPools& p, const ze::LaunchPtrs& c);
/// Hands each pointer member of `t` to `c`.
void launch_ptrs(const RopeTab& t, const ze::LaunchPtrs& c);

namespace xmxg {
struct Grp;
/// Hands the weight pointers of the group's `n` experts to `c`.
void launch_ptrs(const Grp& g, const ze::LaunchPtrs& c);
}  // namespace xmxg

namespace xmx {
struct Src;
/// Hands the gate and up row pointers to `c`.
void launch_ptrs(const Src& s, const ze::LaunchPtrs& c);
}  // namespace xmx

}  // namespace strata::kernels
