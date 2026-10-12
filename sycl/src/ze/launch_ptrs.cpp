// src/ze/launch_ptrs.cpp - see include/strata/ze/launch_ptrs.hpp.
#include "strata/ze/launch_ptrs.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/xmx_gemm.hpp"

namespace strata::kernels {

void launch_ptrs(const FusedGrArgs& a, const ze::LaunchPtrs& c) {
    for (const void* p : {(const void*) a.R, (const void*) a.R_out, (const void*) a.bo_prev, (const void*) a.inj_prev,
                          (const void*) a.w_norm, (const void*) a.w_down, (const void*) a.w_up, (const void*) a.w_inject,
                          (const void*) a.lo, (const void*) a.rs, (const void*) a.inject_out, (const void*) a.mixed,
                          (const void*) a.q8_down, (const void*) a.q8_up, (const void*) a.q8_inject,
                          (const void*) a.q8_mixed, (const void*) a.q8_cnt})
        c(p);
}

void launch_ptrs(const KvHostPools& p, const ze::LaunchPtrs& c) {
    for (const void* q : {(const void*) p.k_pool, (const void*) p.v_pool, (const void*) p.k_q, (const void*) p.v_q,
                          (const void*) p.k_scale, (const void*) p.v_scale, (const void*) p.k_q4, (const void*) p.v_q4})
        c(q);
}

void launch_ptrs(const KvStreamMap& m, const ze::LaunchPtrs& c) {
    for (const void* q : {(const void*) m.page_table, (const void*) m.slot_block, (const void*) m.slot_stamp,
                          (const void*) m.slot_ref, (const void*) m.ctl, (const void*) m.miss_block, (const void*) m.miss_slot})
        c(q);
}

void launch_ptrs(const QsaAttnPools& p, const ze::LaunchPtrs& c) {
    for (const void* q : {(const void*) p.k_pool, (const void*) p.v_pool, (const void*) p.k_q, (const void*) p.v_q,
                          (const void*) p.k_scale, (const void*) p.v_scale, (const void*) p.k_q4, (const void*) p.v_q4,
                          (const void*) p.page_table})
        c(q);
}

void launch_ptrs(const RopeTab& t, const ze::LaunchPtrs& c) {
    c(t.cos);
    c(t.sin);
}

namespace xmxg {
void launch_ptrs(const Grp& g, const ze::LaunchPtrs& c) {
    for (int i = 0; i < g.n && i < kXmxGroupMax; ++i) {
        c(g.a[i]);
        c(g.b[i]);
    }
}
}  // namespace xmxg

namespace xmx {
void launch_ptrs(const Src& s, const ze::LaunchPtrs& c) {
    c(s.gate);
    c(s.up);
}
}  // namespace xmx

}  // namespace strata::kernels
