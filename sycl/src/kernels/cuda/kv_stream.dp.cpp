// src/kernels/cuda/kv_stream.cu - see include/strata/kernels/kv_stream.hpp.
#include "strata/port.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "kv_stream_shared.hpp"

namespace strata::kernels {
namespace {

constexpr int RT = 1024;   // the resolve block

// Block-wide exclusive prefix sum of one int per thread (RT threads); `total` is the sum over the block.
inline int block_scan(int v, int *warp_sums, int &total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              w = item_ct1.get_local_id(2) >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = dpct::experimental::shift_sub_group_right(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    item_ct1.barrier();
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    item_ct1.barrier();
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    item_ct1.barrier();
    return excl;
}

__dpct_inline__ void resolve_kernel(KvStreamMap m,
                                    const int32_t *__restrict__ ids,
                                    const int32_t *__restrict__ steps, int n_q,
                                    int cap, int page_size) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_nmiss = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_lookups = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_cut = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &warp_sums =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int epoch = m.ctl[0] + 1;
    if (item_ct1.get_local_id(2) == 0) { s_nmiss = 0; s_lookups = 0; }
    item_ct1.barrier();
    // 1. hits take this epoch and their reference bit; a missing block is claimed exactly once (-1 -> -2)
    int lookups = 0;
    for (int q = 0; q < n_q; ++q) {
        const int width = steps[q * kStepCount + kStepWidth];
        const int32_t* qi = ids + (long long) q * cap;
        for (int i = item_ct1.get_local_id(2); i < width; i += RT) {
            const int b = qi[i] / page_size;
            if (i > 0 && qi[i - 1] / page_size == b) continue;   // ids are ascending: one lookup per block
            ++lookups;
            const int sl = m.page_table[b];
            if (sl >= 0) {
                m.slot_stamp[sl] = epoch;
                m.slot_ref[sl] = 1;
            } else if (sl == -1 &&
                       dpct::atomic_compare_exchange_strong<
                           sycl::access::address_space::generic_space>(
                           &m.page_table[b], -1, -2) == -1) {
                m.miss_block[dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(&s_nmiss, 1)] =
                    b;
            }
        }
    }
    if (lookups > 0)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &s_lookups, lookups); // (#783, stuchapin909) most threads see none:
                                  // skip the shared atomic
    item_ct1.barrier();
    // 2. one victim per miss: a clock sweep from the hand. A slot this call uses (stamp == epoch) is never taken;
    //    a referenced one loses its bit as the hand passes it and is taken on the next pass.
    const int need = s_nmiss, n = (int) m.n_slots;
    int hand = m.ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = (int)(((long long)hand + item_ct1.get_local_id(2)) % n);
        const bool mine = m.slot_stamp[j] == epoch;
        const bool cand = !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
        int total = 0;
        const int rank = block_scan(cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (item_ct1.get_local_id(2) == 0) s_cut = RT;
        item_ct1.barrier();
        if (cand && rank == want - 1) s_cut =
            item_ct1.get_local_id(2) +
            1; // the hand stops just past the last slot taken
        item_ct1.barrier();
        const int cut = s_cut;
        if (cand && rank < want) {
            m.miss_slot[got + rank] = j;
            m.slot_stamp[j] = epoch;   // taken: a sweep that wraps around must not take it twice
        } else if (item_ct1.get_local_id(2) < cut && !mine) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int) (((long long) hand + cut) % n);
        item_ct1.barrier();
    }
    // 3. re-point the table; the copy kernel fills the slots
    const int placed = got < need ? got : need;
    for (int k = item_ct1.get_local_id(2); k < need; k += RT) {
        const int b = m.miss_block[k];
        if (k >= placed) { m.page_table[b] = -1; continue; }   // overflow: never happens with a legal n_slots
        const int sl = m.miss_slot[k];
        const int old = m.slot_block[sl];
        if (old >= 0) m.page_table[old] = -1;
        m.slot_block[sl] = b;
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
        m.page_table[b] = sl;
    }
    if (item_ct1.get_local_id(2) == 0) {
        m.ctl[0] = epoch;
        m.ctl[1] = hand;
        m.ctl[2] = placed;
        if (placed < need) m.ctl[3] = 1;
        unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
        c[0] += (unsigned long long) placed;
        c[1] += (unsigned long long) s_lookups;
        c[2] += 1ull;
    }
}

// One block per missed block (grid-stride): copy its runs from the host copy into its slot, 16 B per thread.
__dpct_inline__ void copy_kernel(KvStreamMap m, Runs r) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int need = m.ctl[2];
    for (int k = item_ct1.get_group(2); k < need;
         k += item_ct1.get_group_range(2)) {
        const long long b = m.miss_block[k], sl = m.miss_slot[k];
        for (int a = 0; a < r.n; ++a) {
            const sycl::uint4 *src =
                reinterpret_cast<const sycl::uint4 *>(r.src[a] + b * r.len[a]);
            sycl::uint4 *dst =
                reinterpret_cast<sycl::uint4 *>(r.dst[a] + sl * r.len[a]);
#pragma unroll
            for (int i = item_ct1.get_local_id(2); i < r.len[a] / 16;
                 i += item_ct1.get_local_range(2)) dst[i] = src[i];
        }
    }
}

__dpct_inline__ void reset_kernel(KvStreamMap m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i0 = (long long)item_ct1.get_group(2) *
                             item_ct1.get_local_range(2) +
                         item_ct1.get_local_id(2),
                    st = (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2);
#pragma unroll
    for (long long i = i0; i < m.n_blocks; i += st) m.page_table[i] = -1;
#pragma unroll
    for (long long i = i0; i < m.n_slots; i += st) {
        m.slot_block[i] = -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts) m.ctl[i0] = 0;
}

__dpct_inline__ void ring_kernel(int32_t *table, long long n_blocks,
                                 long long n_slots) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
#pragma unroll
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n_blocks; i += (long long)item_ct1.get_group_range(2) *
                            item_ct1.get_local_range(2))
        table[i] = (int32_t) (i % n_slots);
}

}  // namespace

#include "kv_stream.kern.inc"

}  // namespace strata::kernels
