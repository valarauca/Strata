// kv_stream.kern.inc - entry points of kv_stream.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(kv_stream_resolve, (KvStreamMap m, const int32_t* ids, const int32_t* steps, int n_q, int cap, int page_size),
              resolve_kernel(m, ids, steps, n_q, cap, page_size))
STRATA_KERNEL(kv_stream_copy, (KvStreamMap m, Runs r), copy_kernel(m, r))
STRATA_KERNEL(kv_stream_reset, (KvStreamMap m), reset_kernel(m))
STRATA_KERNEL(kv_stream_ring, (int32_t* table, long long n_blocks, long long n_slots), ring_kernel(table, n_blocks, n_slots))
