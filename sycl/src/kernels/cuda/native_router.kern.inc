// native_router.kern.inc - entry points of native_router.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(native_router_route_512, (const float* logits, int32_t* ids, float* weights),
              route<512>(logits, ids, weights))
STRATA_KERNEL(native_router_route_256, (const float* logits, int32_t* ids, float* weights),
              route<256>(logits, ids, weights))
