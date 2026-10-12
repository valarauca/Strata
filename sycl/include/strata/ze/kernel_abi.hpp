// include/strata/ze/kernel_abi.hpp - one declaration per kernel entry point, read by both builds.
//
// A kernel family's entries live in a `*.kern.inc` file as
//
//     STRATA_KERNEL(name, (params...), body)          the default sub-group size (32)
//     STRATA_KERNEL_SG(16, name, (params...), body)   a required sub-group size
//
// The device build (icpx -fsycl-device-only, offline) includes the file after the family's device functions and gets
// `extern "C"` free-function kernels named strata_k_<name> whose body is `body`. The host build includes it and gets
// `strata::kernels::k_<name>`, a strata::ze::Kernel with the same parameter list, which converts each argument to its
// parameter type and appends the launch (strata/ze/ze.hpp). Parameters are plain C++ types on purpose (pointers to
// uint16_t for half or bfloat16 data, PODs from shared headers), so the two compilers agree on every argument's size.
//
// Work-group scratch memory: a body takes it with strata_slm<T>() and the host passes its size in bytes at launch.
#pragma once

#if defined(__SYCL_DEVICE_ONLY__)

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/work_group_scratch_memory.hpp>

/// The work-group's dynamic scratch memory (the size the host passed at launch).
template <class T = unsigned char> inline T* strata_slm() {
    return (T*) sycl::ext::oneapi::experimental::get_work_group_scratch_memory();
}

#define STRATA_KERNEL(name, params, ...)                                                                  \
    extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<3>)) \
    void strata_k_##name params { __VA_ARGS__; }

#define STRATA_KERNEL_SG(sg, name, params, ...)                                                          \
    extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<3>)) \
    [[sycl::reqd_sub_group_size(sg)]] void strata_k_##name params { __VA_ARGS__; }

#else

#include "strata/ze/ze.hpp"

#define STRATA_KERNEL(name, params, ...) \
    inline ::strata::ze::Kernel<void params> k_##name{"__sycl_kernel_strata_k_" #name};

#define STRATA_KERNEL_SG(sg, name, params, ...) STRATA_KERNEL(name, params, __VA_ARGS__)

#endif
