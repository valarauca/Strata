// include/strata/device/sycl_device.hpp - what the engine's device code is compiled against (device builds only).
//
// The kernels are SYCL device code compiled offline by icpx -fsycl-device-only and launched by the Level Zero runtime
// (strata/ze). This header brings in SYCL's device-side library and the device-only part of three SYCLomatic helper
// headers the migrated kernels call (sub-group shuffles, byte permutes, SIMD-in-register math, atomics; copies in
// strata/device/dpct, Apache-2.0 WITH LLVM-exception). No SYCL or dpct host runtime is part of the engine.
#pragma once

#if !defined(__SYCL_DEVICE_ONLY__)
#error "strata/device/sycl_device.hpp is for the device build only"
#endif

#include <limits.h>
#include <math.h>
#include <sycl/sycl.hpp>
#include "strata/device/dpct/atomic.hpp"
#include "strata/device/dpct/util.hpp"
#include "strata/device/dpct/math.hpp"

// The migrated kernels test __CUDA_ARCH__ as DPCT_COMPATIBILITY_TEMP; dpct.hpp set it to 900 for the reference build.
#ifndef DPCT_COMPATIBILITY_TEMP
#define DPCT_COMPATIBILITY_TEMP (900)
#endif
#ifndef __dpct_inline__
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif
#ifndef __dpct_align__
#define __dpct_align__(n) __attribute__((aligned(n)))
#endif
#ifndef __dpct_noinline__
#define __dpct_noinline__ __attribute__((noinline))
#endif

#include "strata/ze/kernel_abi.hpp"
