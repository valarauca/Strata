// include/strata/port.hpp - the first include of a header shared by the Intel engine's host and device builds.
//
// The device build (icpx -fsycl-device-only) gets SYCL's device library; the host build gets nothing, so a shared
// header keeps its device functions under `#if defined(SYCL_LANGUAGE_VERSION)` and its host API outside it.
#pragma once

#if defined(__SYCL_DEVICE_ONLY__)
#include "strata/device/sycl_device.hpp"
#endif

/// The math namespace of a function compiled for both sides: SYCL's builtins on the device, the C++ library's on the
/// host (what SYCL's host builtins called).
#if defined(__SYCL_DEVICE_ONLY__)
#define STRATA_MATH sycl
#else
#include <cmath>
#define STRATA_MATH std
#endif
