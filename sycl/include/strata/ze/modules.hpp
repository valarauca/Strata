// include/strata/ze/modules.hpp - the device modules built offline into the binary (cmake/ze_device.cmake).
//
// Each device source group is compiled to SPIR-V, split by sycl-post-link where kernels need different properties,
// and compiled ahead of time by ocloc. The build packs every resulting native binary with its kernel names into one
// bundle per group and links the bundles into the program; the runtime reads them here.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::ze {

/// One native module of a bundle: its bytes in the program image and the kernels it defines.
struct BundleModule {
    std::string name;
    const uint8_t* data = nullptr;
    size_t size = 0;
    std::vector<std::string> kernels;
};

/// Every module of every bundle linked into the program, parsed once.
const std::vector<BundleModule>& bundle_modules();

}  // namespace strata::ze
