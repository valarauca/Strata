// elementwise_shared.hpp - the by-value kernel argument elementwise.dp.cpp and elementwise.host.cpp share.
#pragma once

#include "strata/kernels/elementwise.hpp"

namespace strata::kernels {

struct MappedCopies {
    MappedCopy c[8];
};

}  // namespace strata::kernels
