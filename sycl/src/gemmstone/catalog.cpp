// src/gemmstone/catalog.cpp - gemmstone's kernel catalog (third_party/gemmstone/selector/db/kernel.db), in a translation
// unit of its own because it is large.
#include "gemmstone/kernel_catalog.hpp"

namespace gemmstone {
#define _CATALOG_ gemm_catalog
#include "selector/db/kernel.db"
#undef _CATALOG_
}  // namespace gemmstone

namespace strata::gs {

::gemmstone::kcatalog::Catalog catalog() { return ::gemmstone::gemm_catalog; }

}  // namespace strata::gs
