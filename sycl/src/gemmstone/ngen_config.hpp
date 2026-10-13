// src/gemmstone/ngen_config.hpp - nGEN's build configuration for Strata (third_party/ngen reads it through
// NGEN_CONFIG): the same settings oneDNN uses, without the assembler.
#pragma once

#define NGEN_NAMESPACE ngen
#define NGEN_CPP11
#define NGEN_SAFE
#define NGEN_NEO_INTERFACE
#define NGEN_NO_OP_NAMES
#define NGEN_WINDOWS_COMPAT
