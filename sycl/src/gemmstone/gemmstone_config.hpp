// src/gemmstone/gemmstone_config.hpp - the host configuration the vendored gemmstone (third_party/gemmstone) reads in
// place of oneDNN's. Kernels come out of nGEN's ELF generator as zebin images: gemmstone never sees a GPU runtime
// handle (strata::ze loads the images and owns everything made from them). No post-op is ever fused, so the post-op
// types below are empty stand-ins that refuse to generate one.
#pragma once

#include <bitset>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "ngen_register_allocator.hpp"
#include "npack/neo_packager.hpp"

#define ZEBIN_OUTPUT
#define GEMMSTONE_ASSERTIONS 1
#define GEMMSTONE_ENABLE_COPY_PLAN_DUMP 0

namespace gemmstone {

#define GENERATOR_SUPER(hw) ngen::ELFCodeGenerator<hw>
#define GENERATOR_BASE(hw) ngen::ELFCodeGenerator<hw>
#define FORWARD(hw) NGEN_FORWARD_ELF(hw)
#define GENERATOR_DEBUGINFO \
    ngen::DebugConfig {}

inline int getEnv(const char*, int def) { return def; }

enum class GEMMVerbose { DebugInfo };
inline int getVerbose(GEMMVerbose) { return 0; }
template <typename... Args>
inline void verbosePrintf(const char*, Args...) {}

struct SerializationStream {
    template <typename... T>
    void append(const T&...) {}
};

enum class BinaryOp;

struct PostOps {};

struct PostOpsProblem {
    PostOpsProblem() = default;
    PostOpsProblem(PostOps&&) {}
    struct Entry {
        bool is_sum() const { return false; }
        bool is_binary() const { return false; }
        void set_scale(float) {}
    };
    template <ngen::HW hw>
    struct Injector {
        template <typename G, typename... A>
        Injector(G*, A&&...) {
            throw std::runtime_error("gemmstone: post-op injector requested (Strata fuses no post-ops)");
        }
        int preferred_scratch_regs() const { return 0; }
        int min_scratch_regs() const { return 0; }
        template <typename R>
        void set_scratch(const R&) {}
        template <typename R>
        void compute(const R&) {}
    };
    static const int maxPostOps = 32;

    static BinaryOp toBinaryOp(const Entry&) {
        throw std::runtime_error("gemmstone: binary post-op requested (Strata fuses no post-ops)");
    }
    bool empty() const { return ops.empty(); }
    size_t len() const { return ops.size(); }
    Entry& operator[](size_t idx) { return ops[idx]; }
    const Entry& operator[](size_t idx) const { return ops[idx]; }
    void transpose() {
        std::swap(binaryRow, binaryCol);
        binaryTrans.flip();
    }
    void serialize(SerializationStream&) const {}

    template <typename G, typename RA, typename... A>
    void injectNonBinaryPostOps(const Entry&, G*, RA, A&&...) const {
        throw std::runtime_error("gemmstone: post-op requested (Strata fuses no post-ops)");
    }
    template <typename G, typename RA, typename... A>
    void injectStochasticRound(G*, RA, A&&...) const {
        throw std::runtime_error("gemmstone: stochastic rounding requested (Strata fuses no post-ops)");
    }
    template <typename G, typename RA, typename... A>
    void injectMXScale(G*, RA, A&&...) const {
        throw std::runtime_error("gemmstone: MX scaling requested (Strata fuses no post-ops)");
    }

    std::vector<Entry> ops;
    std::bitset<maxPostOps> binaryRow;
    std::bitset<maxPostOps> binaryCol;
    std::bitset<maxPostOps> binaryBatch;
    std::bitset<maxPostOps> binaryTrans;
    bool fwd = true;
    bool cStochasticRound = false;
};

}  // namespace gemmstone

#define REG_XEHPG_ISA(...)
#define REG_XEHPC_ISA(...)
#define REG_XE2_ISA(...) __VA_ARGS__
#define REG_XE3_ISA(...)
#define REG_XE3P_ISA(...)
