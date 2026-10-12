// src/gemmstone/gemmstone.cpp - see include/strata/gemmstone.hpp. The kernel selection, strategy finalization, argument
// list and dispatch follow oneDNN's gemmstone driver (src/gpu/intel/gemm/jit/gen_kernel.cpp, gemm/jit.cpp and
// gemm/jit/walk_orders.hpp at the vendored commit), cut down to what Strata asks for: one launch per product (no
// m/n/k blocking), alpha 1, beta 0 or 1, no post-ops, no batch, and no kernel that needs atomics, a temporary C or a
// zeroed status buffer (the selection is made deterministic, so those are never chosen).
//
// What gemmstone and nGEN borrow, and what keeps it alive:
//   - the kernel catalog: kcatalog::Catalog and every Entry pointer select() returns point into the static array of
//     catalog.cpp (static storage, never freed); Built keeps copies, never an Entry pointer;
//   - MatchParams: its precision, layout and tag strings point into its own buffer or at string literals, and a copy
//     re-points them at the copy's buffer, so patterns are copied, never referenced, while the vector holding them
//     grows; select() borrows that vector only for the call;
//   - GEMMProblem, GEMMStrategy, EvaluateParams, the hardware description: plain values, copied into Built;
//   - the interface handler: Generator::gemm copies it, and the handler is declared before the generator so it outlives
//     the generator in any case; the generator lives only inside build() and returns its binary by value;
//   - GPU objects: none are ever passed in. ze::load_native copies the binary into a module the runtime owns, with its
//     kernel, until the process exits; the static cache only holds those handles, never destroys them;
//   - a launch: the argument values are copied into the command list by the append, so the locals they point at only
//     need to live through the call; the buffers they name must stay allocated until the GPU has run the launch
//     (strata/gemmstone.hpp).
#include "strata/gemmstone.hpp"
#include "strata/ze/ze.hpp"

#include "gemmstone/generator.hpp"
#include "gemmstone/kernel_catalog.hpp"
#include "gemmstone/kernel_evaluator.hpp"
#include "gemmstone/kernel_selector.hpp"
#include "gemmstone/strategy_parser.hpp"
#include "generator/pieces/compute_utils.hpp"
#include "generator/pieces/ngen_object_helpers.hpp"
#include "npack/neo_packager.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace strata::gs {

::gemmstone::kcatalog::Catalog catalog();

namespace {

namespace gm = ::gemmstone;
using gm::LoopK;
using gm::LoopM;
using gm::LoopN;
using gm::Type;

struct Key {
    int dev, weights, beta, n_class;
    int64_t m, k;
    int align_a, align_b, align_c;
    bool operator<(const Key& o) const {
        return std::tie(dev, weights, beta, n_class, m, k, align_a, align_b, align_c) <
               std::tie(o.dev, o.weights, o.beta, o.n_class, o.m, o.k, o.align_a, o.align_b, o.align_c);
    }
};

struct Hardware {
    ngen::Product product;
    ngen::HW hw = ngen::HW::Unknown;
    int eu_count = 0;
    int eus_per_wg = 8;
};

struct Built {
    bool ok = false;
    Hardware hw;
    ze::NativeKernel kernel;
    gm::GEMMProblem problem;
    gm::GEMMStrategy strategy;
    gm::CommonDriverInfo info{};
    gm::EvaluateAuxOutput aux;
    uint32_t nargs = 0;
};

std::mutex g_mu;
std::map<Key, std::unique_ptr<Built>> g_cache;

bool verbose() {
    static const bool on = [] { const char* v = std::getenv("STRATA_GS_VERBOSE"); return v && v[0] && v[0] != '0'; }();
    return on;
}

int pow2_align(uint64_t v) {
    int a = 1;
    while (a < 128 && (v & (uint64_t) (2 * a - 1)) == 0) a *= 2;
    return a;
}

int n_class(int64_t T) {
    int c = 1;
    while (c < T && c < 8192) c *= 2;
    return c;
}

uint32_t reciprocal(uint32_t x) {
    if (x == 0) return 0;
    int lg = 0;
    while ((2ull << lg) <= x) ++lg;
    return (uint32_t) (((uint64_t(0x100000000) << lg) + x - 1) / x);
}

bool hardware_of(int dev, Hardware& hw) {
    const ze::DeviceInfo& di = ze::device_info(dev);
    if (di.ip_version != 0) hw.product = ngen::npack::decodeHWIPVersion(di.ip_version);
    else if (di.battlemage) hw.product = ngen::Product(ngen::ProductFamily::BMG, 0, ngen::PlatformType::Discrete);
    else return false;
    hw.product.type = di.integrated ? ngen::PlatformType::Integrated : ngen::PlatformType::Discrete;
    hw.hw = ngen::getCore(hw.product.family);
    hw.eu_count = (int) di.eu_count;
    return hw.hw == ngen::HW::Xe2 && hw.eu_count > 0;
}

Type a_type(Weights w) { return w == Weights::Q4 ? Type::s4 : w == Weights::BF16 ? Type::bf16 : Type::f16; }
Type b_type(Weights w) { return w == Weights::BF16 ? Type::bf16 : Type::f16; }

gm::GEMMProblem make_problem(const Key& key, const Hardware& hw) {
    gm::GEMMProblem p;
    const Weights w = (Weights) key.weights;
    p.Ta = p.Ta_ext = a_type(w);
    p.Tb = p.Tb_ext = b_type(w);
    p.Tc = p.Tc_ext = Type::f32;
    p.Ts = Type::f32;
    p.A.layout = gm::MatrixLayout::T;
    p.B.layout = gm::MatrixLayout::N;
    p.C.layout = gm::MatrixLayout::N;
    p.A.crosspack = p.B.crosspack = p.C.crosspack = 1;
    p.A.packSize = p.B.packSize = p.C.packSize = 0;
    p.A.setAlignment(key.align_a);
    p.B.setAlignment(key.align_b);
    p.C.setAlignment(key.align_c);
    p.AO.layout = p.BO.layout = gm::MatrixLayout::N;
    p.AO.crosspack = p.BO.crosspack = 1;
    p.AO.packSize = p.BO.packSize = 0;
    p.A_scale = p.Ag = p.AO;
    p.B_scale = p.Bg = p.BO;
    p.BO.layout = gm::MatrixLayout::T;
    p.B_scale.layout = gm::MatrixLayout::T;
    p.Bg.layout = gm::MatrixLayout::T;
    if (w == Weights::Q4) {
        p.asPtrDims = 2;
        p.aqGroupM = 1;
        p.aqGroupK = 32;
        p.Ta_scale = Type::f16;
        p.A_scale.setAlignment(2);
    }
    p.alpha = 1;
    p.beta = key.beta;
    p.product = hw.product;
    return p;
}

std::vector<gm::MatchParams> match_params(const gm::GEMMProblem& problem, const Key& key, const Hardware& hw, std::string& tags) {
    using namespace gm::kcatalog;
    std::vector<gm::MatchParams> mp;
    mp.reserve(16);
    gm::MatchParams base(hw.hw, true, hw.product, problem);
    base.precisionCExt = '\0';
    base.sizes.m = key.m;
    base.sizes.n = key.n_class;
    base.sizes.k = key.k;
    base.sizes.batch = 1;
    base.stepping = hw.product.stepping;
    base.ignoreCase = true;
    auto t = const_cast<char*>(base.tags);
    while (*t) t++;
    if (problem.A.alignment % 16 == 0) *t++ = ReqBlock2DA;
    if (problem.B.alignment % 16 == 0) *t++ = ReqBlock2DB;
    if (problem.C.alignment % 16 == 0) *t++ = ReqBlock2DC;
    *t = '\0';
    tags = base.tags;
    mp.push_back(base);
    auto add = [&](const gm::MatchParams start, const char* (*m)(Type)) {
        if (m(problem.Ta_ext)) {
            mp.push_back(start);
            mp.back().selector.precisions[0] = m(problem.Ta_ext);
        }
        if (m(problem.Tb_ext)) {
            mp.push_back(start);
            mp.back().selector.precisions[1] = m(problem.Tb_ext);
        }
        if (m(problem.Ta_ext) && m(problem.Tb_ext)) {
            mp.push_back(start);
            mp.back().selector.precisions[0] = m(problem.Ta_ext);
            mp.back().selector.precisions[1] = m(problem.Tb_ext);
        }
    };
    add(base, [](Type dt) -> const char* { return dt.isInt4() ? "[FO]" : nullptr; });
    const size_t n = mp.size();
    for (size_t i = 0; i < n; ++i) add(mp[i], [](Type dt) -> const char* { return dt == Type::bf16 ? "H" : nullptr; });
    return mp;
}

// gen_nocopy_desc_t::finalize and gen_desc_t::finalize, for one catalog entry; false: the entry does not fit.
bool finalize(const gm::kcatalog::Entry& entry, const gm::EvaluateParams& eval, const std::string& tags, const Key& key,
              const Hardware& hw, Built& b) {
    gm::GEMMProblem& problem = b.problem;
    gm::GEMMStrategy& strategy = b.strategy;
    Type Ta_ext_new, Ta_new, Tb_ext_new, Tb_new;
    gm::parsePrecisions(entry.selector.precisions[0], Ta_ext_new, Ta_new);
    gm::parsePrecisions(entry.selector.precisions[1], Tb_ext_new, Tb_new);
    const Type Tc_new = gm::charToType(entry.selector.precisions[2][0]);
    auto update = [](Type& T, Type T_new) {
        if (T.isFP() && T.bits() == 16 && T_new.isFP() && T_new.bits() == 16) return;
        T = T.isSigned() ? T_new.asSigned() : T_new.asUnsigned();
    };
    update(problem.Ta, Ta_new);
    update(problem.Tb, Tb_new);
    update(problem.Tc, Tc_new);
    if (problem.Ts == Type::invalid) problem.Ts = problem.Tc;
    b.aux = gm::EvaluateAuxOutput();
    gm::evaluate(entry, eval, b.aux);

    if (!gm::isPacked(problem.A.layout) && problem.Ta_ext.paddedSize() >= problem.Ta.paddedSize())
        problem.A.setAlignment(std::max(problem.Ta_ext.paddedSize(), std::min(entry.driverInfo.alignment[0], (int) problem.A.alignment)));
    if (!gm::isPacked(problem.B.layout) && problem.Tb_ext.paddedSize() >= problem.Tb.paddedSize())
        problem.B.setAlignment(std::max(problem.Tb_ext.paddedSize(), std::min(entry.driverInfo.alignment[1], (int) problem.B.alignment)));
    if (!gm::isPacked(problem.C.layout))
        problem.C.setAlignment(std::max(problem.Tc_ext.paddedSize(), entry.restrictions.alignment[2]));
    problem.CO.setAlignment(problem.Tco.paddedSize());

    strategy = gm::GEMMStrategy(hw.hw, hw.product.stepping);
    strategy.unroll[LoopM] = entry.driverInfo.unroll[LoopM];
    strategy.unroll[LoopN] = entry.driverInfo.unroll[LoopN];
    gm::parseStrategy(entry.strategy, problem, strategy);
    gm::modifyStrategy(strategy, b.aux);
    strategy.panelCheck |= (gm::isPacked(problem.A.layout) || gm::isPacked(problem.B.layout));

    if (strategy.kParallelLocal && problem.quantized2DA())
        b.aux.k0 = (b.aux.k0 + problem.aqGroupK - 1) / problem.aqGroupK * problem.aqGroupK;

    if (strategy.raHW == hw.hw) strategy.raHW = ngen::HW::XeHPC;
    bool block_2d_a = false, block_2d_b = false;
    for (char c : tags) {
        block_2d_a |= (c == gm::kcatalog::ReqBlock2DA);
        block_2d_b |= (c == gm::kcatalog::ReqBlock2DB);
    }
    if (block_2d_a && strategy.legalAAlignment(problem, 16)) problem.A.setAlignment(std::max<int>(problem.A.alignment, 16));
    if (block_2d_b && strategy.legalBAlignment(problem, 16)) problem.B.setAlignment(std::max<int>(problem.B.alignment, 16));

    if (strategy.kParallel) {
        if (key.k <= b.aux.k0 * b.aux.wgK) {
            strategy.kParallel = false;
            strategy.C.atomic = false;
            strategy.CO.atomic = false;
        }
    }
    if (strategy.barrierFreq > 0 && key.k < 2 * strategy.barrierFreq) strategy.barrierFreq = 0;
    if (strategy.fixedSystolic) strategy.GRFs = 256;

    {
        const int wg_tile_m = strategy.wg[LoopM] * strategy.unroll[LoopM];
        const int wg_tile_n = strategy.wg[LoopN] * strategy.unroll[LoopN];
        if (wg_tile_m > 0 && wg_tile_n > 0) {
            const int64_t m_tiles = (key.m + wg_tile_m - 1) / wg_tile_m;
            const int64_t n_tiles = (key.n_class + wg_tile_n - 1) / wg_tile_n;
            int64_t thread_per_tg = strategy.wg[LoopM] * strategy.wg[LoopN];
            if (!strategy.kParallelVariable) thread_per_tg *= std::max(strategy.wg[LoopK], 1);
            const int64_t thread_gpu = (int64_t) hw.eu_count * hw.product.threadsPerEU(strategy.GRFs);
            const int64_t tiles_gpu = thread_gpu / thread_per_tg;
            const bool use_linear = (m_tiles * n_tiles <= tiles_gpu);
            bool use_linear_m = (m_tiles * m_tiles <= 2 * tiles_gpu);
            bool use_linear_n = (n_tiles * n_tiles <= 2 * tiles_gpu);
            if (strategy.fused && (strategy.wg[LoopM] % 2 || strategy.wg[LoopN] % 2)) use_linear_m = use_linear_n = false;
            if (use_linear) {
                if (strategy.kParallelVariable) strategy.cWalkOrder = gm::WalkOrder::SimpleLinear;
                else if (strategy.kParallel && (strategy.fuseBeta || strategy.fusePostOps)) {
                    strategy.persistent = false;
                    strategy.cWalkOrder = gm::WalkOrder::SimpleLinear;
                } else {
                    strategy.persistent = false;
                    strategy.cWalkOrder = gm::WalkOrder::HW2D;
                    strategy.blocking[LoopM] = 16777216;
                    strategy.blocking[LoopN] = 16777216;
                }
            } else if (use_linear_m || use_linear_n) {
                if (use_linear_n && !use_linear_m) {
                    strategy.loopOrder[0] = LoopN;
                    strategy.loopOrder[1] = LoopM;
                } else if (use_linear_m && !use_linear_n) {
                    strategy.loopOrder[0] = LoopM;
                    strategy.loopOrder[1] = LoopN;
                }
                strategy.cWalkOrder = gm::WalkOrder::SimpleLinear;
            }
        }
    }

    strategy.systolicAvailable = true;
    problem.autoTypeConversions(strategy.systolicAvailable);
    gm::adjustStrategy(hw.hw, problem, strategy, tags.c_str());
    try {
        strategy.preflight(hw.hw, problem);
    } catch (...) {
        return false;
    }
    if (problem.aOffset2D() || problem.aScale2D())
        if (problem.aqGroupK % strategy.aqGroupKGranularity()) return false;
    if (problem.aScale2D() && problem.aqGroupK % gm::minOuterProductCount(problem, strategy) != 0) return false;

    strategy.kInterleaveChunk = std::min(strategy.kInterleaveChunk, (int) b.aux.k0);
    if (strategy.kInterleave) b.aux.wgK = strategy.wg[LoopK];
    if (b.aux.wgK > strategy.wg[LoopK]) b.aux.wgK = strategy.wg[LoopK];
    b.info = gm::Generator<ngen::HW::Xe2>::driverInfo(problem, strategy);

    if (b.info.needsTempC() || b.info.fusedBeta() || b.info.fusedPostOps() || b.info.kParallel() ||
        b.info.kParallelVariable() || b.info.needsAGroupSums() || b.info.needsBGroupSums())
        return false;
    return true;
}

// gen_kernel_t::init_interface, for the arguments Strata's problems have.
ngen::NEOInterfaceHandler make_interface(const gm::GEMMProblem& problem, const gm::GEMMStrategy& strategy, ngen::HW hw,
                                         const std::string& name, uint32_t& nargs) {
    using ngen::DataType;
    using ngen::ExternalArgumentType;
    ngen::NEOInterfaceHandler iface(hw);
    nargs = 0;
    auto arg = [&](const char* n, DataType t) { iface.newArgument(n, t); ++nargs; };
    auto ptr = [&](const char* n, ngen::GlobalAccessType a) { iface.newArgument(n, ExternalArgumentType::GlobalPtr, a); ++nargs; };
    ptr("A", strategy.A.getGlobalAccessType());
    ptr("B", strategy.B.getGlobalAccessType());
    ptr("C", strategy.C.getGlobalAccessType());
    arg("offset_A", DataType::q);
    arg("offset_B", DataType::q);
    arg("offset_C", DataType::q);
    arg("lda", DataType::d);
    arg("ldb", DataType::d);
    arg("ldc", DataType::d);
    arg("m", DataType::d);
    arg("n", DataType::d);
    arg("k", DataType::d);
    arg("alpha_real", problem.Ts.ngen());
    arg("beta_real", problem.Ts.ngen());
    if (problem.aScale2D()) ptr("a_scale_ptr", strategy.A_scale.getGlobalAccessType());
    if (problem.aOffset2D() || problem.aScale2D()) arg("ldaq", DataType::d);
    arg("flags", DataType::ud);
    if ((strategy.kParallel || strategy.kParallelLocal) && !strategy.kParallelVariable) arg("k0", DataType::d);
    if (strategy.linearOrder()) {
        arg("group_count_m", DataType::ud);
        arg("group_count_n", DataType::ud);
    }
    if (strategy.cWalkOrder == gm::WalkOrder::SimpleLinear) arg("group_count_recip", DataType::ud);
    else if (strategy.cWalkOrder == gm::WalkOrder::Hilbertlike) {
        arg("hilbert_vd", DataType::ud);
        arg("hilbert_uvd_recip", DataType::ud);
        arg("hilbert_bail", DataType::ud);
    } else if (strategy.cWalkOrder == gm::WalkOrder::Boustrophedon) {
        arg("bslice", DataType::d);
        arg("bthresh", DataType::d);
    }
    if (strategy.persistent) arg("group_stride", DataType::ud);
    if (strategy.variableSLM()) {
        iface.newArgument("local_mem", ExternalArgumentType::LocalPtr);
        ++nargs;
    }
    if (problem.aoPtrDims >= 1 || problem.aScale2D()) arg("offset_Aq", DataType::q);
    iface.allowArgumentRearrangement(false);
    iface.externalName(name);
    return iface;
}

std::unique_ptr<Built> build(const Key& key, const Hardware& hw, std::vector<uint8_t>* image) {
    auto out = std::make_unique<Built>();
    const gm::GEMMProblem problem0 = make_problem(key, hw);
    std::string tags;
    const std::vector<gm::MatchParams> mp = match_params(problem0, key, hw, tags);
    gm::EvaluateParams eval;
    eval.sizes.m = key.m;
    eval.sizes.n = key.n_class;
    eval.sizes.k = key.k;
    eval.sizes.batch = 1;
    eval.alpha = 1.0;
    eval.beta = key.beta;
    eval.product = hw.product;
    eval.euCount = hw.eu_count;
    eval.postOps = false;
    eval.cConvert = false;
    eval.batch = false;
    eval.deterministic = true;
    eval.Tc_ext = problem0.Tc_ext;
    gm::EvaluateAuxOutput aux;
    const std::vector<const gm::kcatalog::Entry*> entries = gm::select(catalog(), (int) mp.size(), mp.data(), eval, aux);
    static int serial = 0;
    for (const gm::kcatalog::Entry* e : entries) {
        Built b;
        b.hw = hw;
        b.problem = problem0;
        if (!finalize(*e, eval, tags, key, hw, b)) continue;
        const std::string name = "strata_gs_" + std::to_string(serial++);
        std::vector<uint8_t> binary;
        try {
            const ngen::NEOInterfaceHandler iface = make_interface(b.problem, b.strategy, hw.hw, name, b.nargs);
            gm::Generator<ngen::HW::Xe2> g(hw.product);
            g.setStepping(hw.product.stepping);
            g.gemm(b.problem, b.strategy, iface);
            binary = g.getBinary();
        } catch (const std::exception& ex) {
            if (verbose()) std::fprintf(stderr, "strata gs: %s: generation failed: %s\n", e->strategy, ex.what());
            continue;
        }
        if (image != nullptr) {
            *image = std::move(binary);
            b.ok = true;
            *out = std::move(b);
            return out;
        }
        b.kernel = ze::load_native(key.dev, binary, name);
        if (!b.kernel) continue;
        if (b.kernel.nparams() != b.nargs) {
            std::fprintf(stderr, "strata gs: kernel %s takes %u arguments, the interface declared %u\n", name.c_str(),
                         b.kernel.nparams(), b.nargs);
            continue;
        }
        if (verbose())
            std::fprintf(stderr, "strata gs: weights %d beta %d m %lld n<=%d k %lld: %s (%s), unroll %dx%d wg %dx%dx%d sg %d, "
                         "%zu bytes\n", key.weights, key.beta, (long long) key.m, key.n_class, (long long) key.k,
                         e->strategy, name.c_str(), b.info.unroll[LoopM], b.info.unroll[LoopN], b.info.wg[LoopM],
                         b.info.wg[LoopN], b.info.wg[LoopK], b.info.subgroupSize, binary.size());
        b.ok = true;
        *out = std::move(b);
        return out;
    }
    if (verbose())
        std::fprintf(stderr, "strata gs: no kernel for weights %d beta %d m %lld n<=%d k %lld (%zu candidates)\n", key.weights,
                     key.beta, (long long) key.m, key.n_class, (long long) key.k, entries.size());
    return out;
}

const Built* lookup(const Key& key) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_cache.find(key);
    if (it == g_cache.end()) {
        Hardware hw;
        it = g_cache.emplace(key, hardware_of(key.dev, hw) ? build(key, hw, nullptr) : std::make_unique<Built>()).first;
    }
    return it->second->ok ? it->second.get() : nullptr;
}

Key make_key(Weights w, int dev, int64_t T, int64_t N, int64_t K, bool accumulate, int align_a, int align_b, int align_c) {
    Key key;
    key.dev = dev;
    key.weights = (int) w;
    key.beta = accumulate ? 1 : 0;
    key.n_class = n_class(T);
    key.m = N;
    key.k = K;
    key.align_a = align_a;
    key.align_b = align_b;
    key.align_c = align_c;
    return key;
}

}  // namespace

std::vector<uint8_t> probe(Weights w, int64_t T, int64_t N, int64_t K, bool accumulate, int eu_count, std::string& what) {
    Hardware hw;
    hw.product = ngen::Product(ngen::ProductFamily::BMG, 0, ngen::PlatformType::Discrete);
    hw.hw = ngen::HW::Xe2;
    hw.eu_count = eu_count;
    const Key key = make_key(w, -1, T, N, K, accumulate, 64, 64, 64);
    std::vector<uint8_t> image;
    const std::unique_ptr<Built> b = build(key, hw, &image);
    if (!b->ok) {
        what = "no kernel";
        return {};
    }
    const gm::CommonDriverInfo& i = b->info;
    char buf[256];
    std::snprintf(buf, sizeof buf, "unroll %dx%d wg %dx%dx%d sg %d grf %d slm %d k0 %lld flags 0x%llx order 0x%x",
                  i.unroll[LoopM], i.unroll[LoopN], i.wg[LoopM], i.wg[LoopN], i.wg[LoopK], i.subgroupSize, i.grfCount, i.slm,
                  (long long) b->aux.k0, (unsigned long long) i.flags, (unsigned) i.loopOrder[0]);
    what = std::string(gm::unparseStrategy(hw.hw, b->problem, b->strategy)) + " | " + buf;
    return image;
}

bool gemm(Weights w, const uint16_t* X, int64_t ldx, const void* W, int64_t ldw, const uint16_t* scales, float* Y,
          int64_t ldy, int64_t T, int64_t N, int64_t K, bool accumulate, void* stream) {
    if (T <= 0 || N <= 0) return true;
    if (K <= 0 || T > INT32_MAX || N > INT32_MAX || K > INT32_MAX || ldx > INT32_MAX || ldw > INT32_MAX || ldy > INT32_MAX)
        return false;
    if (w == Weights::Q4 && (scales == nullptr || K % 32 != 0 || ldw % 2 != 0)) return false;
    ze::Stream* s = ze::stream_of(stream);
    const uint64_t a_bytes_ld = w == Weights::Q4 ? (uint64_t) ldw / 2 : (uint64_t) ldw * 2;
    const Key key = make_key(w, s->device(), T, N, K, accumulate, pow2_align((uintptr_t) W | a_bytes_ld),
                             pow2_align((uintptr_t) X | ((uint64_t) ldx * 2)), pow2_align((uintptr_t) Y | ((uint64_t) ldy * 4)));
    const Built* b = lookup(key);
    if (b == nullptr) return false;
    const gm::CommonDriverInfo& info = b->info;
    const gm::GEMMProblem& problem = b->problem;
    const gm::GEMMStrategy& strategy = b->strategy;

    const int32_t m = (int32_t) N, n = (int32_t) T, k = (int32_t) K;
    void* a_ptr = const_cast<void*>(W);
    void* b_ptr = const_cast<uint16_t*>(X);
    void* c_ptr = Y;
    void* s_ptr = const_cast<uint16_t*>(scales);
    int64_t off_a = 0, off_b = 0, off_c = 0, off_aq = 0;
    int32_t lda = (int32_t) ldw, ldb = (int32_t) ldx, ldc = (int32_t) ldy;
    float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    int32_t ldaq = m;
    uint32_t flags = 0;
    int32_t k0 = (int32_t) b->aux.k0;

    uint64_t gws[3], lws[3];
    const uint64_t lws_k = (uint64_t) std::max(b->aux.wgK, 1);
    gws[0] = (uint64_t) (m + info.unroll[LoopM] - 1) / info.unroll[LoopM];
    gws[1] = (uint64_t) (n + info.unroll[LoopN] - 1) / info.unroll[LoopN];
    gws[2] = lws_k;
    lws[0] = (uint64_t) info.wg[LoopM];
    lws[1] = (uint64_t) info.wg[LoopN];
    lws[2] = lws_k;
    {
        const uint64_t chunk = (uint64_t) info.cInterleaveChunk(problem.Tc_ext) * lws[1];
        gws[1] = (gws[1] + chunk - 1) / chunk * chunk;
        if (info.cInterleaveEnabled() && ((int64_t) ldc * problem.Tc.size() % 64 > 0)) {
            const int wg_tile_m = info.wgTile(LoopM);
            const int max_shift = 64 / problem.Tc_ext.size() - 1;
            gws[0] += lws[0] * (uint64_t) ((wg_tile_m + max_shift + wg_tile_m - 1) / wg_tile_m);
        }
    }
    if (info.isNMK()) {
        std::swap(lws[0], lws[1]);
        std::swap(gws[0], gws[1]);
    }
    if (info.fusedEUs() && lws[0] > 1) gws[0] = (gws[0] + 1) / 2 * 2;
    lws[2] = std::min(lws[2], gws[2]);
    int last_non_1 = 2;
    for (; last_non_1 >= 0 && (gws[last_non_1] == 1 || lws[last_non_1] == 1); last_non_1--) {}
    for (int d = 0; d < 3; d++) {
        if (info.fixedWG() || gws[d] > lws[d]) gws[d] = (gws[d] + lws[d] - 1) / lws[d] * lws[d];
        else {
            if (d < last_non_1) {
                uint64_t p = 1;
                while (p < gws[d]) p <<= 1;
                gws[d] = p;
            }
            lws[d] = gws[d];
        }
    }
    lws[1] *= (uint64_t) info.wgExpand;
    gws[1] *= (uint64_t) info.wgExpand;

    uint32_t groups_m = 0, groups_n = 0, gc_recip = 0, h_vd = 0, h_uvd_recip = 0, h_bail = 0, group_stride = 0;
    int32_t bslice = 0, bthresh = 0;
    if (info.isLinearOrder()) {
        const int m_index = info.isNMK() ? 1 : 0, n_index = info.isNMK() ? 0 : 1;
        groups_m = (uint32_t) (gws[m_index] / lws[m_index]);
        groups_n = (uint32_t) (gws[n_index] / lws[n_index]);
        uint32_t group_count = groups_m * groups_n;
        const Hardware& hw = b->hw;
        const uint32_t ss_count = (uint32_t) (hw.eu_count / hw.eus_per_wg);
        const uint32_t thread_per_ss = (uint32_t) (hw.eu_count * hw.product.threadsPerEU(info.grfCount)) / ss_count;
        const uint32_t thread_per_tg = (uint32_t) (lws[0] * lws[1] * lws[2]);
        const uint32_t concurrent_tg = (thread_per_ss / thread_per_tg) * ss_count;
        if (info.isSimpleLinear()) gc_recip = reciprocal(info.isMNK() ? groups_m : groups_n);
        else if (info.isHilbert()) {
            const double ratio = double(groups_n) / double(groups_m);
            if (ratio >= 1) {
                h_vd = (uint32_t) std::ceil(groups_n / std::round(2 * ratio));
                h_uvd_recip = reciprocal(groups_m * h_vd);
            } else {
                h_vd = (uint32_t) std::ceil(groups_m / std::round(2 / ratio));
                h_uvd_recip = reciprocal(groups_n * h_vd);
                h_vd |= 0xFFFF0000u;
            }
            h_bail = (k <= 64) ? 512 : 1;
        } else if (info.isBoustrophedon()) {
            const double bias = double(info.wg[0] * info.unroll[0]) / double(info.wg[1] * info.unroll[1]);
            const double sm = std::sqrt(concurrent_tg / bias), sn = std::sqrt(concurrent_tg * bias);
            int32_t slice = 0, thresh = 0;
            bool ok = false;
            for (bool nslice : {groups_m > groups_n, groups_m <= groups_n}) {
                double sd = nslice ? sn : sm;
                int sf = int(std::floor(sd)), sc = int(std::ceil(sd));
                if (concurrent_tg % (uint32_t) sc == 0) sf = sc;
                if (concurrent_tg % (uint32_t) (sc + 1) == 0) sf = sc = sc + 1;
                const int gcn = nslice ? (int) groups_n : (int) groups_m;
                const int gco = nslice ? (int) groups_m : (int) groups_n;
                for (int srange = 0; srange <= 2 && !ok; srange++) {
                    const int s0 = (srange < 2) ? sc : sf;
                    const bool up = (srange == 1);
                    const int s1 = s0 + (up ? 1 : -1);
                    if (s1 <= 0) continue;
                    const int rem = gcn % s0;
                    if (!rem || up) thresh = gcn / s0 - rem;
                    else thresh = (gcn + s0 - 1) / s0 - (s0 - rem);
                    ok = (thresh >= 0) && (gco >= 2 * std::max(s0, s1));
                    slice = s0;
                    if (!up) {
                        if (thresh > 0) thresh = -thresh;
                        else {
                            slice--;
                            thresh = gcn;
                        }
                    }
                    if (nslice) slice *= -1;
                }
                if (ok) break;
            }
            if (!ok) {
                const bool nslice = (groups_m > groups_n);
                const double sd = nslice ? sn : sm;
                const int gcn = nslice ? (int) groups_n : (int) groups_m;
                if (gcn < sd * 1.5) slice = gcn;
                else slice = gcn / ((gcn + int(std::round(sd)) - 1) / int(std::round(sd)));
                thresh = std::max(0, (gcn / slice) - (gcn % slice));
                if (nslice) slice *= -1;
            }
            if (slice == 0) {
                slice = 1;
                thresh = (int32_t) groups_m;
            }
            bslice = slice;
            bthresh = thresh;
        }
        if (info.isPersistent()) {
            group_count = std::min(group_count, concurrent_tg);
            group_stride = group_count;
        }
        gws[0] = lws[0] * (group_count + (uint32_t) info.extraWGs());
        gws[1] = lws[1];
    }
    size_t slm = (size_t) info.slm;
    if (info.perKSLM > 0 && lws[2] > 1) slm = std::max(slm, (size_t) info.perKSLM * lws[2]);

    lws[0] *= (uint64_t) info.subgroupSize;
    gws[0] *= (uint64_t) info.subgroupSize;

    std::vector<void*> args;
    args.reserve(b->nargs);
    args.push_back(&a_ptr);
    args.push_back(&b_ptr);
    args.push_back(&c_ptr);
    args.push_back(&off_a);
    args.push_back(&off_b);
    args.push_back(&off_c);
    args.push_back(&lda);
    args.push_back(&ldb);
    args.push_back(&ldc);
    int32_t mm = m, nn = n, kk = k;
    args.push_back(&mm);
    args.push_back(&nn);
    args.push_back(&kk);
    args.push_back(&alpha);
    args.push_back(&beta);
    if (problem.aScale2D()) args.push_back(&s_ptr);
    if (problem.aOffset2D() || problem.aScale2D()) args.push_back(&ldaq);
    args.push_back(&flags);
    if ((strategy.kParallel || strategy.kParallelLocal) && !strategy.kParallelVariable) args.push_back(&k0);
    if (strategy.linearOrder()) {
        args.push_back(&groups_m);
        args.push_back(&groups_n);
    }
    if (strategy.cWalkOrder == gm::WalkOrder::SimpleLinear) args.push_back(&gc_recip);
    else if (strategy.cWalkOrder == gm::WalkOrder::Hilbertlike) {
        args.push_back(&h_vd);
        args.push_back(&h_uvd_recip);
        args.push_back(&h_bail);
    } else if (strategy.cWalkOrder == gm::WalkOrder::Boustrophedon) {
        args.push_back(&bslice);
        args.push_back(&bthresh);
    }
    if (strategy.persistent) args.push_back(&group_stride);
    if (strategy.variableSLM()) args.push_back(&slm);
    if (problem.aoPtrDims >= 1 || problem.aScale2D()) args.push_back(&off_aq);
    if (args.size() != b->nargs) {
        std::fprintf(stderr, "strata gs: %s: %zu arguments built for %u declared\n", b->kernel.name(), args.size(), b->nargs);
        std::abort();
    }
    const ze::Dim3 block{(uint32_t) lws[0], (uint32_t) lws[1], (uint32_t) lws[2]};
    const ze::Dim3 grid{(uint32_t) (gws[0] / lws[0]), (uint32_t) (gws[1] / lws[1]), (uint32_t) (gws[2] / lws[2])};
    s->launch(b->kernel, grid, block, args.data());
    return true;
}

}  // namespace strata::gs
