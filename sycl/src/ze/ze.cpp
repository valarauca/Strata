// src/ze/ze.cpp - see include/strata/ze/ze.hpp.
#include "strata/ze/ze.hpp"
#include "strata/ze/modules.hpp"

#include <level_zero/zes_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

namespace strata::ze {

namespace {

bool env_on(const char* name, bool def) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == 0) return def;
    return !(v[0] == '0' && v[1] == 0);
}

struct Counters {
    std::atomic<uint64_t> launches{0}, copies{0}, graph_launches{0}, host_waits{0}, events{0}, captures{0};
    std::atomic<uint64_t> dropped_waits{0}, frees{0}, pageable{0};
};
Counters g_count;

struct Alloc {
    size_t bytes = 0;
    MemKind kind = MemKind::Unknown;
    int dev = -1;
};

struct DeviceState {
    ze_device_handle_t h = nullptr;
    zes_device_handle_t sysman = nullptr;
    DeviceInfo info;
    std::mutex mod_mu;
    std::vector<ze_module_handle_t> modules;
    std::unique_ptr<Stream> def;
    std::unique_ptr<Stream> copy;
    std::once_flag def_once, copy_once;
    std::mutex streams_mu;
    std::vector<Stream*> streams;
    std::atomic<uint64_t> allocated{0};
};

struct EventPool {
    ze_event_pool_handle_t h = nullptr;
    uint32_t used = 0;
};

class Runtime {
public:
    static Runtime& get() {
        static Runtime* rt = new Runtime();
        return *rt;
    }

    ze_driver_handle_t driver = nullptr;
    ze_context_handle_t ctx = nullptr;
    std::vector<std::unique_ptr<DeviceState>> devs;
    bool serialize = false;
    bool report_on = false;

    std::mutex alloc_mu;
    std::map<uintptr_t, Alloc> allocs;

    std::mutex ev_mu;
    std::vector<EventPool> pools;
    std::vector<std::pair<void*, uint32_t>> free_events;

    std::mutex names_mu;
    bool names_ready = false;
    std::unordered_map<std::string, int> kernel_module;   // kernel name -> index into the bundle module list

    uint64_t* stamps = nullptr;   // timing events' timestamp slots (host memory)
    std::vector<uint32_t> free_stamps;

private:
    Runtime() { init(); }
    void init();
};

const char* kDeviceLost = "DEVICE_LOST";

void Runtime::init() {
    serialize = env_on("STRATA_L0_SERIALIZE", false);
    report_on = env_on("STRATA_ZE_REPORT", false);
    STRATA_ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t nd = 0;
    STRATA_ZE_CHECK(zeDriverGet(&nd, nullptr));
    std::vector<ze_driver_handle_t> drivers(nd);
    STRATA_ZE_CHECK(zeDriverGet(&nd, drivers.data()));
    std::vector<ze_device_handle_t> all;
    for (ze_driver_handle_t d : drivers) {
        uint32_t n = 0;
        if (zeDeviceGet(d, &n, nullptr) != ZE_RESULT_SUCCESS || n == 0) continue;
        all.resize(n);
        STRATA_ZE_CHECK(zeDeviceGet(d, &n, all.data()));
        driver = d;
        break;
    }
    if (driver == nullptr || all.empty()) {
        std::fprintf(stderr, "strata ze: no Level Zero GPU found\n");
        std::exit(1);
    }
    auto pci_of = [](ze_device_handle_t h) {
        ze_pci_ext_properties_t p{ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
        char buf[32] = "unknown";
        if (zeDevicePciGetPropertiesExt(h, &p) == ZE_RESULT_SUCCESS)
            std::snprintf(buf, sizeof buf, "%04x:%02x:%02x.%x", p.address.domain, p.address.bus, p.address.device,
                          p.address.function);
        return std::string(buf);
    };
    std::vector<int> pick;
    std::string spec;
    if (const char* s = std::getenv("STRATA_ZE_DEVICES"); s && *s) spec = s;
    else if (const char* o = std::getenv("ONEAPI_DEVICE_SELECTOR"); o && *o) {
        const std::string v = o;
        const size_t c = v.find(':');
        if (v.rfind("level_zero", 0) != 0 || c == std::string::npos) {
            std::fprintf(stderr, "strata ze: ONEAPI_DEVICE_SELECTOR='%s' is not a level_zero:<list> selector\n", o);
            std::exit(1);
        }
        spec = v.substr(c + 1);
        std::replace(spec.begin(), spec.end(), ';', ',');
    }
    if (spec.empty() || spec == "*") {
        for (size_t i = 0; i < all.size(); ++i) pick.push_back((int) i);
    } else {
        size_t a = 0;
        while (a <= spec.size()) {
            size_t b = spec.find(',', a);
            if (b == std::string::npos) b = spec.size();
            const std::string item = spec.substr(a, b - a);
            a = b + 1;
            if (item.empty()) continue;
            int found = -1;
            if (item.find(':') != std::string::npos) {
                for (size_t i = 0; i < all.size(); ++i)
                    if (pci_of(all[i]) == item) found = (int) i;
            } else {
                char* end = nullptr;
                const long k = std::strtol(item.c_str(), &end, 10);
                if (end && *end == 0 && k >= 0 && k < (long) all.size()) found = (int) k;
            }
            if (found < 0) {
                std::fprintf(stderr, "strata ze: GPU '%s' (from '%s') is not one of the %zu Level Zero GPUs\n", item.c_str(),
                             spec.c_str(), all.size());
                std::exit(1);
            }
            pick.push_back(found);
        }
    }
    if ((int) pick.size() > kMaxDevices) pick.resize(kMaxDevices);
    std::vector<ze_device_handle_t> used;
    for (int i : pick) used.push_back(all[(size_t) i]);
    ze_context_desc_t cd{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    STRATA_ZE_CHECK(zeContextCreateEx(driver, &cd, (uint32_t) used.size(), used.data(), &ctx));

    const bool sysman_ok = zesInit(0) == ZE_RESULT_SUCCESS;
    std::vector<std::pair<std::string, zes_device_handle_t>> zes;
    if (sysman_ok) {
        uint32_t zn = 0;
        if (zesDriverGet(&zn, nullptr) == ZE_RESULT_SUCCESS && zn > 0) {
            std::vector<zes_driver_handle_t> zd(zn);
            zesDriverGet(&zn, zd.data());
            for (zes_driver_handle_t d : zd) {
                uint32_t n = 0;
                if (zesDeviceGet(d, &n, nullptr) != ZE_RESULT_SUCCESS) continue;
                std::vector<zes_device_handle_t> h(n);
                zesDeviceGet(d, &n, h.data());
                for (zes_device_handle_t x : h) {
                    zes_pci_properties_t pp{ZES_STRUCTURE_TYPE_PCI_PROPERTIES};
                    if (zesDevicePciGetProperties(x, &pp) != ZE_RESULT_SUCCESS) continue;
                    char buf[32];
                    std::snprintf(buf, sizeof buf, "%04x:%02x:%02x.%x", pp.address.domain, pp.address.bus, pp.address.device,
                                  pp.address.function);
                    zes.emplace_back(buf, x);
                }
            }
        }
    }

    for (ze_device_handle_t h : used) {
        auto st = std::make_unique<DeviceState>();
        st->h = h;
        DeviceInfo& di = st->info;
        ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2};
        STRATA_ZE_CHECK(zeDeviceGetProperties(h, &p));
        di.name = p.name;
        di.device_id = p.deviceId;
        di.max_alloc = p.maxMemAllocSize;
        di.eu_count = p.numSlices * p.numSubslicesPerSlice * p.numEUsPerSubslice;
        di.timer_ns_num = 1000000000ull;
        di.timer_ns_den = p.timerResolution ? p.timerResolution : 1;
        di.timestamp_mask = p.timestampValidBits >= 64 ? ~0ull : ((1ull << p.timestampValidBits) - 1);
        di.pci = pci_of(h);
        di.battlemage = (p.deviceId & 0xff00) == 0xe200;
        ze_device_compute_properties_t cp{ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES};
        STRATA_ZE_CHECK(zeDeviceGetComputeProperties(h, &cp));
        di.max_group_size = cp.maxTotalGroupSize;
        di.max_slm = cp.maxSharedLocalMemory;
        uint32_t nm = 0;
        STRATA_ZE_CHECK(zeDeviceGetMemoryProperties(h, &nm, nullptr));
        std::vector<ze_device_memory_properties_t> mp(nm, {ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES});
        STRATA_ZE_CHECK(zeDeviceGetMemoryProperties(h, &nm, mp.data()));
        for (auto& m : mp) di.total_mem += m.totalSize;
        uint32_t ng = 0;
        STRATA_ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(h, &ng, nullptr));
        std::vector<ze_command_queue_group_properties_t> g(ng, {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES});
        STRATA_ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(h, &ng, g.data()));
        di.compute_ordinal = -1;
        for (uint32_t i = 0; i < ng; ++i) {
            if ((g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) && di.compute_ordinal < 0) di.compute_ordinal = (int) i;
            if ((g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) && !(g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) &&
                di.copy_ordinal < 0)
                di.copy_ordinal = (int) i;
        }
        if (di.compute_ordinal < 0) {
            std::fprintf(stderr, "strata ze: GPU %s has no compute queue group\n", di.pci.c_str());
            std::exit(1);
        }
        for (auto& z : zes)
            if (z.first == di.pci) st->sysman = z.second;
        devs.push_back(std::move(st));
    }
    std::string line;
    for (size_t i = 0; i < devs.size(); ++i) {
        const DeviceInfo& di = devs[i]->info;
        char buf[256];
        std::snprintf(buf, sizeof buf, "%s%zu: %s (%s, %.1f GiB, queue groups compute %d copy %d)", i ? "; " : "", i,
                      di.name.c_str(), di.pci.c_str(), (double) di.total_mem / 1073741824.0, di.compute_ordinal, di.copy_ordinal);
        line += buf;
    }
    std::fprintf(stderr, "strata ze: Level Zero, one context over %zu GPU%s: %s\n", devs.size(), devs.size() == 1 ? "" : "s",
                 line.c_str());
    if (serialize) std::fprintf(stderr, "strata ze: STRATA_L0_SERIALIZE=1: a host wait after every command (debug)\n");
}

Runtime& rt() { return Runtime::get(); }

thread_local int t_dev = 0;

DeviceState& dev_state(int ordinal) {
    Runtime& r = rt();
    if (ordinal < 0) ordinal = t_dev;
    if (ordinal < 0 || ordinal >= (int) r.devs.size()) {
        std::fprintf(stderr, "strata ze: GPU ordinal %d out of range (%zu GPUs)\n", ordinal, r.devs.size());
        std::exit(1);
    }
    return *r.devs[(size_t) ordinal];
}

void register_alloc(void* p, size_t bytes, MemKind kind, int dev) {
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.alloc_mu);
    r.allocs[(uintptr_t) p] = Alloc{bytes, kind, dev};
}

void ensure_names() {
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.names_mu);
    if (r.names_ready) return;
    const auto& mods = bundle_modules();
    for (size_t m = 0; m < mods.size(); ++m)
        for (const std::string& n : mods[m].kernels) r.kernel_module.emplace(n, (int) m);
    r.names_ready = true;
}

ze_module_handle_t module_on(int ordinal, int m) {
    DeviceState& st = dev_state(ordinal);
    std::lock_guard<std::mutex> lk(st.mod_mu);
    const auto& mods = bundle_modules();
    if (st.modules.empty()) st.modules.assign(mods.size(), nullptr);
    if (st.modules[(size_t) m] == nullptr) {
        ze_module_desc_t md{ZE_STRUCTURE_TYPE_MODULE_DESC};
        md.format = ZE_MODULE_FORMAT_NATIVE;
        md.inputSize = mods[(size_t) m].size;
        md.pInputModule = mods[(size_t) m].data;
        ze_module_build_log_handle_t log = nullptr;
        const ze_result_t r = zeModuleCreate(rt().ctx, st.h, &md, &st.modules[(size_t) m], &log);
        if (r != ZE_RESULT_SUCCESS) {
            size_t n = 0;
            std::string text;
            if (log && zeModuleBuildLogGetString(log, &n, nullptr) == ZE_RESULT_SUCCESS && n > 0) {
                text.resize(n);
                zeModuleBuildLogGetString(log, &n, text.data());
            }
            std::fprintf(stderr, "strata ze: module %s on GPU %d: %s\n%s\n", mods[(size_t) m].name.c_str(), ordinal,
                         result_name(r), text.c_str());
            std::exit(1);
        }
        if (log) zeModuleBuildLogDestroy(log);
    }
    return st.modules[(size_t) m];
}

void* new_event_slot(ze_event_handle_t* ev, uint32_t* index) {
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.ev_mu);
    if (!r.free_events.empty()) {
        auto [pool, idx] = r.free_events.back();
        r.free_events.pop_back();
        ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC};
        ed.index = idx;
        ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        ed.wait = ZE_EVENT_SCOPE_FLAG_HOST;
        STRATA_ZE_CHECK(zeEventCreate((ze_event_pool_handle_t) pool, &ed, ev));
        *index = idx;
        return pool;
    }
    constexpr uint32_t kPool = 256;
    if (r.pools.empty() || r.pools.back().used == kPool) {
        ze_event_pool_counter_based_exp_desc_t cb{ZE_STRUCTURE_TYPE_COUNTER_BASED_EVENT_POOL_EXP_DESC};
        cb.flags = ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_IMMEDIATE | ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_NON_IMMEDIATE;
        ze_event_pool_desc_t pd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
        pd.pNext = &cb;
        pd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
        pd.count = kPool;
        std::vector<ze_device_handle_t> hs;
        for (auto& d : r.devs) hs.push_back(d->h);
        EventPool ep;
        STRATA_ZE_CHECK(zeEventPoolCreate(r.ctx, &pd, (uint32_t) hs.size(), hs.data(), &ep.h));
        r.pools.push_back(ep);
    }
    EventPool& ep = r.pools.back();
    ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC};
    ed.index = ep.used;
    ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    ed.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    STRATA_ZE_CHECK(zeEventCreate(ep.h, &ed, ev));
    *index = ep.used++;
    g_count.events.fetch_add(1, std::memory_order_relaxed);
    return (void*) ep.h;
}

uint64_t* new_stamp() {
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.ev_mu);
    if (r.stamps == nullptr) {
        constexpr uint32_t kStamps = 65536;
        ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
        STRATA_ZE_CHECK(zeMemAllocHost(r.ctx, &hd, kStamps * sizeof(uint64_t), 64, (void**) &r.stamps));
        std::memset(r.stamps, 0, kStamps * sizeof(uint64_t));
        for (uint32_t i = kStamps; i > 0; --i) r.free_stamps.push_back(i - 1);
    }
    if (r.free_stamps.empty()) {
        std::fprintf(stderr, "strata ze: out of timing event slots\n");
        std::exit(1);
    }
    const uint32_t i = r.free_stamps.back();
    r.free_stamps.pop_back();
    return r.stamps + i;
}

}  // namespace

[[noreturn]] void fatal(const char* what, ze_result_t r, const char* file, int line) {
    std::fprintf(stderr, "strata ze: %s failed: %s (%s:%d)%s\n", what, result_name(r), file, line,
                 r == ZE_RESULT_ERROR_DEVICE_LOST ? " - DEVICE_LOST" : "");
    std::fflush(stderr);
    std::_Exit(1);
}

const char* result_name(ze_result_t r) {
    switch (r) {
        case ZE_RESULT_SUCCESS: return "ZE_RESULT_SUCCESS";
        case ZE_RESULT_NOT_READY: return "ZE_RESULT_NOT_READY";
        case ZE_RESULT_ERROR_DEVICE_LOST: return "ZE_RESULT_ERROR_DEVICE_LOST";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
        case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE: return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
        case ZE_RESULT_ERROR_MODULE_LINK_FAILURE: return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
        case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET: return "ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET";
        case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE: return "ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE";
        case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS: return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
        case ZE_RESULT_ERROR_NOT_AVAILABLE: return "ZE_RESULT_ERROR_NOT_AVAILABLE";
        case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE: return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
        case ZE_RESULT_ERROR_UNINITIALIZED: return "ZE_RESULT_ERROR_UNINITIALIZED";
        case ZE_RESULT_ERROR_UNSUPPORTED_VERSION: return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE: return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
        case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE: return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER: return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
        case ZE_RESULT_ERROR_INVALID_SIZE: return "ZE_RESULT_ERROR_INVALID_SIZE";
        case ZE_RESULT_ERROR_UNSUPPORTED_SIZE: return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
        case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT: return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
        case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT: return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
        case ZE_RESULT_ERROR_INVALID_ENUMERATION: return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
        case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION: return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
        case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT: return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
        case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY: return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
        case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME: return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
        case ZE_RESULT_ERROR_INVALID_KERNEL_NAME: return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
        case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME: return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
        case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION: return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
        case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION: return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
        case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX: return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
        case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE: return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
        case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE: return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
        case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED: return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
        case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE: return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
        case ZE_RESULT_ERROR_OVERLAPPING_REGIONS: return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
        default: break;
    }
    static thread_local char buf[32];
    std::snprintf(buf, sizeof buf, "ze_result 0x%x", (unsigned) r);
    return buf;
}

int device_count() { return (int) rt().devs.size(); }
int current_device() { return t_dev; }
void set_device(int ordinal) {
    if (ordinal < 0 || ordinal >= device_count()) {
        std::fprintf(stderr, "strata ze: set_device(%d): %d GPUs\n", ordinal, device_count());
        std::exit(1);
    }
    t_dev = ordinal;
}
const DeviceInfo& device_info(int ordinal) { return dev_state(ordinal).info; }
ze_device_handle_t device_handle(int ordinal) { return dev_state(ordinal).h; }
ze_context_handle_t context() { return rt().ctx; }

bool mem_get_info(int ordinal, size_t* free_bytes, size_t* total_bytes) {
    DeviceState& st = dev_state(ordinal);
    if (st.sysman) {
        uint32_t n = 0;
        if (zesDeviceEnumMemoryModules(st.sysman, &n, nullptr) == ZE_RESULT_SUCCESS && n > 0) {
            std::vector<zes_mem_handle_t> mh(n);
            zesDeviceEnumMemoryModules(st.sysman, &n, mh.data());
            uint64_t fr = 0, tot = 0;
            bool any = false;
            for (zes_mem_handle_t m : mh) {
                zes_mem_properties_t mp{ZES_STRUCTURE_TYPE_MEM_PROPERTIES};
                if (zesMemoryGetProperties(m, &mp) != ZE_RESULT_SUCCESS || mp.location != ZES_MEM_LOC_DEVICE) continue;
                zes_mem_state_t ms{ZES_STRUCTURE_TYPE_MEM_STATE};
                if (zesMemoryGetState(m, &ms) != ZE_RESULT_SUCCESS) continue;
                fr += ms.free;
                tot += ms.size;
                any = true;
            }
            if (any) {
                if (free_bytes) *free_bytes = (size_t) fr;
                if (total_bytes) *total_bytes = (size_t) tot;
                return true;
            }
        }
    }
    const uint64_t tot = st.info.total_mem, used = st.allocated.load();
    if (free_bytes) *free_bytes = (size_t) (tot > used ? tot - used : 0);
    if (total_bytes) *total_bytes = (size_t) tot;
    return tot > 0;
}

bool can_access_peer(int a, int b) {
    ze_bool_t can = 0;
    if (zeDeviceCanAccessPeer(dev_state(a).h, dev_state(b).h, &can) != ZE_RESULT_SUCCESS) return false;
    return can != 0;
}

void* malloc_device(size_t bytes, int ordinal) {
    if (bytes == 0) return nullptr;
    if (ordinal < 0) ordinal = t_dev;
    DeviceState& st = dev_state(ordinal);
    ze_relaxed_allocation_limits_exp_desc_t rl{ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC};
    rl.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
    ze_device_mem_alloc_desc_t dd{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    if (bytes > st.info.max_alloc) dd.pNext = &rl;
    void* p = nullptr;
    const ze_result_t r = zeMemAllocDevice(rt().ctx, &dd, bytes, 64, st.h, &p);
    if (r == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY || r == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY) return nullptr;
    if (r != ZE_RESULT_SUCCESS) fatal("zeMemAllocDevice", r, __FILE__, __LINE__);
    st.allocated.fetch_add(bytes);
    register_alloc(p, bytes, MemKind::Device, ordinal);
    return p;
}

void* malloc_host(size_t bytes) {
    if (bytes == 0) return nullptr;
    ze_relaxed_allocation_limits_exp_desc_t rl{ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC};
    rl.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
    ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    if (bytes > ((size_t) 4 << 30)) hd.pNext = &rl;
    void* p = nullptr;
    const ze_result_t r = zeMemAllocHost(rt().ctx, &hd, bytes, 64, &p);
    if (r == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY || r == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) return nullptr;
    if (r != ZE_RESULT_SUCCESS) fatal("zeMemAllocHost", r, __FILE__, __LINE__);
    register_alloc(p, bytes, MemKind::Host, -1);
    return p;
}

void* malloc_host_uncached(size_t bytes) {
    if (bytes == 0) return nullptr;
    ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    hd.flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED;
    void* p = nullptr;
    const ze_result_t r = zeMemAllocHost(rt().ctx, &hd, bytes, 64, &p);
    if (r == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY) return nullptr;
    if (r != ZE_RESULT_SUCCESS) fatal("zeMemAllocHost(uncached)", r, __FILE__, __LINE__);
    register_alloc(p, bytes, MemKind::Host, -1);
    return p;
}

void free(void* p) {
    if (p == nullptr) return;
    Runtime& r = rt();
    Alloc a;
    {
        std::lock_guard<std::mutex> lk(r.alloc_mu);
        auto it = r.allocs.find((uintptr_t) p);
        if (it == r.allocs.end()) {
            std::fprintf(stderr, "strata ze: free(%p): not a Strata allocation\n", p);
            std::fflush(stderr);
            std::abort();
        }
        a = it->second;
    }
    if (a.kind == MemKind::Device) device_sync(a.dev);
    else sync_all();
    {
        std::lock_guard<std::mutex> lk(r.alloc_mu);
        r.allocs.erase((uintptr_t) p);
    }
    if (a.kind == MemKind::Device) dev_state(a.dev).allocated.fetch_sub(a.bytes);
    STRATA_ZE_CHECK(zeMemFree(r.ctx, p));
    g_count.frees.fetch_add(1, std::memory_order_relaxed);
}

MemKind kind_of(const void* p, int* dev) {
    if (dev) *dev = -1;
    if (p == nullptr) return MemKind::Unknown;
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.alloc_mu);
    auto it = r.allocs.upper_bound((uintptr_t) p);
    if (it == r.allocs.begin()) return MemKind::Unknown;
    --it;
    if ((uintptr_t) p >= it->first + it->second.bytes) return MemKind::Unknown;
    if (dev) *dev = it->second.dev;
    return it->second.kind;
}

size_t device_offset_end(const void* p, size_t bytes) {
    if (p == nullptr) return 0;
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.alloc_mu);
    auto it = r.allocs.upper_bound((uintptr_t) p);
    if (it == r.allocs.begin()) return 0;
    --it;
    if (it->second.kind != MemKind::Device) return 0;
    const uintptr_t end = (uintptr_t) p + bytes;
    if (end > it->first + it->second.bytes) return 0;
    return (size_t) (end - it->first);
}

MemTotals mem_totals() {
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.alloc_mu);
    MemTotals t;
    for (auto& [k, a] : r.allocs) {
        if (a.kind == MemKind::Device) { ++t.device_n; t.device_bytes += a.bytes; }
        else { ++t.host_n; t.host_bytes += a.bytes; }
    }
    return t;
}

ze_kernel_handle_t KernelBase::handle(int ordinal) const {
    ze_kernel_handle_t h = h_[ordinal].load(std::memory_order_acquire);
    return h ? h : create(ordinal);
}

ze_kernel_handle_t KernelBase::create(int ordinal) const {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    ze_kernel_handle_t h = h_[ordinal].load(std::memory_order_acquire);
    if (h) return h;
    ensure_names();
    Runtime& r = rt();
    auto it = r.kernel_module.find(name_);
    if (it == r.kernel_module.end()) {
        std::fprintf(stderr, "strata ze: kernel %s is not in the engine's device modules (a build without it?)\n", name_);
        std::exit(1);
    }
    ze_module_handle_t m = module_on(ordinal, it->second);
    ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC};
    kd.pKernelName = name_;
    STRATA_ZE_CHECK(zeKernelCreate(m, &kd, &h));
    ze_kernel_properties_t kp{ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES};
    STRATA_ZE_CHECK(zeKernelGetProperties(h, &kp));
    if (kp.numKernelArgs != nparams_ && kp.numKernelArgs != nparams_ + 1) {
        std::fprintf(stderr, "strata ze: kernel %s takes %u arguments, the host declares %u\n", name_, kp.numKernelArgs,
                     nparams_);
        std::exit(1);
    }
    if (kp.spillMemSize > 0 && env_on("STRATA_ZE_SPILL_REPORT", false))
        std::fprintf(stderr, "strata ze: kernel %s spills %u bytes\n", name_, kp.spillMemSize);
    h_[ordinal].store(h, std::memory_order_release);
    return h;
}

void check_geometry(const KernelBase& k, Dim3 grid, Dim3 block) {
    const uint64_t g = (uint64_t) block.x * block.y * block.z;
    if (g == 0 || g > 1024) {
        std::fprintf(stderr, "strata ze: kernel %s launched with grid %ux%ux%u block %ux%ux%u\n", k.name(), grid.x, grid.y,
                     grid.z, block.x, block.y, block.z);
        std::fflush(stderr);
        std::abort();
    }
}

Stream::Stream(int ordinal, Engine engine, bool high_priority) : dev_(ordinal < 0 ? t_dev : ordinal), engine_(engine) {
    DeviceState& st = dev_state(dev_);
    ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    qd.ordinal = (uint32_t) (engine == Engine::Copy && st.info.copy_ordinal >= 0 ? st.info.copy_ordinal : st.info.compute_ordinal);
    if (engine == Engine::Copy && st.info.copy_ordinal < 0) engine_ = Engine::Compute;
    qd.index = 0;
    qd.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qd.priority = high_priority ? ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH : ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    STRATA_ZE_CHECK(zeCommandListCreateImmediate(rt().ctx, st.h, &qd, &imm_));
    std::lock_guard<std::mutex> lk(st.streams_mu);
    st.streams.push_back(this);
}

Stream::~Stream() {
    if (runtime_owned_) {
        std::fprintf(stderr, "strata ze: a runtime-owned stream (default or copy stream of GPU %d) was deleted\n", dev_);
        std::fflush(stderr);
        std::abort();
    }
    {
        DeviceState& st = dev_state(dev_);
        std::lock_guard<std::mutex> lk(st.streams_mu);
        st.streams.erase(std::remove(st.streams.begin(), st.streams.end(), this), st.streams.end());
    }
    if (imm_) {
        zeCommandListHostSynchronize(imm_, UINT64_MAX);
        zeCommandListDestroy(imm_);
    }
    if (bounce_) free(bounce_);
}

void Stream::after_append() {
    appends_.fetch_add(1, std::memory_order_relaxed);
    if (cap_) {
        ++cap_->nodes;
        return;
    }
    if (rt().serialize) {
        const ze_result_t r = zeCommandListHostSynchronize(imm_, UINT64_MAX);
        if (r != ZE_RESULT_SUCCESS) fatal("zeCommandListHostSynchronize (STRATA_L0_SERIALIZE)", r, __FILE__, __LINE__);
    }
}

void Stream::launch(const KernelBase& k, Dim3 grid, Dim3 block, void** args) {
    check_geometry(k, grid, block);
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) return;
    std::lock_guard<std::recursive_mutex> lk(mu_);
    const int d = cap_ ? cap_->dev : dev_;
    ze_kernel_handle_t h = k.handle(d);
    const ze_group_count_t gc{grid.x, grid.y, grid.z};
    const ze_group_size_t gs{block.x, block.y, block.z};
    const ze_result_t r = zeCommandListAppendLaunchKernelWithArguments(target(), h, gc, gs, args, nullptr, nullptr, 0, nullptr);
    if (r != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr, "strata ze: launch of %s (grid %ux%ux%u block %ux%ux%u) failed: %s%s\n", k.name(), grid.x, grid.y,
                     grid.z, block.x, block.y, block.z, result_name(r), r == ZE_RESULT_ERROR_DEVICE_LOST ? " - DEVICE_LOST" : "");
        std::fflush(stderr);
        std::_Exit(1);
    }
    g_count.launches.fetch_add(1, std::memory_order_relaxed);
    after_append();
}

void Stream::memcpy(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    std::lock_guard<std::recursive_mutex> lk(mu_);
    const bool src_ok = kind_of(src) != MemKind::Unknown, dst_ok = kind_of(dst) != MemKind::Unknown;
    if (src_ok && dst_ok) {
        STRATA_ZE_CHECK(zeCommandListAppendMemoryCopy(target(), dst, src, bytes, nullptr, 0, nullptr));
        g_count.copies.fetch_add(1, std::memory_order_relaxed);
        after_append();
        return;
    }
    if (cap_) {
        std::fprintf(stderr, "strata ze: a copy from or to pageable host memory inside a capture (%p -> %p)\n", src, dst);
        std::exit(1);
    }
    g_count.pageable.fetch_add(1, std::memory_order_relaxed);
    sync();
    if (!src_ok && !dst_ok) {
        std::memcpy(dst, src, bytes);
        return;
    }
    constexpr size_t kChunk = (size_t) 8 << 20;
    if (bounce_ == nullptr) bounce_ = malloc_host(kChunk);
    for (size_t off = 0; off < bytes; off += kChunk) {
        const size_t n = bytes - off < kChunk ? bytes - off : kChunk;
        if (!src_ok) {
            std::memcpy(bounce_, (const uint8_t*) src + off, n);
            STRATA_ZE_CHECK(zeCommandListAppendMemoryCopy(imm_, (uint8_t*) dst + off, bounce_, n, nullptr, 0, nullptr));
            sync();
        } else {
            STRATA_ZE_CHECK(zeCommandListAppendMemoryCopy(imm_, bounce_, (const uint8_t*) src + off, n, nullptr, 0, nullptr));
            sync();
            std::memcpy((uint8_t*) dst + off, bounce_, n);
        }
        g_count.copies.fetch_add(1, std::memory_order_relaxed);
    }
}

void Stream::memcpy_2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
    if (width == 0 || height == 0) return;
    if (dpitch == width && spitch == width) {
        memcpy(dst, src, width * height);
        return;
    }
    std::lock_guard<std::recursive_mutex> lk(mu_);
    if (kind_of(dst) == MemKind::Unknown || kind_of(src) == MemKind::Unknown) {
        for (size_t r = 0; r < height; ++r) memcpy((uint8_t*) dst + r * dpitch, (const uint8_t*) src + r * spitch, width);
        return;
    }
    const ze_copy_region_t dr{0, 0, 0, (uint32_t) width, (uint32_t) height, 1};
    const ze_copy_region_t sr{0, 0, 0, (uint32_t) width, (uint32_t) height, 1};
    STRATA_ZE_CHECK(zeCommandListAppendMemoryCopyRegion(target(), dst, &dr, (uint32_t) dpitch, 0, src, &sr, (uint32_t) spitch, 0,
                                                       nullptr, 0, nullptr));
    g_count.copies.fetch_add(1, std::memory_order_relaxed);
    after_append();
}

void Stream::memset(void* dst, int value, size_t bytes) {
    if (bytes == 0) return;
    const uint8_t v = (uint8_t) value;
    fill(dst, &v, 1, bytes);
}

void Stream::fill(void* dst, const void* pattern, size_t pattern_bytes, size_t bytes) {
    if (bytes == 0) return;
    std::lock_guard<std::recursive_mutex> lk(mu_);
    if (kind_of(dst) == MemKind::Unknown) {
        if (cap_) {
            std::fprintf(stderr, "strata ze: a fill of pageable host memory inside a capture (%p)\n", dst);
            std::exit(1);
        }
        sync();
        for (size_t off = 0; off < bytes; off += pattern_bytes)
            std::memcpy((uint8_t*) dst + off, pattern, bytes - off < pattern_bytes ? bytes - off : pattern_bytes);
        return;
    }
    STRATA_ZE_CHECK(zeCommandListAppendMemoryFill(target(), dst, pattern, pattern_bytes, bytes, nullptr, 0, nullptr));
    g_count.copies.fetch_add(1, std::memory_order_relaxed);
    after_append();
}

void Stream::write_timestamp(uint64_t* dst) {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    STRATA_ZE_CHECK(zeCommandListAppendWriteGlobalTimestamp(target(), dst, nullptr, 0, nullptr));
    after_append();
}

void Stream::record(Event& e) {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    if (cap_) {
        if (e.timing_) {
            STRATA_ZE_CHECK(zeCommandListAppendWriteGlobalTimestamp(cap_->list_, e.stamp_, nullptr, 0, nullptr));
            e.stamp_dev_ = cap_->dev;
            ++cap_->nodes;
        }
        e.captured_in_ = cap_;
        e.recorded_ = true;
        return;
    }
    e.captured_in_ = nullptr;
    if (e.timing_) {
        STRATA_ZE_CHECK(zeCommandListAppendWriteGlobalTimestamp(imm_, e.stamp_, nullptr, 0, nullptr));
        e.stamp_dev_ = dev_;
    }
    STRATA_ZE_CHECK(zeCommandListAppendBarrier(imm_, e.ev_, 0, nullptr));
    e.recorded_ = true;
    after_append();
}

void Stream::wait(Event& e) {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    if (!e.recorded_) return;
    if (e.captured_in_ != nullptr) {
        Capture* c = e.captured_in_;
        if (cap_ == c) return;
        if (cap_ == nullptr && c->list_ != nullptr) {
            if (c->dev != dev_) {
                std::fprintf(stderr, "strata ze: a stream of GPU %d joined a capture of GPU %d\n", dev_, c->dev);
                std::exit(1);
            }
            cap_ = c;
            c->members.push_back(this);
            return;
        }
        if (c->list_ == nullptr) {
            std::fprintf(stderr, "strata ze: wait on an event recorded inside a finished capture\n");
            std::exit(1);
        }
        std::fprintf(stderr, "strata ze: a stream in one capture waited on an event of another capture\n");
        std::exit(1);
    }
    if (cap_) {
        g_count.dropped_waits.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    STRATA_ZE_CHECK(zeCommandListAppendWaitOnEvents(imm_, 1, &e.ev_));
    after_append();
}

void Stream::sync() {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    g_count.host_waits.fetch_add(1, std::memory_order_relaxed);
    const ze_result_t r = zeCommandListHostSynchronize(imm_, UINT64_MAX);
    if (r != ZE_RESULT_SUCCESS) fatal("zeCommandListHostSynchronize", r, __FILE__, __LINE__);
}

bool Stream::sync_for(uint64_t timeout_ms) {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    g_count.host_waits.fetch_add(1, std::memory_order_relaxed);
    const ze_result_t r = zeCommandListHostSynchronize(imm_, timeout_ms * 1000000ull);
    if (r == ZE_RESULT_NOT_READY) return false;
    if (r != ZE_RESULT_SUCCESS) fatal("zeCommandListHostSynchronize", r, __FILE__, __LINE__);
    return true;
}

bool Stream::idle() {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    const ze_result_t r = zeCommandListHostSynchronize(imm_, 0);
    if (r == ZE_RESULT_NOT_READY) return false;
    if (r != ZE_RESULT_SUCCESS) fatal("zeCommandListHostSynchronize(0)", r, __FILE__, __LINE__);
    return true;
}

void Stream::begin_capture() {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    if (cap_) {
        std::fprintf(stderr, "strata ze: begin_capture on a stream that is capturing\n");
        std::exit(1);
    }
    auto* c = new Capture();
    c->dev = dev_;
    DeviceState& st = dev_state(dev_);
    ze_command_list_desc_t ld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ld.commandQueueGroupOrdinal = (uint32_t) (engine_ == Engine::Copy ? st.info.copy_ordinal : st.info.compute_ordinal);
    ld.flags = ZE_COMMAND_LIST_FLAG_IN_ORDER;
    STRATA_ZE_CHECK(zeCommandListCreate(rt().ctx, st.h, &ld, &c->list_));
    c->members.push_back(this);
    cap_ = c;
    g_count.captures.fetch_add(1, std::memory_order_relaxed);
}

Graph* Stream::end_capture() {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    Capture* c = cap_;
    if (c == nullptr) {
        std::fprintf(stderr, "strata ze: end_capture on a stream that is not capturing\n");
        std::exit(1);
    }
    for (Stream* s : c->members) s->cap_ = nullptr;
    STRATA_ZE_CHECK(zeCommandListClose(c->list_));
    Graph* g = new Graph();
    g->list_ = c->list_;
    g->nodes_ = c->nodes;
    g->dev_ = c->dev;
    c->list_ = nullptr;
    c->members.clear();
    return g;
}

void Stream::launch(Graph& g) {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    if (cap_) {
        std::fprintf(stderr, "strata ze: a graph launched into a capture\n");
        std::exit(1);
    }
    if (g.dev_ != dev_) {
        std::fprintf(stderr, "strata ze: a graph of GPU %d launched on a stream of GPU %d\n", g.dev_, dev_);
        std::exit(1);
    }
    STRATA_ZE_CHECK(zeCommandListImmediateAppendCommandListsExp(imm_, 1, &g.list_, nullptr, 0, nullptr));
    g_count.graph_launches.fetch_add(1, std::memory_order_relaxed);
    after_append();
}

Graph::~Graph() {
    if (list_) zeCommandListDestroy(list_);
}

Stream& default_stream(int ordinal) {
    if (ordinal < 0) ordinal = t_dev;
    DeviceState& st = dev_state(ordinal);
    std::call_once(st.def_once, [&] {
        st.def = std::make_unique<Stream>(ordinal, Stream::Engine::Compute);
        st.def->runtime_owned_ = true;
    });
    return *st.def;
}

Stream& copy_stream(int ordinal) {
    if (ordinal < 0) ordinal = t_dev;
    DeviceState& st = dev_state(ordinal);
    static const bool off = !env_on("STRATA_COPY_ENGINE", true);
    if (off || st.info.copy_ordinal < 0) return default_stream(ordinal);
    std::call_once(st.copy_once, [&] {
        st.copy = std::make_unique<Stream>(ordinal, Stream::Engine::Copy);
        st.copy->runtime_owned_ = true;
    });
    return *st.copy;
}

bool has_copy_engine(int ordinal) {
    static const bool off = !env_on("STRATA_COPY_ENGINE", true);
    return !off && dev_state(ordinal).info.copy_ordinal >= 0;
}

void device_sync(int ordinal) {
    DeviceState& st = dev_state(ordinal);
    std::vector<Stream*> ss;
    {
        std::lock_guard<std::mutex> lk(st.streams_mu);
        ss = st.streams;
    }
    for (Stream* s : ss) s->sync();
}

void sync_all() {
    for (int i = 0; i < device_count(); ++i) device_sync(i);
}

Event::Event(bool timing) : timing_(timing) {
    pool_ = new_event_slot(&ev_, &index_);
    if (timing) stamp_ = new_stamp();
}

Event::~Event() {
    if (ev_) {
        zeEventDestroy(ev_);
        Runtime& r = rt();
        std::lock_guard<std::mutex> lk(r.ev_mu);
        r.free_events.emplace_back(pool_, index_);
        if (stamp_) r.free_stamps.push_back((uint32_t) (stamp_ - r.stamps));
    }
}

bool Event::query() {
    if (!recorded_ || captured_in_ != nullptr) return true;
    const ze_result_t r = zeEventQueryStatus(ev_);
    if (r == ZE_RESULT_NOT_READY) return false;
    if (r != ZE_RESULT_SUCCESS) fatal("zeEventQueryStatus", r, __FILE__, __LINE__);
    return true;
}

void Event::sync() {
    if (!recorded_ || captured_in_ != nullptr) return;
    g_count.host_waits.fetch_add(1, std::memory_order_relaxed);
    const ze_result_t r = zeEventHostSynchronize(ev_, UINT64_MAX);
    if (r != ZE_RESULT_SUCCESS) fatal("zeEventHostSynchronize", r, __FILE__, __LINE__);
}

float Event::elapsed_ms_since(const Event& start) const {
    if (!timing_ || !start.timing_ || !recorded_ || !start.recorded_) return 0.0f;
    const DeviceInfo& di = device_info(stamp_dev_);
    const uint64_t ticks = (*stamp_ - *start.stamp_) & di.timestamp_mask;
    return (float) ((double) ticks * (double) di.timer_ns_num / (double) di.timer_ns_den / 1.0e6);
}

void report(const char* when) {
    if (!rt().report_on) return;
    const MemTotals t = mem_totals();
    std::fprintf(stderr,
                 "strata ze report (%s): device allocations %zu (%.2f GiB), host allocations %zu (%.2f GiB), kernel launches "
                 "%llu, copies %llu, graph replays %llu, captures %llu, events %llu, host waits %llu, waits dropped in "
                 "capture %llu, frees %llu, pageable copies %llu\n",
                 when, t.device_n, (double) t.device_bytes / 1073741824.0, t.host_n, (double) t.host_bytes / 1073741824.0,
                 (unsigned long long) g_count.launches.load(), (unsigned long long) g_count.copies.load(),
                 (unsigned long long) g_count.graph_launches.load(), (unsigned long long) g_count.captures.load(),
                 (unsigned long long) g_count.events.load(), (unsigned long long) g_count.host_waits.load(),
                 (unsigned long long) g_count.dropped_waits.load(), (unsigned long long) g_count.frees.load(),
                 (unsigned long long) g_count.pageable.load());
}

}  // namespace strata::ze
