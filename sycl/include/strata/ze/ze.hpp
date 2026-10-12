// include/strata/ze/ze.hpp - the Intel engine's GPU runtime: Level Zero, owned end to end.
//
// One driver, one context over the GPUs Strata was given, and every GPU object Strata touches made and destroyed
// here: device and host allocations, in-order command streams on the compute and copy queue groups, counter-based
// events, recorded command lists for the decode windows, and the kernel modules built offline into the binary.
//
// Ownership rules, which the rest of the engine relies on:
// - Every allocation comes from malloc_device / malloc_host and goes back through free. free() waits for the work of
//   the devices that can reach the allocation before it releases it: the driver here has no deferred-free policy, and a
//   free under a running kernel is a page fault on xe.
// - A Stream is one immediate in-order command list. Appends from several threads are serialised by the stream.
//   Ordering between streams is only what Events say; there is no implicit device-wide order.
// - A Graph is a closed regular command list recorded from one stream. Kernel arguments are captured when they are
//   appended, so anything that changes per replay must be data in a buffer the recorded kernels read.
// - Host waits are explicit: Stream::sync, Event::sync, device_sync. Nothing in this file waits on the host unless it
//   is asked to, except free() (above) and STRATA_L0_SERIALIZE=1 (a debug mode that syncs after every append).
#pragma once

#include <level_zero/ze_api.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::ze {

/// Prints the failed call, its result and the location, then exits the process. DEVICE_LOST lands here.
[[noreturn]] void fatal(const char* what, ze_result_t r, const char* file, int line);
/// The ZE_RESULT_* name of `r`.
const char* result_name(ze_result_t r);

#define STRATA_ZE_CHECK(call)                                                                  \
    do {                                                                                       \
        const ze_result_t strata_ze_r_ = (call);                                               \
        if (strata_ze_r_ != ZE_RESULT_SUCCESS) ::strata::ze::fatal(#call, strata_ze_r_, __FILE__, __LINE__); \
    } while (0)

/// Upper bound on the GPUs one process uses.
constexpr int kMaxDevices = 8;

/// Properties of one GPU Strata uses.
struct DeviceInfo {
    std::string name;
    std::string pci;             ///< "0000:03:00.0"
    uint32_t device_id = 0;      ///< PCI device id (0xe223 for the Arc Pro B70)
    uint64_t total_mem = 0;      ///< device-local memory in bytes
    uint64_t max_alloc = 0;
    uint32_t eu_count = 0;
    uint32_t max_group_size = 0;
    uint32_t max_slm = 0;        ///< shared local memory per work-group, bytes
    uint64_t timer_ns_num = 1;   ///< global timestamp ticks -> ns: ns = ticks * num / den
    uint64_t timer_ns_den = 1;
    uint64_t timestamp_mask = ~0ull;
    bool battlemage = false;     ///< Xe2 discrete (BMG-G21 / G31)
    bool integrated = false;
    uint32_t ip_version = 0;     ///< graphics IP version (ZE_extension_device_ip_version), 0 when not reported
    int compute_ordinal = 0;     ///< queue group with compute
    int copy_ordinal = -1;       ///< copy-only queue group, -1 when the device has none
};

/// Number of GPUs Strata uses: STRATA_ZE_DEVICES ("0,1" or PCI addresses), or ONEAPI_DEVICE_SELECTOR="level_zero:0,1"
/// kept as an alias for existing configs (its level_zero GPUs, other backends ignored), else every GPU of the driver.
/// Ordinals below index this list.
int device_count();
/// The calling thread's current GPU (default 0).
int current_device();
/// Make `ordinal` the calling thread's current GPU.
void set_device(int ordinal);
/// Properties of GPU `ordinal`.
const DeviceInfo& device_info(int ordinal);
/// The Level Zero handle of GPU `ordinal`.
ze_device_handle_t device_handle(int ordinal);
/// The one context every GPU object of the process lives in.
ze_context_handle_t context();
/// Free and total device memory of GPU `ordinal` in bytes, from Sysman when available, else total minus what Strata
/// allocated. Returns false when neither is known.
bool mem_get_info(int ordinal, size_t* free_bytes, size_t* total_bytes);
/// True when GPU `a` can read and write memory of GPU `b` directly.
bool can_access_peer(int a, int b);

/// What an address is, for diagnostics and copies.
enum class MemKind { Unknown, Device, Host, Shared };

/// Device-local memory on GPU `ordinal` (-1: current). Large requests use relaxed allocation limits.
void* malloc_device(size_t bytes, int ordinal = -1);
/// malloc_device for a large arena, checked for the xe aliasing fault (two GPU pages of one allocation mapped onto the
/// same memory, seen on kernel 7.0): every 64 KiB is tagged and read back by two kernels, and an aliased range is
/// allocated again behind a growing spacer. Requests under 32 MiB and STRATA_ARENA_ALIAS_CHECK=0 skip the check.
void* malloc_device_guarded(size_t bytes, int ordinal, const char* what);
/// Pinned host memory every GPU of the context can read and write (zeMemAllocHost).
void* malloc_host(size_t bytes);
/// Pinned host memory with the uncached bias, for flags a running kernel polls (the doorbell).
void* malloc_host_uncached(size_t bytes);
/// Maps existing page-aligned system memory [p, p + bytes) for every GPU of the context (ZE_extension_external_memmap_sysmem),
/// keeping its address, so kernels and copies may use it like malloc_host memory. False when the driver refuses; free(p)
/// releases the mapping, not the memory. Mapped ranges must not overlap.
bool import_host(void* p, size_t bytes);
/// Releases a Strata allocation. It returns at once: the memory goes back to the driver once every stream of every GPU
/// has passed the point where free was called (a fence on each), and an allocation that finds no room first waits for
/// the pending ones.
void free(void* p);
/// free(p), then a host wait until p and every free before it are back with the driver: for memory whose backing goes
/// away next (an import_host range before munmap) or whose address must not come back from the next allocation.
void free_sync(void* p);
/// The kind and owning GPU of `p` (dev = -1 for host memory or unknown pointers).
MemKind kind_of(const void* p, int* dev = nullptr);
/// How far into its Strata device allocation the range [p, p + bytes) ends; 0 when p is not in one.
size_t device_offset_end(const void* p, size_t bytes);
/// The process's live allocations: count and bytes by kind.
struct MemTotals { size_t device_n = 0, device_bytes = 0, host_n = 0, host_bytes = 0; };
MemTotals mem_totals();

/// Launch geometry in Level Zero order: x is the fastest dimension (SYCL's dimension 2, CUDA's x).
struct Dim3 {
    uint32_t x = 1, y = 1, z = 1;
};

class Stream;
class Event;
class Graph;

/// One kernel of the offline-built modules, found by name on first use per GPU.
class KernelBase {
public:
    constexpr KernelBase(const char* name, uint32_t nparams) : name_(name), nparams_(nparams) {}
    const char* name() const { return name_; }
    uint32_t nparams() const { return nparams_; }
    /// The kernel's handle on GPU `ordinal`; exits with a message when the modules do not have it or its argument
    /// count is not the declared one.
    ze_kernel_handle_t handle(int ordinal) const;

private:
    ze_kernel_handle_t create(int ordinal) const;
    const char* name_;
    uint32_t nparams_;
    mutable std::atomic<ze_kernel_handle_t> h_[kMaxDevices] = {};
};

/// A kernel generated at run time as a native device binary (strata/gemmstone.hpp). The runtime creates its module and
/// kernel and keeps both for the life of the process, so a NativeKernel is a plain value that never owns anything.
class NativeKernel {
public:
    NativeKernel() = default;
    const char* name() const { return name_; }
    int device() const { return dev_; }
    uint32_t nparams() const { return nparams_; }
    ze_kernel_handle_t handle() const { return h_; }
    explicit operator bool() const { return h_ != nullptr; }

private:
    friend NativeKernel load_native(int, const std::vector<uint8_t>&, const std::string&);
    const char* name_ = "";
    int dev_ = -1;
    uint32_t nparams_ = 0;
    ze_kernel_handle_t h_ = nullptr;
};

/// Loads the native binary `binary` on GPU `ordinal` and creates its kernel `name`. Returns an empty NativeKernel when
/// the driver refuses the binary (the build log goes to stderr). The runtime keeps the module and the kernel until the
/// process exits.
NativeKernel load_native(int ordinal, const std::vector<uint8_t>& binary, const std::string& name);

/// A kernel entry point with its C++ signature, declared by STRATA_KERNEL (strata/ze/kernel_abi.hpp).
template <class Sig> class Kernel;
template <class... P> class Kernel<void(P...)> : public KernelBase {
public:
    constexpr explicit Kernel(const char* name) : KernelBase(name, sizeof...(P)) {}
    /// Appends the launch to `s` (nullptr: the current GPU's default stream). `slm` is the dynamic work-group scratch
    /// memory in bytes. Each argument is converted to the declared parameter type before it is passed.
    template <class... A> void operator()(Stream* s, Dim3 grid, Dim3 block, size_t slm, A&&... a) const;
};

/// An in-order command stream on one engine of one GPU (an immediate command list).
class Stream {
public:
    enum class Engine { Compute, Copy };
    /// A stream on GPU `ordinal` (-1: current). `high_priority` asks the scheduler to prefer it.
    explicit Stream(int ordinal = -1, Engine engine = Engine::Compute, bool high_priority = false);
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    int device() const { return dev_; }
    Engine engine() const { return engine_; }

    /// Appends kernel `k` with argument pointers `args` (k.nparams() of them, then a pointer to the size_t scratch size).
    void launch(const KernelBase& k, Dim3 grid, Dim3 block, void** args);
    /// Appends native kernel `k` with argument pointers `args` (k.nparams() of them, a size_t* for a local-memory one).
    void launch(const NativeKernel& k, Dim3 grid, Dim3 block, void** args);
    /// Copies between device memory and Strata host memory are asynchronous. A copy from or to pageable host memory
    /// (not from malloc_host) goes through a pinned bounce buffer and returns when it is done, as CUDA's does.
    void memcpy(void* dst, const void* src, size_t bytes);
    /// A 2-D copy: `height` rows of `width` bytes, row pitches in bytes.
    void memcpy_2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height);
    /// Sets `bytes` bytes at `dst` to `value` (a fill with a one-byte pattern).
    void memset(void* dst, int value, size_t bytes);
    /// Fills `bytes` with a repeated pattern of 1, 2, 4, 8, 16, 32, 64 or 128 bytes.
    void fill(void* dst, const void* pattern, size_t pattern_bytes, size_t bytes);
    /// Writes the GPU's global timestamp (ticks) into `dst` (host or device memory) when the work before it is done.
    void write_timestamp(uint64_t* dst);
    /// `e` completes when everything appended to this stream before it has.
    void record(Event& e);
    /// Work appended to this stream after this call starts after `e`'s last record. When that record is on another GPU
    /// the host waits for it here: a GPU may only wait on a counter-based event of a peer it has access to, and the
    /// event's counter lives with the GPU that signaled it.
    void wait(Event& e);
    /// Host wait for everything appended so far, host functions included. Exits on DEVICE_LOST.
    void sync();
    /// Host wait with a bound; false when `timeout_ms` passed first.
    bool sync_for(uint64_t timeout_ms);
    /// True when everything appended so far has finished, host functions included (no wait).
    bool idle();

    /// From here until end_capture, appends go to a new regular command list instead of the GPU. A stream that waits
    /// on an event recorded inside the capture joins it: its appends go to the same list, in order.
    void begin_capture();
    /// Closes the capture into a Graph (nodes() == 0 when nothing was recorded). Streams that joined leave it.
    Graph* end_capture();
    bool capturing() const { return cap_ != nullptr; }
    /// Appends a recorded graph.
    void launch(Graph& g);

    /// Number of commands this stream appended (a counter for STRATA_ZE_REPORT).
    uint64_t appends() const { return appends_.load(std::memory_order_relaxed); }
    ze_command_list_handle_t immediate_list() const { return imm_; }

    /// The open recording of begin_capture, shared by the streams that joined it, then owned by its Graph.
    struct Capture {
        ze_command_list_handle_t list_ = nullptr;
        int dev = 0;
        size_t nodes = 0;
        std::vector<Stream*> members;
        std::vector<Event*> events;   ///< recorded inside: signaled again at the end of each replay
    };

private:
    friend class Event;
    friend Stream& default_stream(int);
    friend Stream& copy_stream(int);
    friend void host_func(Stream& s, void (*fn)(void*), void* arg);
    /// The runtime's host-function worker reports a finished callback of `s` (strata/ze/ze_util.cpp).
    friend void host_func_finished(Stream& s);
    friend void free(void* p);
    friend void device_sync(int ordinal);
    ze_command_list_handle_t target() const { return cap_ ? cap_->list_ : imm_; }
    void after_append();
    void append_launch(ze_kernel_handle_t h, const char* name, int dev, Dim3 grid, Dim3 block, void** args);
    void check_operand(const void* p, const char* what) const;
    void host_func_done();
    void wait_host_funcs();
    static constexpr int kBounce = 4;
    int dev_;
    Engine engine_;
    ze_command_list_handle_t imm_ = nullptr;
    Capture* cap_ = nullptr;
    void* bounce_[kBounce] = {};
    Event* bounce_ev_[kBounce] = {};
    int bounce_next_ = 0;
    bool runtime_owned_ = false;
    std::recursive_mutex mu_;
    std::atomic<uint64_t> appends_{0};
    std::mutex hf_mu_;
    std::condition_variable hf_cv_;
    int hf_pending_ = 0;
};

/// The current GPU's default compute stream (ordinal -1), shared by every thread of the process.
Stream& default_stream(int ordinal = -1);
/// The GPU's copy-engine stream, shared; the compute default stream when the GPU has no copy-only queue group or
/// STRATA_COPY_ENGINE=0.
Stream& copy_stream(int ordinal = -1);
/// True when copy_stream(ordinal) is on a copy-only engine.
bool has_copy_engine(int ordinal = -1);
/// The stream a launcher was handed: `s` itself, or the current GPU's default stream for nullptr.
inline Stream* stream_of(const void* s) { return s ? (Stream*) s : &default_stream(); }
/// Host wait for every stream of GPU `ordinal` (-1: current).
void device_sync(int ordinal = -1);
/// Host wait for every stream of every GPU.
void sync_all();
/// Copies `bytes` from `src` to `dst` on GPU `ordinal`'s default stream (-1: current) and waits for that stream, so the
/// copy and everything queued before it have completed on return. Aborts on a null pointer with a nonzero size.
void copy_sync(void* dst, const void* src, size_t bytes, int ordinal = -1);

/// A counter-based event: records a point in a stream, waited on by streams or the host.
class Event {
public:
    /// An event usable by every GPU of the context. `timing` events also carry a GPU timestamp for elapsed_ms.
    explicit Event(bool timing = false);
    ~Event();
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    /// True once the last record has completed (true when never recorded).
    bool query();
    /// Host wait for the last record. Exits on DEVICE_LOST.
    void sync();
    /// True once the event was recorded (in a stream or inside a capture).
    bool recorded() const { return recorded_; }
    /// Milliseconds between two recorded timing events, both complete. A timing event recorded inside a capture carries
    /// the GPU time of the graph's last replay.
    float elapsed_ms_since(const Event& start) const;

private:
    friend class Stream;
    friend class Graph;
    friend void free(void* p);
    ze_event_handle_t ev_ = nullptr;
    void* pool_ = nullptr;
    uint32_t index_ = 0;
    bool recorded_ = false;
    bool timing_ = false;
    uint64_t* stamp_ = nullptr;
    int stamp_dev_ = 0;
    int dev_ = -1;                       ///< GPU of the last signal (a stream record or a graph replay)
    Stream::Capture* captured_in_ = nullptr;
    bool replayed_ = false;              ///< recorded in a capture whose graph has been launched since
};

/// A recorded command list, replayed with Stream::launch(Graph&).
class Graph {
public:
    ~Graph();
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    size_t nodes() const { return nodes_; }
    int device() const { return dev_; }

private:
    friend class Stream;
    Graph() = default;
    ze_command_list_handle_t list_ = nullptr;
    size_t nodes_ = 0;
    int dev_ = 0;
    Stream::Capture* cap_ = nullptr;
};

/// Zeroes a large device range with compute kernels of at most `chunk` bytes, the host waiting after each: on the
/// Arc Pro B70 (xe) one memset of many GiB runs on the blitter and times out ("Engine memory CAT error", GT reset).
/// Small ranges, unaligned ones and STRATA_CHUNKED_FILL=0 use the stream's memset.
void fill_zero(Stream& s, void* p, size_t bytes, size_t chunk = (size_t) 256 << 20);

/// Runs `fn` on a runtime thread once everything appended to `s` before this call has finished, without blocking the
/// caller (CUDA's host function). Not allowed while `s` is capturing.
void host_func(Stream& s, void (*fn)(void*), void* arg);

/// STRATA_ZE_REPORT=1: one line of counters (allocations by kind, appends, graph replays, host waits) to stderr.
void report(const char* when);

/// Exits the process when a launch would be invalid: an empty group or one above 1024 work-items. A grid with a zero
/// dimension launches nothing (as in SYCL and CUDA).
void check_geometry(const KernelBase& k, Dim3 grid, Dim3 block);

/// STRATA_ZE_CHECK_DEVICES=1: every launch is checked against the GPU it runs on (see check_launch_ptr), and a launch
/// from a thread whose current GPU is another one is reported.
bool check_devices();
/// Exits with a message when `p` is a device allocation of a GPU other than `dev`: argument `index` of kernel `name`
/// would read or write memory that GPU never mapped.
void check_launch_ptr(const char* name, int index, const void* p, int dev);

template <class... P>
template <class... A>
void Kernel<void(P...)>::operator()(Stream* s, Dim3 grid, Dim3 block, size_t slm, A&&... a) const {
    static_assert(sizeof...(A) == sizeof...(P), "kernel launch: argument count differs from the declared signature");
    if (s == nullptr) s = &default_stream();
    std::tuple<std::remove_cv_t<P>...> v{static_cast<std::remove_cv_t<P>>(std::forward<A>(a))...};
    if (check_devices()) {
        int i = 0;
        auto one = [&](const auto& e) {
            if constexpr (std::is_pointer_v<std::decay_t<decltype(e)>>) check_launch_ptr(name(), i, (const void*) e, s->device());
            ++i;
        };
        std::apply([&](const auto&... e) { (one(e), ...); }, v);
    }
    void* argv[sizeof...(P) + 1];
    std::apply([&](auto&... e) {
        size_t i = 0;
        ((argv[i++] = (void*) &e), ...);
    }, v);
    argv[sizeof...(P)] = &slm;
    s->launch(*this, grid, block, argv);
}

}  // namespace strata::ze
