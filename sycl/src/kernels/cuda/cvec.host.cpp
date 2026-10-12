// src/kernels/cuda/cvec.host.cpp - the host side of cvec.dp.cpp: the per-device tables and the launcher.
#include "strata/kernels/cvec.hpp"
#include "strata/ze/kernel_abi.hpp"

#include <stdexcept>

namespace strata::kernels {

#include "cvec.kern.inc"

namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

Cvec g_cvec;                 // the description; its device pointers are the uploading device's
bool g_on_host = false;
// the tables on every device that holds them (a layer split applies the vector on several)
constexpr int kDevices = 64;
struct DevTables { float* dir = nullptr; float* s = nullptr; int* on = nullptr; };
DevTables g_dev[kDevices];
std::vector<float> g_dir_host, g_s_host;
int cur_device() {
    int d = ze::current_device();
    if (d < 0 || d >= kDevices) d = 0;
    return d;
}
bool upload_here(std::string &err) {
    DevTables& t = g_dev[cur_device()];
    if (t.dir != nullptr) return true;
    const int flag = g_on_host ? 1 : 0;
    if ((t.dir = (float*) ze::malloc_device(g_dir_host.size() * sizeof(float))) == nullptr ||
        (t.s = (float*) ze::malloc_device(g_s_host.size() * sizeof(float))) == nullptr ||
        (t.on = (int*) ze::malloc_device(sizeof(int))) == nullptr) {
        err = "control vector: device allocation failed";
        t = DevTables{};
        return false;
    }
    ze::Stream& q = ze::default_stream();
    q.memcpy(t.dir, g_dir_host.data(), g_dir_host.size() * sizeof(float));
    q.memcpy(t.s, g_s_host.data(), g_s_host.size() * sizeof(float));
    q.memcpy(t.on, &flag, sizeof(int));
    q.sync();
    return true;
}

}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    int prev = 0;   // a new vector replaces the old one on every device
    prev = ze::current_device();
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].dir == nullptr) continue;
        ze::set_device(d);
        ze::device_sync();
        ze::free(g_dev[d].dir);
        ze::free(g_dev[d].s);
        ze::free(g_dev[d].on);
        g_dev[d] = DevTables{};
    }
    ze::set_device(prev);
    g_dir_host = dir;
    g_s_host = s;
    g_on_host = true;
    if (!upload_here(err)) return false;
    const DevTables& t = g_dev[cur_device()];
    g_cvec.dir = t.dir;
    g_cvec.s = t.s;
    g_cvec.on = t.on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    return true;
}

bool cvec_replicate(std::string& err) { return !g_cvec.loaded() || upload_here(err); }

void cvec_set_enabled(bool on) {
    if (!g_cvec.loaded() || on == g_on_host) return;
    int prev = 0;
    prev = ze::current_device();
    const int v = on ? 1 : 0;
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].on == nullptr) continue;
        ze::set_device(d);
        ze::device_sync(); // nothing in flight may still read the flag
        ze::copy_sync(g_dev[d].on, &v, sizeof(int));
    }
    ze::set_device(prev);
    g_on_host = on;
}

bool cvec_enabled() { return g_cvec.loaded() && g_on_host; }

bool cvec_tables(const float** dir, const float** s, const int** on) {
    if (!g_cvec.loaded()) return false;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) return false;
    *dir = t.dir; *s = t.s; *on = t.on;
    return true;
}

void cvec_apply(float *R, int64_t layer, int64_t T, int64_t r_ld,
                const float *bo, int64_t bo_ld, const float *inj,
                int64_t inj_ld, bool write, void *stream) {
    if (!g_cvec.loaded() || T < 1) return;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) throw std::runtime_error("cvec_apply: the control vector is not on this device (cvec_replicate)");
    const ze::Dim3 grid{(unsigned)g_cvec.hc, (unsigned)T, 1};
    k_cvec_apply(ze::stream_of(stream), grid, {THREADS, 1, 1}, 0, R, t.dir, t.s, t.on, g_cvec.mode, layer,
                 (int)g_cvec.n_embd, (int)g_cvec.hc, r_ld, bo, bo_ld, inj, inj_ld, write ? 1 : 0);
}

}  // namespace strata::kernels
