// src/core/graph.cpp - P2.S5: the GraphRegistry implementation, on Level Zero recorded command lists.
#include "strata/ze/ze.hpp"
#include "strata/core/graph.hpp"

#include <immintrin.h>

#include <chrono>
#include <cstdio>

namespace strata::core {

CapturedGraph& CapturedGraph::operator=(CapturedGraph&& o) noexcept {
    if (this != &o) {
        reset();
        graph_ = o.graph_;
        done_ = o.done_;
        nodes_ = o.nodes_;
        o.graph_ = nullptr;
        o.done_ = nullptr;
        o.nodes_ = 0;
    }
    return *this;
}

void CapturedGraph::reset() {
    // ORDER MATTERS: destroying a recorded list while a replay of it runs is undefined, so the event that marks the
    // last replay is waited for first.  `bench/micro/graph_capture.cu` was fooled by the CUDA form of this once - it
    // looked like captured graphs made a doorbell visible and direct launches did not.
    if (done_) { done_->sync(); delete done_; done_ = nullptr; }
    if (graph_) { delete graph_; graph_ = nullptr; }
    nodes_ = 0;
}

bool CapturedGraph::begin(void* stream, std::string& err) {
    if (graph_) { err = "begin: this CapturedGraph is already recorded"; return false; }
    strata::ze::stream_of(stream)->begin_capture();
    return true;
}

bool CapturedGraph::end(void* stream, std::string& err) {
    graph_ = strata::ze::stream_of(stream)->end_capture();
    // A capture that recorded NOTHING is a wiring mistake, and a graph that replays nothing produces no error
    // and no output - the silent kind of failure this project keeps paying for.
    if (graph_ == nullptr) { err = "end: the capture recorded ZERO nodes - the body launched nothing"; reset(); return false; }
    nodes_ = graph_->nodes();
    // The completion event is created ONCE and reused: `launch` records it again after each replay, which is
    // what makes `wait_ms` a query rather than a sync.
    done_ = new strata::ze::Event();
    return true;
}

bool CapturedGraph::launch(void* stream, std::string& err) const {
    if (!graph_) { err = "launch: not recorded"; return false; }
    strata::ze::Stream* s = strata::ze::stream_of(stream);
    s->launch(*graph_);
    s->record(*done_);
    return true;
}

bool CapturedGraph::wait_ms(int timeout_ms) const {
    if (!done_) return false;
    // A BOUNDED wait, and the bound is the point: an unbounded spin turns a protocol bug into a hung run, and
    // a hung run says nothing about which side is stuck.  The event query does not block and does not
    // synchronise, so this satisfies P2.X3.
    //
    // The deadline is a REAL CLOCK, not a count of pause instructions: `_mm_pause` is a few cycles, so
    // counting pauses as microseconds would make the timeout tens of times longer than the caller asked for -
    // a timeout that does not time out is worse than none, because it reports a hang as a pass.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        if (done_->query()) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        for (int i = 0; i < 64; ++i) _mm_pause();
    }
}

bool GraphRegistry::record(LayerType type, int n_tokens, const std::function<void()>& body, std::string& err) {
    const Key k{(int) type, n_tokens};
    if (graphs_.count(k)) return true;              // already recorded: the point of a registry

    CapturedGraph g;
    if (!g.begin(stream_, err)) return false;
    body();                                        // the caller launches into the captured stream
    if (!g.end(stream_, err)) return false;

    graphs_.emplace(k, std::move(g));
    ++captures_;
    return true;
}

const CapturedGraph* GraphRegistry::find(LayerType type, int n_tokens) const {
    const auto it = graphs_.find(Key{(int) type, n_tokens});
    return it == graphs_.end() ? nullptr : &it->second;
}

bool GraphRegistry::launch(LayerType type, int n_tokens, int timeout_ms, std::string& err) const {
    const CapturedGraph* g = find(type, n_tokens);
    if (!g) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "launch: no graph recorded for %s n=%d", to_string(type), n_tokens);
        err = buf;
        return false;
    }
    if (!g->launch(stream_, err)) return false;
    if (!g->wait_ms(timeout_ms)) { err = "launch: timed out waiting for the graph to complete"; return false; }
    return true;
}

}  // namespace strata::core
