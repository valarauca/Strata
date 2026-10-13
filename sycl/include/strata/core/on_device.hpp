#pragma once
// Multi-GPU: make `device` current for a scope and restore the caller's device after it.  An object that owns
// streams, graphs and buffers on one device (a verify stage, the drafter) wraps its public calls in this, so the
// caller's thread may be on any device.  A negative device, or the device already current, does nothing.

#include "strata/ze/ze.hpp"

namespace strata::core {

struct OnDevice {
    int previous = -1;
    explicit OnDevice(int device) {
        const int cur = strata::ze::current_device();
        if (device >= 0 && cur != device) {
            strata::ze::set_device(device);
            previous = cur;
        }
    }
    ~OnDevice() { if (previous >= 0) strata::ze::set_device(previous); }
    OnDevice(const OnDevice&) = delete;
    OnDevice& operator=(const OnDevice&) = delete;
};

}  // namespace strata::core
