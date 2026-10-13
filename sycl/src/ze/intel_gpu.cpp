// src/ze/intel_gpu.cpp - see include/strata/intel_gpu.hpp.
#include "strata/intel_gpu.hpp"

#include <filesystem>
#include <fstream>

namespace strata {

const std::string& intel_gpu_driver() {
    static const std::string drv = [] {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
            const std::string n = e.path().filename().string();
            if (n.rfind("card", 0) != 0 || n.find('-') != std::string::npos) continue;
            std::ifstream vf(e.path() / "device" / "vendor");
            std::string vendor;
            if (!(vf >> vendor) || vendor != "0x8086") continue;
            const auto d = std::filesystem::read_symlink(e.path() / "device" / "driver", ec);
            if (!ec) return d.filename().string();
        }
        return std::string();
    }();
    return drv;
}

IntelGpuGen intel_gpu_gen(int ordinal) {
    const ze::DeviceInfo& d = ze::device_info(ordinal < 0 ? ze::current_device() : ordinal);
    if (d.battlemage) return IntelGpuGen::Battlemage;
    const uint32_t id = d.device_id;
    if ((id >= 0x5690 && id <= 0x56BF) || id == 0x56C0 || id == 0x56C1) return IntelGpuGen::Alchemist;
    return IntelGpuGen::Unknown;
}

}  // namespace strata
