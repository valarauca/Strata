// src/gemmstone/gs_probe.cpp - generates the gemmstone kernels of the model's prompt shapes on the CPU alone (no GPU is
// opened) and prints the strategy each would use.
//
//   gs_probe [out_dir]    with out_dir: also write each binary there (gs_<case>.bin) for ocloc disasm
//   gs_probe --classes    every power-of-two token count 1..8192 of the Q4 expert shapes (gate/up, down)
#include "strata/gemmstone.hpp"

#include <cstring>

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    struct Case {
        const char* what;
        strata::gs::Weights w;
        long long T, N, K;
        bool acc;
    };
    using W = strata::gs::Weights;
    const Case cases[] = {
        {"q4_gate_up_8", W::Q4, 8, 1280, 2560, false},     {"q4_gate_up_161", W::Q4, 161, 1280, 2560, false},
        {"q4_gate_up_420", W::Q4, 420, 1280, 2560, false}, {"q4_down_161", W::Q4, 161, 2560, 640, false},
        {"f16_gate_up_161", W::F16, 161, 1280, 2560, false}, {"f16_down_420", W::F16, 420, 2560, 640, false},
        {"f16_dense_4096", W::F16, 4096, 2560, 2560, false}, {"bf16_dense_4096", W::BF16, 4096, 2560, 2560, false},
        {"bf16_dense_acc", W::BF16, 4096, 640, 2560, true},  {"f16_dense_wide", W::F16, 4096, 10240, 2560, false},
    };
    int missing = 0;
    if (argc > 1 && std::strcmp(argv[1], "--classes") == 0) {
        for (long long T = 1; T <= 8192; T *= 2)
            for (const auto& nk : {std::pair<long long, long long>{1280, 2560}, {2560, 640}}) {
                std::string what;
                const std::vector<uint8_t> bin = strata::gs::probe(W::Q4, T, nk.first, nk.second, false, 256, what);
                std::printf("q4 T %5lld N %5lld K %5lld  %6zu B  %.90s\n", T, nk.first, nk.second, bin.size(), what.c_str());
                missing += bin.empty();
            }
        return missing == 0 ? 0 : 1;
    }
    for (const Case& c : cases) {
        std::string what;
        const std::vector<uint8_t> bin = strata::gs::probe(c.w, c.T, c.N, c.K, c.acc, 256, what);
        std::printf("%-18s T %5lld N %6lld K %6lld  %6zu B  %s\n", c.what, c.T, c.N, c.K, bin.size(), what.c_str());
        missing += bin.empty();
        if (argc > 1 && !bin.empty()) {
            const std::string path = std::string(argv[1]) + "/gs_" + c.what + ".bin";
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(bin.data(), 1, bin.size(), f);
                std::fclose(f);
            }
        }
    }
    return missing == 0 ? 0 : 1;
}
