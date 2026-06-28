// Standalone validation for expert_cache_lru: replays a routing trace
// (GGML_MOE_ROUTING_TRACE format: "<layer> <e0> <e1> ...") through the C++ LRU
// cache and reports the hit rate, which should match
// tools/predictive-prefetch/simulate_cache.py's "lru" row (~83% code / ~88%
// prose at M=42). No model or CUDA needed.
//
// Build (from repo root, in a VS x64 dev shell):
//   cl /EHsc /std:c++17 /O2 /I ggml/src/ggml-cuda \
//      tools/predictive-prefetch/test_expert_cache.cpp /Fe:test_expert_cache.exe
// Run:
//   test_expert_cache.exe metrics/predictive-prefetch/routing-code.txt 42

#include "expert-cache.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <trace> [slots_per_layer=42]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int M = argc > 2 ? std::atoi(argv[2]) : 42;

    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return 2;
    }

    // per-layer access streams (preserve order)
    std::map<int, std::vector<std::vector<int>>> per_layer;
    int max_expert = 0;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        int layer;
        if (!(ss >> layer)) continue;
        std::vector<int> experts;
        int e;
        while (ss >> e) { experts.push_back(e); if (e > max_expert) max_expert = e; }
        if (!experts.empty()) per_layer[layer].push_back(std::move(experts));
    }
    const int n_experts = max_expert + 1;

    // --- per-layer partitioned cache (matches simulate_cache.py) ---
    long hit = 0, tot = 0;
    for (auto & kv : per_layer) {
        expert_cache_lru cache(M);
        for (auto & toks : kv.second) {
            for (int e : toks) {
                ++tot;
                if (cache.access((uint64_t) e).hit) ++hit;
            }
        }
    }
    const double per_layer_hr = tot ? (double) hit / tot : 0.0;

    // --- single global pool of M * n_layers slots, key = (layer<<32)|expert ---
    const int n_layers = (int) per_layer.size();
    long ghit = 0, gtot = 0;
    {
        expert_cache_lru cache(M * n_layers);
        // replay in token order across all layers: reconstruct interleaving
        // (per pass: layer 0..L). Use the max token count as the outer loop.
        size_t max_tokens = 0;
        for (auto & kv : per_layer) max_tokens = std::max(max_tokens, kv.second.size());
        for (size_t t = 0; t < max_tokens; ++t) {
            for (auto & kv : per_layer) {
                if (t >= kv.second.size()) continue;
                const uint64_t base = (uint64_t) kv.first << 32;
                for (int e : kv.second[t]) {
                    ++gtot;
                    if (cache.access(base | (uint64_t) e).hit) ++ghit;
                }
            }
        }
    }
    const double global_hr = gtot ? (double) ghit / gtot : 0.0;

    std::printf("%s\n", path.c_str());
    std::printf("  layers=%d  experts=%d  slots/layer M=%d  accesses=%ld\n",
                n_layers, n_experts, M, tot);
    std::printf("  per-layer LRU hit rate: %.1f%%\n", per_layer_hr * 100.0);
    std::printf("  global-pool LRU hit rate (%d slots): %.1f%%\n", M * n_layers, global_hr * 100.0);
    return 0;
}
