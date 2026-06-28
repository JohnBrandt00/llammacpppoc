// CUDA implementation of the persistent MoE expert-weight cache.
//
// Backs the ggml_backend_set_moe_expert_cache_cpy hook (see
// ggml-backend-impl.h): a dedicated VRAM slot pool the graph allocator never
// touches, keyed by the expert's (unique, stable) host source address. On a
// hit the expert weights are already resident, so they are served device->device
// into input_cpy; on a miss they are loaded host->device once and kept resident
// across tokens. The mul_mat_id GEMM still reads input_cpy and is unchanged.
//
// Enable with GGML_MOE_EXPERT_CACHE=1; size with GGML_MOE_EXPERT_CACHE_BYTES
// (default 3 GiB). The LRU policy was selected offline (see
// tools/predictive-prefetch/simulate_cache.py).

#include "common.cuh"
#include "expert-cache.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <mutex>

// cudaMemcpyBatchAsync was added in CUDA 12.8; without it the batched path is
// unavailable and the cache falls back to per-expert copies.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
#define GGML_CUDA_HAS_MEMCPY_BATCH 1
#else
#define GGML_CUDA_HAS_MEMCPY_BATCH 0
#endif

struct cuda_expert_cache {
    void *   pool        = nullptr;
    size_t   expert_size = 0;   // bytes per expert (no padding)
    size_t   slot_size   = 0;   // expert_size + read-ahead padding
    int      n_slots     = 0;
    int      device      = -1;
    bool     failed      = false;
    expert_cache_lru * lru = nullptr;

    // runtime stats (guarded by g_cuda_expert_cache_mutex)
    uint64_t hits        = 0;
    uint64_t misses      = 0;
    uint64_t miss_bytes  = 0;   // host->device bytes actually transferred
    uint64_t stats_dumped = 0;

    // reusable scratch for the batched (cudaMemcpyBatchAsync) path
    std::vector<void *>       serve_dst, pop_dst;
    std::vector<const void *> serve_src, pop_src;
    std::vector<size_t>       serve_size, pop_size;
};

static bool cuda_expert_cache_stats_enabled() {
    static const int enabled = []() {
        const char * env = getenv("GGML_MOE_EXPERT_CACHE_STATS");
        return env && atoi(env) != 0 ? 1 : 0;
    }();
    return enabled != 0;
}

static bool cuda_expert_cache_batch_enabled() {
    // default on (when the batched API is available); set the env to 0 to force
    // the per-expert path
    static const int enabled = []() {
        const char * env = getenv("GGML_MOE_EXPERT_CACHE_BATCH");
        return env ? (atoi(env) != 0 ? 1 : 0) : 1;
    }();
    return enabled != 0;
}

static cuda_expert_cache g_cuda_expert_cache;
static std::mutex        g_cuda_expert_cache_mutex;

static size_t cuda_expert_cache_budget_bytes() {
    const char * env = getenv("GGML_MOE_EXPERT_CACHE_BYTES");
    if (env != nullptr) {
        const long long v = atoll(env);
        if (v > 0) {
            return (size_t) v;
        }
    }
    return (size_t) 2 << 30; // 2 GiB
}

static bool cuda_expert_cache_init(cuda_expert_cache & c, int device, size_t expert_size) {
    const size_t padding   = expert_size < 512 ? expert_size : 512;
    const size_t slot_size = expert_size + padding;

    // Never blindly grab the requested budget: the pool competes with the KV
    // cache and compute buffers. Cap to the currently free VRAM minus a headroom
    // so the model's own allocations can't be starved (which would OOM-crash).
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        c.failed = true;
        return false;
    }
    // The pool is allocated lazily on the first MoE copy, so keep a generous
    // headroom for compute buffers / fragmentation that may still grow after.
    // Better a smaller cache than an OOM crash. The default budget is modest;
    // raise GGML_MOE_EXPERT_CACHE_BYTES if you have spare VRAM.
    const size_t headroom = (size_t) 3 << 29; // keep 1.5 GiB free
    const size_t usable   = free_bytes > headroom ? free_bytes - headroom : 0;
    const size_t budget   = std::min(cuda_expert_cache_budget_bytes(), usable);
    const int    n_slots  = (int) (budget / slot_size);

    fprintf(stderr, "ggml_moe_expert_cache: device %d free=%.2f GiB -> pool %d slots x %zu B = %.2f GiB\n",
        device, (double) free_bytes / (1<<30), n_slots, slot_size,
        (double) ((size_t) n_slots * slot_size) / (1<<30));
    fflush(stderr);

    if (n_slots < 64) { // too small to be worth it
        c.failed = true;
        return false;
    }

    void * pool = nullptr;
    if (cudaMalloc(&pool, (size_t) n_slots * slot_size) != cudaSuccess) {
        (void) cudaGetLastError(); // clear the error, fall back to plain copies
        c.failed = true;
        return false;
    }

    c.pool        = pool;
    c.expert_size = expert_size;
    c.slot_size   = slot_size;
    c.n_slots     = n_slots;
    c.device      = device;
    c.lru         = new expert_cache_lru(n_slots);

    GGML_LOG_INFO("%s: %d slots x %zu B = %.2f GiB on device %d\n", __func__,
        n_slots, slot_size, (double) ((size_t) n_slots * slot_size) / (1024.0 * 1024.0 * 1024.0), device);
    return true;
}

extern "C" bool ggml_cuda_moe_expert_cache_cpy(
        ggml_backend_t backend, struct ggml_tensor * dst,
        const struct ggml_tensor * src, const uint32_t * used_ids,
        int64_t n_expert, size_t expert_size) {
    // only handle the CUDA backend; anything else falls back to a normal copy
    if (!ggml_backend_is_cuda(backend)) {
        return false;
    }

    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    cudaStream_t stream = ctx->stream();

    std::lock_guard<std::mutex> lock(g_cuda_expert_cache_mutex);
    cuda_expert_cache & c = g_cuda_expert_cache;

    if (c.pool == nullptr) {
        if (c.failed || !cuda_expert_cache_init(c, ctx->device, expert_size)) {
            return false;
        }
    }

    // the pool is sized for one expert size on one device; anything else falls
    // back to the normal grouped host->device copy
    if (expert_size != c.expert_size || ctx->device != c.device) {
        return false;
    }

    const bool   batch    = cuda_expert_cache_batch_enabled() && GGML_CUDA_HAS_MEMCPY_BATCH;
    const bool   stats    = cuda_expert_cache_stats_enabled();
    const size_t pad      = expert_size < 512 ? expert_size : 512;
    char * const       dst_base = (char *) dst->data;
    const char * const src_base = (const char *) src->data;

    if (batch) {
        c.serve_dst.clear(); c.serve_src.clear(); c.serve_size.clear();
        c.pop_dst.clear();   c.pop_src.clear();   c.pop_size.clear();
    }

    for (int64_t e = 0; e < n_expert; ++e) {
        if (!((used_ids[e >> 5] >> (e & 31)) & 1u)) {
            continue;
        }
        const size_t       off       = (size_t) e * expert_size;
        const size_t       copy_size = expert_size + (e < n_expert - 1 ? pad : 0);
        const void * const host_src  = src_base + off;
        char * const       cpy_dst   = dst_base + off;

        // key = the expert's unique, stable host address
        const expert_cache_lru::access_result r = c.lru->access((uint64_t) (uintptr_t) host_src);
        char * const slot = (char *) c.pool + (size_t) r.slot * c.slot_size;

        if (batch) {
            // serve every expert from its slot (D2D); misses are loaded into
            // their slot first by the populate batch, which runs before serve
            c.serve_dst.push_back(cpy_dst);
            c.serve_src.push_back(slot);
            c.serve_size.push_back(copy_size);
            if (!r.hit) {
                c.pop_dst.push_back(slot);
                c.pop_src.push_back(host_src);
                c.pop_size.push_back(copy_size);
            }
        } else {
            if (!r.hit) {
                CUDA_CHECK(cudaMemcpyAsync(slot, host_src, copy_size, cudaMemcpyHostToDevice, stream));
            }
            CUDA_CHECK(cudaMemcpyAsync(cpy_dst, slot, copy_size, cudaMemcpyDeviceToDevice, stream));
        }

        if (stats) {
            if (r.hit) { c.hits++; } else { c.misses++; c.miss_bytes += copy_size; }
        }
    }

#if GGML_CUDA_HAS_MEMCPY_BATCH
    if (batch && !c.serve_dst.empty()) {
        cudaMemcpyAttributes attr = {};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t attr_idx = 0;
        // load misses into their slots first, then serve all experts into
        // input_cpy; both are stream-ordered so serve sees the loaded slots
        if (!c.pop_dst.empty()) {
            CUDA_CHECK(cudaMemcpyBatchAsync(c.pop_dst.data(), c.pop_src.data(), c.pop_size.data(),
                c.pop_dst.size(), &attr, &attr_idx, 1, stream));
        }
        CUDA_CHECK(cudaMemcpyBatchAsync(c.serve_dst.data(), c.serve_src.data(), c.serve_size.data(),
            c.serve_dst.size(), &attr, &attr_idx, 1, stream));
    }
#endif

    if (stats) {
        const uint64_t total = c.hits + c.misses;
        if (total - c.stats_dumped >= 200000) {
            c.stats_dumped = total;
            fprintf(stderr, "ggml_moe_expert_cache_stats: accesses=%llu hit_rate=%.1f%% miss_GiB=%.2f mode=%s\n",
                (unsigned long long) total, 100.0 * (double) c.hits / (double) total,
                (double) c.miss_bytes / (1024.0 * 1024.0 * 1024.0), batch ? "batch" : "per-expert");
            fflush(stderr);
        }
    }
    return true;
}

void ggml_cuda_register_moe_expert_cache() {
    ggml_backend_set_moe_expert_cache_cpy(ggml_cuda_moe_expert_cache_cpy);
}
