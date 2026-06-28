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

struct cuda_expert_cache {
    void *  pool        = nullptr;
    size_t  expert_size = 0;   // bytes per expert (no padding)
    size_t  slot_size   = 0;   // expert_size + read-ahead padding
    int     n_slots     = 0;
    int     device      = -1;
    bool    failed      = false;
    expert_cache_lru * lru = nullptr;
};

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
        ggml_backend_t backend, struct ggml_tensor * dst, size_t dst_offset,
        const void * src, size_t size, bool last) {
    // only handle the CUDA backend; anything else falls back to a normal copy
    if (!ggml_backend_is_cuda(backend)) {
        return false;
    }

    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    cudaStream_t stream = ctx->stream();

    std::lock_guard<std::mutex> lock(g_cuda_expert_cache_mutex);
    cuda_expert_cache & c = g_cuda_expert_cache;

    if (c.pool == nullptr) {
        if (c.failed || !cuda_expert_cache_init(c, ctx->device, size)) {
            return false;
        }
    }

    // the pool is sized for one expert size on one device; anything else falls
    // back to a normal host->device copy
    if (size != c.expert_size || ctx->device != c.device) {
        return false;
    }

    const size_t padding   = last ? 0 : (size < 512 ? size : 512);
    const size_t copy_size = size + padding;

    // each (layer, projection, expert) weight has a unique, stable host address
    const uint64_t key = (uint64_t) (uintptr_t) src;
    const expert_cache_lru::access_result r = c.lru->access(key);
    char * slot = (char *) c.pool + (size_t) r.slot * c.slot_size;

    if (!r.hit) {
        // miss: load the expert (plus read-ahead padding) into its slot once
        CUDA_CHECK(cudaMemcpyAsync(slot, src, copy_size, cudaMemcpyHostToDevice, stream));
    }
    // serve the resident expert into input_cpy at its natural offset
    CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + dst_offset, slot, copy_size, cudaMemcpyDeviceToDevice, stream));
    return true;
}

void ggml_cuda_register_moe_expert_cache() {
    ggml_backend_set_moe_expert_cache_cpy(ggml_cuda_moe_expert_cache_cpy);
}
