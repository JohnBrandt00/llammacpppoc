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
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// platform bits for the disk tier (mmap introspection, pinning, readahead,
// positional reads)
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

// cudaMemcpyBatchAsync was added in CUDA 12.8; without it the batched path is
// unavailable and the cache falls back to per-expert copies.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
#define GGML_CUDA_HAS_MEMCPY_BATCH 1
#else
#define GGML_CUDA_HAS_MEMCPY_BATCH 0
#endif

// one expert's slot->input_cpy copy, consumed by the single-kernel gather path
struct moe_gather_desc {
    void *       dst;
    const void * src;
    size_t       size;
};

struct moe_disk_file; // an opened GGUF (disk tier)

// one used expert classified in pass 1, emitted in pass 2 of the hook
struct moe_pass_item {
    int64_t      e;
    bool         hit;   // resident in the VRAM pool
    char *       slot;  // its VRAM slot
    char *       dst;   // its place in input_cpy
    const char * src;   // host bytes: mmap ptr, or a RAM-tier buffer
    size_t       size;
};

// one whole-expert read from the GGUF into a RAM-tier buffer
struct moe_cold_job {
    moe_disk_file * file;
    size_t          off;          // file offset of the expert
    void *          buf;          // destination RAM-tier buffer
    size_t          size;
    const char *    fallback_src; // mmap ptr; used if the read fails
};

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

    // scratch for the single-kernel gather path (Stage A)
    std::vector<moe_gather_desc> gdesc;       // host staging, one entry per used expert
    moe_gather_desc *            d_desc = nullptr; // device copy of gdesc
    size_t                       d_cap  = 0;       // capacity of d_desc, in entries

    // scratch for the two-pass serve (classify, then emit) used by all modes
    std::vector<moe_pass_item> pass;
    std::vector<moe_cold_job>  cold;

    // expert heat: per (tensor name, expert id) access counts, persisted across
    // runs (GGML_MOE_EXPERT_CACHE_HEAT=<file>) and used to pre-warm the pool
    // with the known-hot experts at startup (idea borrowed from colibri's
    // learned .coli_usage pinning). Traffic is highly skewed, so a small warm
    // set removes most cold-start misses.
    std::unordered_map<std::string, std::vector<uint64_t>> heat_counts;
    std::unordered_map<std::string, std::vector<std::pair<uint32_t, uint64_t>>> heat_hot; // loaded, sorted desc by count
    std::unordered_set<std::string> heat_prewarmed; // tensors already pre-warmed this run
    uint64_t heat_prewarm_count = 0;                // experts pre-loaded this run
    bool     heat_loaded        = false;
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

static bool cuda_expert_cache_gather_enabled() {
    // Stage A: serve all hits with a single gather kernel instead of one D2D
    // copy per expert. Opt-in (GGML_MOE_EXPERT_CACHE_GATHER=1); takes precedence
    // over the batched path. This is the substrate for on-device routing.
    static const int enabled = []() {
        const char * env = getenv("GGML_MOE_EXPERT_CACHE_GATHER");
        return env && atoi(env) != 0 ? 1 : 0;
    }();
    return enabled != 0;
}

// Serve N experts slot->input_cpy in one launch: one block per expert, the block
// cooperatively copies that expert's bytes (vectorised 16 B/word when aligned).
static __global__ void moe_gather_kernel(const moe_gather_desc * __restrict__ descs) {
    const moe_gather_desc d = descs[blockIdx.x];
    char *       __restrict__ dp = (char *)       d.dst;
    const char * __restrict__ sp = (const char *) d.src;
    const size_t sz = d.size;

    if ((((uintptr_t) dp | (uintptr_t) sp | sz) & 15u) == 0) {
        const size_t n4 = sz >> 4;
        for (size_t i = threadIdx.x; i < n4; i += blockDim.x) {
            reinterpret_cast<uint4 *>(dp)[i] = reinterpret_cast<const uint4 *>(sp)[i];
        }
    } else {
        for (size_t i = threadIdx.x; i < sz; i += blockDim.x) {
            dp[i] = sp[i];
        }
    }
}

// Upload the staged descriptors and launch the gather (grows device scratch as
// needed). Stream-ordered after any miss host->device loads, so the kernel sees
// freshly populated slots.
static void cuda_expert_cache_gather_launch(cuda_expert_cache & c, cudaStream_t stream) {
    const size_t n = c.gdesc.size();
    if (c.d_cap < n) {
        if (c.d_desc) {
            (void) cudaFree(c.d_desc);
            c.d_desc = nullptr;
        }
        const size_t cap = n + n / 2 + 64;
        if (cudaMalloc(&c.d_desc, cap * sizeof(moe_gather_desc)) != cudaSuccess) {
            (void) cudaGetLastError();
            c.d_cap = 0;
            return;
        }
        c.d_cap = cap;
    }
    CUDA_CHECK(cudaMemcpyAsync(c.d_desc, c.gdesc.data(), n * sizeof(moe_gather_desc),
        cudaMemcpyHostToDevice, stream));
    moe_gather_kernel<<<(unsigned) n, 256, 0, stream>>>(c.d_desc);
    CUDA_CHECK(cudaGetLastError());
}

static cuda_expert_cache g_cuda_expert_cache;
static std::mutex        g_cuda_expert_cache_mutex;

// ============================== disk tier ===================================
// Phase 1+2 of docs/DISK_TIER_DESIGN.md: treat experts as the paging unit so
// MoE models whose experts exceed RAM run from mmap without collapsing into
// blind 4 KB page faults. Opt-in: GGML_MOE_EXPERT_DISK=1 (requires mmap; with
// --no-mmap the source never resolves and behaviour is unchanged).
//
//   tier 1 (hot):  heat-ranked experts mlock'd in RAM at first sight
//                  (GGML_MOE_EXPERT_RAM_BYTES budget; needs a heat file)
//   tier 2 (cold): whole-expert pread() into a bounded RAM LRU
//                  (GGML_MOE_EXPERT_DISK_CACHE_BYTES, default 2 GiB) by a
//                  small thread pool, then H2D from that buffer
//
// Tensors that bypass the VRAM cache (mismatched expert size) still get
// Phase-1 treatment: pinning + whole-expert readahead advice, so the stock
// grouped copy reads sequential MBs instead of faulting page by page.
// File offsets are discovered from the mapping itself (/proc/self/maps on
// Linux, VirtualQuery+GetMappedFileName on Windows) -- no loader changes.

static bool cuda_expert_disk_enabled() {
    static const int enabled = []() {
        const char * env = getenv("GGML_MOE_EXPERT_DISK");
        return env && atoi(env) != 0 ? 1 : 0;
    }();
    return enabled != 0;
}

// mlock budget for the hot tier; 0 (default) disables pinning
static size_t cuda_expert_disk_pin_bytes() {
    static const size_t v = []() -> size_t {
        const char * env = getenv("GGML_MOE_EXPERT_RAM_BYTES");
        return env != nullptr ? (size_t) std::max(0LL, atoll(env)) : 0;
    }();
    return v;
}

// RAM LRU budget for streamed cold experts
static size_t cuda_expert_disk_cache_bytes() {
    static const size_t v = []() -> size_t {
        const char * env = getenv("GGML_MOE_EXPERT_DISK_CACHE_BYTES");
        const long long b = env != nullptr ? atoll(env) : (2LL << 30);
        return (size_t) std::max(b, 256LL << 20); // floor: op working set must fit
    }();
    return v;
}

static int cuda_expert_disk_threads() {
    static const int v = []() {
        const char * env = getenv("GGML_MOE_EXPERT_DISK_THREADS");
        const int n = env != nullptr ? atoi(env) : 4;
        return std::min(std::max(n, 1), 16);
    }();
    return v;
}

static size_t moe_page_size() {
#ifdef _WIN32
    static const size_t ps = []() { SYSTEM_INFO si; GetSystemInfo(&si); return (size_t) si.dwPageSize; }();
#else
    static const size_t ps = (size_t) sysconf(_SC_PAGESIZE);
#endif
    return ps;
}

// whole-expert readahead advice (page-aligned)
static void moe_advise_willneed(const void * p, size_t n) {
    const size_t    ps = moe_page_size();
    const uintptr_t a  = (uintptr_t) p & ~(uintptr_t) (ps - 1);
    const size_t    len = (((uintptr_t) p + n + ps - 1) & ~(uintptr_t) (ps - 1)) - a;
#ifdef _WIN32
    WIN32_MEMORY_RANGE_ENTRY r;
    r.VirtualAddress = (PVOID) a;
    r.NumberOfBytes  = len;
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &r, 0);
#else
    madvise((void *) a, len, MADV_WILLNEED);
#endif
}

// pin a range in RAM (mlock/VirtualLock); faults it in as a side effect
static bool moe_lock_range(const void * p, size_t n) {
    const size_t    ps = moe_page_size();
    const uintptr_t a  = (uintptr_t) p & ~(uintptr_t) (ps - 1);
    const size_t    len = (((uintptr_t) p + n + ps - 1) & ~(uintptr_t) (ps - 1)) - a;
#ifdef _WIN32
    return VirtualLock((LPVOID) a, len) != 0;
#else
    return mlock((const void *) a, len) == 0;
#endif
}

struct moe_disk_file {
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
};

// positional whole-expert read; thread-safe (no shared file pointer)
static bool moe_disk_read(moe_disk_file * f, size_t off, void * buf, size_t n) {
#ifdef _WIN32
    char * p = (char *) buf;
    while (n > 0) {
        OVERLAPPED ov = {};
        ov.Offset     = (DWORD) (off & 0xffffffffu);
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD got = 0;
        if (!ReadFile(f->h, p, (DWORD) std::min<size_t>(n, 1u << 30), &got, &ov) || got == 0) {
            return false;
        }
        p += got; off += got; n -= got;
    }
    return true;
#else
    char * p = (char *) buf;
    while (n > 0) {
        const ssize_t got = pread(f->fd, p, n, (off_t) off);
        if (got <= 0) {
            return false;
        }
        p += got; off += (size_t) got; n -= (size_t) got;
    }
    return true;
#endif
}

struct moe_disk_tensor {
    bool   tried    = false;
    bool   resolved = false;
    moe_disk_file * file = nullptr;  // owned by moe_disk_state::files
    size_t file_off = 0;             // file offset of the tensor base (src->data)
    std::vector<uint8_t> pinned;     // per-expert: 1 = mlock'd hot-tier resident
};

struct moe_disk_state {
    std::unordered_map<const void *, moe_disk_tensor> tensors; // key: src->data
    std::unordered_map<std::string, moe_disk_file>    files;   // dedup opened GGUFs

    // RAM LRU for streamed cold experts
    struct ram_ent {
        void * buf;
        size_t size;
        std::list<uint64_t>::iterator it;
    };
    std::unordered_map<uint64_t, ram_ent> ram;      // key: expert host address
    std::list<uint64_t>                   ram_lru;  // front = most recent
    size_t                                ram_bytes = 0;

    // evicted buffers may still be the source of in-flight async H2D copies;
    // they are freed only after an event recorded behind those copies fires
    std::vector<std::pair<void *, cudaEvent_t>> quarantine;

    bool     pin_failed   = false; // hit RLIMIT_MEMLOCK etc.; stop trying
    uint64_t pinned_bytes = 0;
    int      pinned_tensors = 0;
    uint64_t n_pin_hit = 0, n_ram_hit = 0, n_pread = 0, pread_bytes = 0, n_read_fail = 0;
};
static moe_disk_state g_moe_disk;

// find the file-backed mapping containing p -> opened file + file offset of p
static bool moe_disk_resolve(const void * p, moe_disk_file *& file_out, size_t & off_out) {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(p, &mbi, sizeof(mbi)) == 0 || mbi.Type != MEM_MAPPED) {
        return false;
    }
    wchar_t dev[1024];
    if (K32GetMappedFileNameW(GetCurrentProcess(), (LPVOID) p, dev, 1024) == 0) {
        return false;
    }
    // \Device\HarddiskVolumeN\... -> DOS drive path
    std::wstring wpath;
    wchar_t drive[3] = L"A:";
    for (wchar_t c = L'A'; c <= L'Z'; ++c) {
        drive[0] = c;
        wchar_t target[512];
        if (QueryDosDeviceW(drive, target, 512) == 0) {
            continue;
        }
        const size_t tlen = wcslen(target);
        if (wcsncmp(dev, target, tlen) == 0 && dev[tlen] == L'\\') {
            wpath = std::wstring(drive) + std::wstring(dev + tlen);
            break;
        }
    }
    if (wpath.empty()) {
        return false;
    }
    // llama.cpp maps the whole file from offset 0, so AllocationBase is offset 0
    off_out = (size_t) ((const char *) p - (const char *) mbi.AllocationBase);
    std::string key(wpath.begin(), wpath.end()); // key only; open uses the wide path
    auto it = g_moe_disk.files.find(key);
    if (it == g_moe_disk.files.end()) {
        moe_disk_file f;
        f.h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f.h == INVALID_HANDLE_VALUE) {
            return false;
        }
        it = g_moe_disk.files.emplace(key, f).first;
    }
    file_out = &it->second;
    return true;
#else
    FILE * maps = fopen("/proc/self/maps", "r");
    if (maps == nullptr) {
        return false;
    }
    const uintptr_t addr = (uintptr_t) p;
    char line[1024];
    bool found = false;
    uintptr_t start = 0;
    size_t map_off = 0;
    std::string path;
    while (fgets(line, sizeof(line), maps) != nullptr) {
        uintptr_t s, e;
        unsigned long long off;
        char perms[8];
        int consumed = 0;
        if (sscanf(line, "%zx-%zx %7s %llx %*s %*s %n", &s, &e, perms, &off, &consumed) < 4) {
            continue;
        }
        if (addr < s || addr >= e) {
            continue;
        }
        const char * rest = line + consumed;
        while (*rest == ' ' || *rest == '\t') {
            rest++;
        }
        if (rest[0] != '/') {
            break; // anonymous mapping (--no-mmap): no disk source
        }
        path.assign(rest);
        while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) {
            path.pop_back();
        }
        start = s; map_off = (size_t) off; found = true;
        break;
    }
    fclose(maps);
    if (!found || path.empty()) {
        return false;
    }
    off_out = map_off + (size_t) (addr - start);
    auto it = g_moe_disk.files.find(path);
    if (it == g_moe_disk.files.end()) {
        moe_disk_file f;
        f.fd = open(path.c_str(), O_RDONLY);
        if (f.fd < 0) {
            return false;
        }
        it = g_moe_disk.files.emplace(path, f).first;
    }
    file_out = &it->second;
    return true;
#endif
}

// ---- tiny persistent I/O pool for parallel whole-expert reads ----
struct moe_io_pool {
    std::vector<std::thread>    workers;
    std::deque<moe_cold_job *>  q;
    std::mutex                  m;
    std::condition_variable     cv;
    std::condition_variable     cv_done;
    size_t                      pending = 0;
};
static moe_io_pool g_moe_io;

static void moe_disk_run_job(moe_cold_job & j) {
    if (!moe_disk_read(j.file, j.off, j.buf, j.size)) {
        // fall back to the mapped pointer (page faults, but stays correct)
        memcpy(j.buf, j.fallback_src, j.size);
        std::lock_guard<std::mutex> lock(g_moe_io.m);
        g_moe_disk.n_read_fail++;
    }
}

static void moe_io_worker() {
    for (;;) {
        moe_cold_job * j = nullptr;
        {
            std::unique_lock<std::mutex> lock(g_moe_io.m);
            g_moe_io.cv.wait(lock, [] { return !g_moe_io.q.empty(); });
            j = g_moe_io.q.front();
            g_moe_io.q.pop_front();
        }
        moe_disk_run_job(*j);
        {
            std::lock_guard<std::mutex> lock(g_moe_io.m);
            if (--g_moe_io.pending == 0) {
                g_moe_io.cv_done.notify_all();
            }
        }
    }
}

// run all reads (parallel when >1), return when every buffer is filled
static void moe_disk_read_jobs(std::vector<moe_cold_job> & jobs) {
    if (jobs.empty()) {
        return;
    }
    if (jobs.size() == 1) {
        moe_disk_run_job(jobs[0]);
        return;
    }
    if (g_moe_io.workers.empty()) {
        for (int i = 0; i < cuda_expert_disk_threads(); ++i) {
            g_moe_io.workers.emplace_back(moe_io_worker);
            g_moe_io.workers.back().detach();
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_moe_io.m);
        for (auto & j : jobs) {
            g_moe_io.q.push_back(&j);
        }
        g_moe_io.pending += jobs.size();
    }
    g_moe_io.cv.notify_all();
    std::unique_lock<std::mutex> lock(g_moe_io.m);
    g_moe_io.cv_done.wait(lock, [] { return g_moe_io.pending == 0; });
}

// ---- RAM LRU for streamed cold experts ----

// free quarantined buffers whose guarding event has fired
static void moe_disk_quarantine_poll() {
    auto & q = g_moe_disk.quarantine;
    for (size_t i = 0; i < q.size();) {
        if (cudaEventQuery(q[i].second) == cudaSuccess) {
            (void) cudaEventDestroy(q[i].second);
            free(q[i].first);
            q[i] = q.back();
            q.pop_back();
        } else {
            (void) cudaGetLastError(); // clear cudaErrorNotReady
            ++i;
        }
    }
}

static void * moe_disk_ram_get(uint64_t key) {
    auto it = g_moe_disk.ram.find(key);
    if (it == g_moe_disk.ram.end()) {
        return nullptr;
    }
    g_moe_disk.ram_lru.erase(it->second.it);
    g_moe_disk.ram_lru.push_front(key);
    it->second.it = g_moe_disk.ram_lru.begin();
    return it->second.buf;
}

static void * moe_disk_ram_insert(uint64_t key, size_t size, cudaStream_t stream) {
    // evict LRU tail until the new buffer fits; evicted buffers go to the
    // quarantine behind a stream event (in-flight copies may still read them)
    const size_t cap = cuda_expert_disk_cache_bytes();
    while (g_moe_disk.ram_bytes + size > cap && !g_moe_disk.ram_lru.empty()) {
        const uint64_t victim = g_moe_disk.ram_lru.back();
        auto vit = g_moe_disk.ram.find(victim);
        cudaEvent_t ev = nullptr;
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess &&
            cudaEventRecord(ev, stream) == cudaSuccess) {
            g_moe_disk.quarantine.push_back({ vit->second.buf, ev });
        } else {
            (void) cudaGetLastError();
            free(vit->second.buf); // no event: free immediately (init-time only)
        }
        g_moe_disk.ram_bytes -= vit->second.size;
        g_moe_disk.ram_lru.pop_back();
        g_moe_disk.ram.erase(vit);
    }
    void * buf = malloc(size);
    if (buf == nullptr) {
        return nullptr;
    }
    g_moe_disk.ram_lru.push_front(key);
    g_moe_disk.ram.insert({ key, { buf, size, g_moe_disk.ram_lru.begin() } });
    g_moe_disk.ram_bytes += size;
    return buf;
}

// forward decl (heat structures live on cuda_expert_cache, defined below)
static void moe_disk_pin_tensor(cuda_expert_cache & c, moe_disk_tensor & dt, const char * name,
        const char * base, size_t expert_size, int64_t n_expert);

// first-sight resolution + hot-tier pinning for one expert tensor
static moe_disk_tensor * moe_disk_tensor_get(cuda_expert_cache & c, const struct ggml_tensor * src,
        size_t expert_size, int64_t n_expert) {
    moe_disk_tensor & dt = g_moe_disk.tensors[src->data];
    if (!dt.tried) {
        dt.tried = true;
        moe_disk_file * file = nullptr;
        size_t off = 0;
        if (moe_disk_resolve(src->data, file, off)) {
            dt.resolved = true;
            dt.file     = file;
            dt.file_off = off;
        } else {
            fprintf(stderr, "ggml_moe_expert_disk: '%s' has no file-backed mapping (using --no-mmap?); tier inactive for it\n",
                src->name);
        }
        moe_disk_pin_tensor(c, dt, src->name, (const char *) src->data, expert_size, n_expert);
    }
    return &dt;
}

// ============================ end disk tier =================================

static const char * cuda_expert_cache_heat_path() {
    static const char * path = getenv("GGML_MOE_EXPERT_CACHE_HEAT");
    return path; // not set = feature off
}

// Persist the accumulated heat on process exit so the next run can pre-warm.
// Counts are cumulative across runs (loaded counts are merged on load).
static void cuda_expert_cache_heat_save() {
    const char * path = cuda_expert_cache_heat_path();
    if (path == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_cuda_expert_cache_mutex);
    cuda_expert_cache & c = g_cuda_expert_cache;
    FILE * f = fopen(path, "w");
    if (f == nullptr) {
        fprintf(stderr, "ggml_moe_expert_cache: cannot write heat file '%s'\n", path);
        return;
    }
    uint64_t entries = 0;
    for (const auto & kv : c.heat_counts) {
        for (uint32_t e = 0; e < kv.second.size(); ++e) {
            if (kv.second[e] != 0) {
                fprintf(f, "%s %u %llu\n", kv.first.c_str(), e, (unsigned long long) kv.second[e]);
                entries++;
            }
        }
    }
    fclose(f);
    const uint64_t total = c.hits + c.misses;
    fprintf(stderr, "ggml_moe_expert_cache: heat saved to '%s' (%llu entries); "
        "run: accesses=%llu hit_rate=%.1f%% prewarmed=%llu\n",
        path, (unsigned long long) entries, (unsigned long long) total,
        total ? 100.0 * (double) c.hits / (double) total : 0.0,
        (unsigned long long) c.heat_prewarm_count);
    if (cuda_expert_disk_enabled()) {
        fprintf(stderr, "ggml_moe_expert_disk: run: pin_hit=%llu ram_hit=%llu pread=%llu pread_GiB=%.2f read_fail=%llu pinned=%.2f GiB/%d tensors\n",
            (unsigned long long) g_moe_disk.n_pin_hit,
            (unsigned long long) g_moe_disk.n_ram_hit,
            (unsigned long long) g_moe_disk.n_pread,
            (double) g_moe_disk.pread_bytes / (1024.0 * 1024.0 * 1024.0),
            (unsigned long long) g_moe_disk.n_read_fail,
            (double) g_moe_disk.pinned_bytes / (1024.0 * 1024.0 * 1024.0),
            g_moe_disk.pinned_tensors);
    }
    fflush(stderr);
}

// Load the heat file once and build the per-tensor hot lists for pre-warming.
static void cuda_expert_cache_heat_load(cuda_expert_cache & c) {
    c.heat_loaded = true;
    const char * path = cuda_expert_cache_heat_path();
    if (path == nullptr) {
        return;
    }
    atexit(cuda_expert_cache_heat_save);
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        fprintf(stderr, "ggml_moe_expert_cache: heat file '%s' not found; tracking, will create on exit\n", path);
        return;
    }
    char name[256];
    unsigned e;
    unsigned long long n;
    while (fscanf(f, "%255s %u %llu", name, &e, &n) == 3) {
        auto & v = c.heat_counts[name];
        if (v.size() <= e) {
            v.resize(e + 1, 0);
        }
        v[e] += n;
    }
    fclose(f);
    for (const auto & kv : c.heat_counts) {
        auto & hot = c.heat_hot[kv.first];
        for (uint32_t i = 0; i < kv.second.size(); ++i) {
            if (kv.second[i] != 0) {
                hot.push_back({i, kv.second[i]});
            }
        }
        std::sort(hot.begin(), hot.end(),
            [](const std::pair<uint32_t, uint64_t> & a, const std::pair<uint32_t, uint64_t> & b) {
                return a.second > b.second;
            });
    }
    fprintf(stderr, "ggml_moe_expert_cache: heat loaded from '%s' (%zu tensors) -> pre-warm enabled\n",
        path, c.heat_hot.size());
    fflush(stderr);
}

// Pin this tensor's hottest experts in RAM (hot tier). The mlock budget is
// divided evenly between the tensors in the heat file; locking faults the
// pages in, so startup pays one sequential read of the hot set. On the first
// mlock failure (usually RLIMIT_MEMLOCK) pinning is disabled globally and the
// tier degrades to readahead + the cold RAM LRU.
static void moe_disk_pin_tensor(cuda_expert_cache & c, moe_disk_tensor & dt, const char * name,
        const char * base, size_t expert_size, int64_t n_expert) {
    dt.pinned.assign((size_t) n_expert, 0);
    const size_t budget = cuda_expert_disk_pin_bytes();
    if (!dt.resolved || budget == 0 || g_moe_disk.pin_failed || c.heat_hot.empty()) {
        return;
    }
    const auto it = c.heat_hot.find(name);
    if (it == c.heat_hot.end() || it->second.empty()) {
        return;
    }
    const size_t share = budget / c.heat_hot.size();
    size_t used = 0;
    for (const auto & he : it->second) {
        const uint32_t e = he.first;
        if ((int64_t) e >= n_expert) {
            continue;
        }
        if (used + expert_size > share) {
            break;
        }
        if (!moe_lock_range(base + (size_t) e * expert_size, expert_size)) {
            g_moe_disk.pin_failed = true;
            fprintf(stderr, "ggml_moe_expert_disk: mlock failed (RLIMIT_MEMLOCK/CAP_IPC_LOCK?); "
                "pinning disabled, falling back to readahead + cold cache\n");
            fflush(stderr);
            break;
        }
        dt.pinned[e] = 1;
        used += expert_size;
    }
    if (used > 0) {
        g_moe_disk.pinned_bytes += used;
        g_moe_disk.pinned_tensors++;
    }
}

// Pre-load this tensor's hottest experts into their slots (one-time, first
// access). The pool is divided evenly between the tensors in the heat file so
// pre-warming can't evict another tensor's warm set. In disk mode only pinned
// (RAM-resident) experts are pre-warmed -- warming cold ones would stall
// startup on disk reads.
static void cuda_expert_cache_prewarm(cuda_expert_cache & c, const char * name,
        const char * src_base, size_t expert_size, int64_t n_expert, cudaStream_t stream,
        const moe_disk_tensor * dt) {
    const auto it = c.heat_hot.find(name);
    if (it == c.heat_hot.end() || it->second.empty()) {
        return;
    }
    const size_t budget = std::max<size_t>(1, (size_t) c.n_slots / c.heat_hot.size());
    const size_t pad    = expert_size < 512 ? expert_size : 512;
    const size_t n      = std::min(budget, it->second.size());
    for (size_t i = 0; i < n; ++i) {
        const uint32_t e = it->second[i].first;
        if ((int64_t) e >= n_expert) {
            continue; // heat file from a different model
        }
        if (dt != nullptr && dt->resolved && !(e < dt->pinned.size() && dt->pinned[e] != 0)) {
            continue; // disk mode: this expert is cold, don't fault it in now
        }
        const char * host_src = src_base + (size_t) e * expert_size;
        const expert_cache_lru::access_result r = c.lru->access((uint64_t) (uintptr_t) host_src);
        if (r.hit) {
            continue; // already resident
        }
        char * const slot = (char *) c.pool + (size_t) r.slot * c.slot_size;
        const size_t copy_size = expert_size + ((int64_t) e < n_expert - 1 ? pad : 0);
        CUDA_CHECK(cudaMemcpyAsync(slot, host_src, copy_size, cudaMemcpyHostToDevice, stream));
        c.heat_prewarm_count++;
    }
}

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

    // heat + disk setup run BEFORE the size/device guard so tensors that
    // bypass the VRAM cache (mismatched expert size) still get traffic
    // counts, hot-tier pinning and whole-expert readahead
    const bool heat_on = cuda_expert_cache_heat_path() != nullptr;
    if (heat_on && !c.heat_loaded) {
        cuda_expert_cache_heat_load(c);
    }
    std::vector<uint64_t> * heat_vec = nullptr;
    if (heat_on) {
        auto & v = c.heat_counts[src->name];
        if ((int64_t) v.size() < n_expert) {
            v.resize(n_expert, 0);
        }
        heat_vec = &v;
    }
    moe_disk_tensor * dt = nullptr;
    if (cuda_expert_disk_enabled()) {
        moe_disk_quarantine_poll();
        dt = moe_disk_tensor_get(c, src, expert_size, n_expert);
        if (!dt->resolved) {
            dt = nullptr; // anonymous memory (--no-mmap): plain RAM behaviour
        }
    }

    const size_t pad = expert_size < 512 ? expert_size : 512;
    const char * const src_base = (const char *) src->data;

    // the pool is sized for one expert size on one device; anything else falls
    // back to the normal grouped host->device copy -- but with Phase-1 disk
    // treatment first, so the stock copy reads sequential whole experts
    // instead of faulting page by page
    if (expert_size != c.expert_size || ctx->device != c.device) {
        for (int64_t e = 0; e < n_expert; ++e) {
            if (!((used_ids[e >> 5] >> (e & 31)) & 1u)) {
                continue;
            }
            if (heat_vec != nullptr) {
                (*heat_vec)[e]++;
            }
            if (dt != nullptr && !((size_t) e < dt->pinned.size() && dt->pinned[e] != 0)) {
                moe_advise_willneed(src_base + (size_t) e * expert_size,
                    expert_size + (e < n_expert - 1 ? pad : 0));
            }
        }
        return false;
    }

    const bool   gather   = cuda_expert_cache_gather_enabled();
    const bool   batch    = !gather && cuda_expert_cache_batch_enabled() && GGML_CUDA_HAS_MEMCPY_BATCH;
    const bool   stats    = cuda_expert_cache_stats_enabled();
    char * const dst_base = (char *) dst->data;

    if (heat_on && c.heat_prewarmed.insert(src->name).second) {
        cuda_expert_cache_prewarm(c, src->name, src_base, expert_size, n_expert, stream, dt);
    }

    if (batch) {
        c.serve_dst.clear(); c.serve_src.clear(); c.serve_size.clear();
        c.pop_dst.clear();   c.pop_src.clear();   c.pop_size.clear();
    }
    if (gather) {
        c.gdesc.clear();
    }

    // pass 1: LRU access + tier classification; VRAM misses resolve their
    // host source (pinned mmap / RAM LRU / whole-expert read from disk)
    c.pass.clear();
    c.cold.clear();
    for (int64_t e = 0; e < n_expert; ++e) {
        if (!((used_ids[e >> 5] >> (e & 31)) & 1u)) {
            continue;
        }
        const size_t       off       = (size_t) e * expert_size;
        const size_t       copy_size = expert_size + (e < n_expert - 1 ? pad : 0);
        const char * const host_src  = src_base + off;
        char * const       cpy_dst   = dst_base + off;

        // key = the expert's unique, stable host address
        const expert_cache_lru::access_result r = c.lru->access((uint64_t) (uintptr_t) host_src);
        char * const slot = (char *) c.pool + (size_t) r.slot * c.slot_size;

        moe_pass_item item = { e, r.hit, slot, cpy_dst, host_src, copy_size };
        if (!r.hit && dt != nullptr) {
            if ((size_t) e < dt->pinned.size() && dt->pinned[e] != 0) {
                g_moe_disk.n_pin_hit++; // hot tier: mmap ptr is locked resident
            } else {
                const uint64_t key = (uint64_t) (uintptr_t) host_src;
                void * buf = moe_disk_ram_get(key);
                if (buf != nullptr) {
                    g_moe_disk.n_ram_hit++;
                    item.src = (const char *) buf;
                } else if ((buf = moe_disk_ram_insert(key, copy_size, stream)) != nullptr) {
                    item.src = (const char *) buf;
                    c.cold.push_back({ dt->file, dt->file_off + off, buf, copy_size, host_src });
                }
                // else: allocation failed -> mmap pointer (faults, stays correct)
            }
        }

        if (stats) {
            if (r.hit) { c.hits++; } else { c.misses++; c.miss_bytes += copy_size; }
        }
        if (heat_vec != nullptr) {
            (*heat_vec)[e]++;
        }
        c.pass.push_back(item);
    }

    // stream the cold experts from disk in parallel before any copy needs them
    if (!c.cold.empty()) {
        moe_disk_read_jobs(c.cold);
        g_moe_disk.n_pread     += c.cold.size();
        for (const auto & j : c.cold) {
            g_moe_disk.pread_bytes += j.size;
        }
    }

    // pass 2: emit the copies (identical semantics to the old single loop;
    // only the miss source pointer may now be a RAM-tier buffer)
    for (const moe_pass_item & it : c.pass) {
        if (gather) {
            // load misses into their slot now (rare), then serve every used
            // expert from its slot in a single gather kernel after the loop
            if (!it.hit) {
                CUDA_CHECK(cudaMemcpyAsync(it.slot, it.src, it.size, cudaMemcpyHostToDevice, stream));
            }
            c.gdesc.push_back({ it.dst, it.slot, it.size });
        } else if (batch) {
            // serve every expert from its slot (D2D); misses are loaded into
            // their slot first by the populate batch, which runs before serve
            c.serve_dst.push_back(it.dst);
            c.serve_src.push_back(it.slot);
            c.serve_size.push_back(it.size);
            if (!it.hit) {
                c.pop_dst.push_back(it.slot);
                c.pop_src.push_back(it.src);
                c.pop_size.push_back(it.size);
            }
        } else {
            if (!it.hit) {
                CUDA_CHECK(cudaMemcpyAsync(it.slot, it.src, it.size, cudaMemcpyHostToDevice, stream));
            }
            CUDA_CHECK(cudaMemcpyAsync(it.dst, it.slot, it.size, cudaMemcpyDeviceToDevice, stream));
        }
    }

    if (gather && !c.gdesc.empty()) {
        cuda_expert_cache_gather_launch(c, stream);
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
            fprintf(stderr, "ggml_moe_expert_cache_stats: accesses=%llu hit_rate=%.1f%% miss_GiB=%.2f mode=%s prewarm=%llu\n",
                (unsigned long long) total, 100.0 * (double) c.hits / (double) total,
                (double) c.miss_bytes / (1024.0 * 1024.0 * 1024.0),
                gather ? "gather" : (batch ? "batch" : "per-expert"),
                (unsigned long long) c.heat_prewarm_count);
            if (cuda_expert_disk_enabled()) {
                fprintf(stderr, "ggml_moe_expert_disk_stats: pin_hit=%llu ram_hit=%llu pread=%llu pread_GiB=%.2f read_fail=%llu pinned=%.2f GiB/%d tensors cold_cache=%.0f MiB\n",
                    (unsigned long long) g_moe_disk.n_pin_hit,
                    (unsigned long long) g_moe_disk.n_ram_hit,
                    (unsigned long long) g_moe_disk.n_pread,
                    (double) g_moe_disk.pread_bytes / (1024.0 * 1024.0 * 1024.0),
                    (unsigned long long) g_moe_disk.n_read_fail,
                    (double) g_moe_disk.pinned_bytes / (1024.0 * 1024.0 * 1024.0),
                    g_moe_disk.pinned_tensors,
                    (double) g_moe_disk.ram_bytes / (1024.0 * 1024.0));
            }
            fflush(stderr);
        }
    }
    return true;
}

void ggml_cuda_register_moe_expert_cache() {
    ggml_backend_set_moe_expert_cache_cpy(ggml_cuda_moe_expert_cache_cpy);
}
