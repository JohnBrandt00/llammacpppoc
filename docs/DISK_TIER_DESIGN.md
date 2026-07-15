# Disk tier design — running MoE models bigger than RAM

Goal: run models whose **expert weights exceed system RAM** (GLM-4.5-Air,
58–78 GB, on guanshiyin's ~40 GB usable) at usable speed. Today those models
either OOM (`--no-mmap`) or crawl (mmap: 1.47 t/s prefill, measured) because
the OS demand-pages random 4 KB chunks with no idea what an expert is.

Idea adopted from JustVugg/colibri (744B GLM from disk in ~25 GB RAM): treat
experts as the paging unit. Three tiers:

```
   VRAM slot pool  (LRU + heat pre-warm)          <- exists, validated
        ^ H2D on VRAM-miss
   RAM hot set     (heat-ranked, pinned)          <- NEW
        ^ whole-expert read on RAM-miss
   Disk (the GGUF) (cold tail, expert-granular)   <- NEW
```

## Why this works: the arithmetic

- MoE traffic is highly skewed (measured on Qwen3-30B: top-50% of experts take
  ~91% of activations; colibri reports the same shape on GLM).
- Guanshiyin budget: ~30 GB of RAM for the hot expert set + dense layers and KV
  on the P40 (24 GB VRAM, `-ngl 999`), VRAM pool caching the hottest of the hot.
- A cold expert is a **single sequential ~1–2 MB read** (~0.5–1 ms on NVMe),
  not ~400 scattered 4 KB faults. At ~90% RAM-tier hit and top-8 routing over
  ~46 layers, decode pays ~30–40 cold reads/token worst case — a few ms/token
  of overlappable I/O -> **single-digit t/s for a model that cannot load today**.
  (Colibri sustains 0.4–2 t/s streaming a 744B from disk; our target model is
  5× smaller with a GPU cache on top.)

## Phase 1 — mmap-assisted (small, testable, no new I/O machinery)

Keep mmap (so the address space works for model > RAM) but stop letting the OS
guess. Two mechanisms, both inside the existing cache hook (it already sees
every used expert id + host address before the copy):

1. **Heat-ranked RAM pinning at load.** Rank experts by the persisted heat
   file; `mlock()` the hot set's address ranges up to a budget
   (`GGML_MOE_EXPERT_RAM_BYTES`, default: MemAvailable minus headroom). Hot
   experts then physically cannot be evicted; only the cold tail ever faults.
   (Linux: needs `ulimit -l` / CAP_IPC_LOCK — check LXC config. Windows dev
   box: `VirtualLock` + `SetProcessWorkingSetSize`.)

2. **Batched whole-expert readahead on miss.** When the hook sees the op's
   used experts, issue `madvise(MADV_WILLNEED)` on every non-resident expert's
   full range *before* the copies. The kernel then does parallel sequential
   readahead of whole experts instead of serial random faults. (Windows:
   `PrefetchVirtualMemory`.)

Pre-warm interaction: at startup, pre-warm the VRAM pool only from *pinned*
(RAM-resident) experts so startup doesn't fault the cold tail in from disk.

Phase 1 success gate: GLM-4.5-Air quant that OOMs today loads under mmap and
decodes ≥ 2–3 t/s on guanshiyin with sane prefill (≥ 20 t/s). If the LXC
forbids mlock or the page cache fights us, move to Phase 2.

## Phase 2 — explicit expert pager (full control, colibri-equivalent)

Replace reliance on the page cache with our own machinery:

- A **RAM slab** of expert slots (like the VRAM pool, one tier down): pinned
  hot section (heat-ranked) + LRU section for streamed cold experts.
- **Own I/O path**: `pread` whole experts from the GGUF (tensor file offsets
  exposed from the loader), issued by a small thread pool ahead of the compute
  stream; the hook's miss path becomes RAM-slot -> H2D.
- Removes OS unpredictability (eviction policy, readahead heuristics, mlock
  limits) at the cost of loader surgery (file-offset plumbing into the hook).

## Phase 3 — router-lookahead prefetch (only if measurement demands it)

We killed predictors for the VRAM tier because LRU was within ~8% of optimal
*at microsecond copy costs*. Disk misses cost **milliseconds**, which changes
the calculus: colibri measures next-layer routing ~71.6% predictable, enough to
overlap most cold reads one layer ahead. Build only if Phase 1/2 profiling
shows cold-read stalls dominating decode. We already have the routing-trace
tooling (`GGML_MOE_ROUTING_TRACE` + analyzers) to measure predictability on
GLM before writing any code.

## Risks / open questions

- **LXC mlock limits** (RLIMIT_MEMLOCK) may block Phase 1 pinning — detect and
  fall back to plain WILLNEED readahead (still fixes the 4 KB-fault disaster).
- **Heat coverage for a new model**: first GLM run has no heat file; Phase 1
  degrades to readahead-only until one accumulate-run completes.
- **Mixed expert sizes**: Q4_K_M-style quants give `down_exps` a different
  expert_size; today those tensors bypass the VRAM cache (measured: 120/144
  cached). The RAM tier must handle multiple sizes from day one (per-size
  budgets), and a multi-size VRAM pool is a follow-up.
- **Storage speed** on the LXC (VHDX/virtio?) — microbench expert-sized preads
  before trusting the arithmetic above.
- **Windows dev box parity**: APIs exist (`PrefetchVirtualMemory`/`VirtualLock`)
  but primary target is the Linux P40 box; Windows is build-and-losslessness
  only for Phase 1.

## Build order

1. Phase 1a: WILLNEED batched readahead on miss (smallest useful piece; helps
   the measured 1.47 t/s mmap disaster immediately, no privileges needed).
2. Phase 1b: heat-ranked mlock pinning + resident-only pre-warm.
3. Gate on guanshiyin GLM-4.5-Air measurements -> Phase 2 (explicit pager)
   and/or Phase 3 (lookahead) only as the data demands.

Every phase keeps the invariant that has held since Stage 0: **bytes reaching
the GEMM are identical** — the tiers change where weights live and how they
travel, never the math.
