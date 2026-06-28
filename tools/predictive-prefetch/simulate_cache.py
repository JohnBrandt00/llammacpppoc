#!/usr/bin/env python3
"""Replay MoE routing traces through a per-layer VRAM expert cache and report the
realized *online* hit rate for several eviction policies.

This is the design decision before writing the CUDA cache: which policy actually
delivers, and how close does it get to the oracle/Belady ceiling at the VRAM
budget. Each layer is an independent cache of `M` expert slots (M = budget *
n_experts). A token "accesses" its routed experts; a resident expert is a hit, a
miss loads it (evicting per policy when full).

Policies:
  lru     - evict least recently used
  lfu     - evict least frequently used (ties -> least recently used)
  lru2    - LRU-K with K=2 (evict by oldest 2nd-most-recent access)
  static  - precompute top-M by frequency from a warmup prefix, never evict
  belady  - evict the expert whose next use is furthest in the future (optimal)
"""
import argparse
from collections import defaultdict, Counter
from pathlib import Path


def load_trace(path: Path):
    per_layer = defaultdict(list)
    n_exp = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            try:
                layer = int(p[0]); experts = [int(x) for x in p[1:]]
            except ValueError:
                continue
            per_layer[layer].append(experts)
            if experts:
                n_exp = max(n_exp, max(experts) + 1)
    return per_layer, n_exp


def accesses(seq):
    """Flatten a layer's token->experts list into a time-ordered access stream."""
    for experts in seq:
        for e in experts:
            yield e


def sim_lru(seq, M):
    cache = {}            # expert -> last_used tick
    hit = tot = 0; t = 0
    for e in accesses(seq):
        t += 1; tot += 1
        if e in cache:
            hit += 1
        elif len(cache) >= M:
            victim = min(cache, key=cache.get)
            del cache[victim]
        cache[e] = t
    return hit / tot if tot else 0.0


def sim_lfu(seq, M):
    freq = defaultdict(int); last = {}; t = 0; hit = tot = 0
    cache = set()
    for e in accesses(seq):
        t += 1; tot += 1; freq[e] += 1
        if e in cache:
            hit += 1
        else:
            if len(cache) >= M:
                victim = min(cache, key=lambda x: (freq[x], last[x]))
                cache.discard(victim)
            cache.add(e)
        last[e] = t
    return hit / tot if tot else 0.0


def sim_lru2(seq, M):
    hist = defaultdict(lambda: [0, 0])  # expert -> [prev, last] access ticks
    cache = set(); t = 0; hit = tot = 0
    for e in accesses(seq):
        t += 1; tot += 1
        h = hist[e]; h[0] = h[1]; h[1] = t
        if e in cache:
            hit += 1
        else:
            if len(cache) >= M:
                # evict smallest 2nd-most-recent access (0 = never -> evict first)
                victim = min(cache, key=lambda x: hist[x][0])
                cache.discard(victim)
            cache.add(e)
    return hit / tot if tot else 0.0


def sim_static(seq, M, warmup_frac=0.5):
    half = max(1, int(len(seq) * warmup_frac))
    c = Counter()
    for experts in seq[:half]:
        c.update(experts)
    cached = {e for e, _ in c.most_common(M)}
    hit = tot = 0
    for experts in seq[half:]:
        for e in experts:
            tot += 1
            if e in cached:
                hit += 1
    return hit / tot if tot else 0.0


def sim_belady(seq, M):
    stream = list(accesses(seq))
    # next-use index per position
    future = defaultdict(list)
    for i, e in enumerate(stream):
        future[e].append(i)
    ptr = defaultdict(int)
    cache = set(); hit = 0
    for i, e in enumerate(stream):
        ptr[e] += 1  # advance past current occurrence
        if e in cache:
            hit += 1
        else:
            if len(cache) >= M:
                def next_use(x):
                    idx = ptr[x]
                    occ = future[x]
                    return occ[idx] if idx < len(occ) else float("inf")
                victim = max(cache, key=next_use)
                cache.discard(victim)
            cache.add(e)
    return hit / len(stream) if stream else 0.0


POLICIES = {
    "lru": sim_lru, "lfu": sim_lfu, "lru2": sim_lru2,
    "static": sim_static, "belady": sim_belady,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--budget", type=float, default=0.33)
    args = ap.parse_args()

    for path in args.traces:
        per_layer, n_exp = load_trace(path)
        if not per_layer:
            print(f"{path}: no routing data"); continue
        M = max(1, round(args.budget * n_exp))
        print(f"{path}  (M={M} of {n_exp} experts/layer, budget {args.budget:.0%})")
        results = {}
        for name, fn in POLICIES.items():
            rates = [fn(seq, M) for seq in per_layer.values()]
            results[name] = sum(rates) / len(rates)
        ceil = results["belady"]
        for name in ("static", "lru", "lru2", "lfu", "belady"):
            r = results[name]
            gap = "" if name == "belady" else f"  ({r/ceil*100:4.1f}% of optimal)"
            print(f"    {name:7s} {r:6.1%}{gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
