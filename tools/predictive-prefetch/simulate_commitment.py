#!/usr/bin/env python3
"""Causal commitment windows -- Stage 0 go/no-go (simulation on routing traces).

The idea: when the router is confident, lock its decision for N tokens (reuse the
window's first token's experts), so fewer distinct experts fire and the cache hit
rate rises. But forcing reuse is *lossy* -- locked tokens run on experts they
would not have chosen. This simulates the trade on the captured traces:

  - forced hit rate : LRU hit rate when routing is locked into windows of N
  - natural hit rate: LRU hit rate on the real routing (baseline)
  - routing fidelity: mean overlap between the forced and the natural expert set
                      over the locked (non-first-in-window) tokens -- the proxy
                      for how much quality is sacrificed (1.0 = no change).

If fidelity stays high while hit rate climbs, commitment is worth building.
If fidelity collapses, locking damages routing and is not worth it.
"""
import argparse
from collections import defaultdict
from pathlib import Path


def load_trace(path):
    per_layer = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            try:
                layer = int(p[0]); experts = [int(x) for x in p[1:]]
            except ValueError:
                continue
            per_layer[layer].append(experts)
    return per_layer


def lru_hit_rate(seqs, M):
    hit = tot = 0
    for seq in seqs.values():
        cache = {}; t = 0
        for experts in seq:
            t += 1
            for e in experts:
                tot += 1
                if e in cache:
                    hit += 1
                elif len(cache) >= M:
                    del cache[min(cache, key=cache.get)]
                cache[e] = t
    return hit / tot if tot else 0.0


def force_windows(seqs, N):
    """Replace each window's tokens with the window's first token's experts."""
    out = {}
    for layer, seq in seqs.items():
        forced = []
        for i, experts in enumerate(seq):
            forced.append(seq[(i // N) * N])  # the window's first token
        out[layer] = forced
    return out


def fidelity(seqs, forced, N):
    num = den = 0
    for layer, seq in seqs.items():
        f = forced[layer]
        for i in range(len(seq)):
            if i % N == 0:
                continue  # first token in window is unchanged
            nat = set(seq[i]); fc = set(f[i])
            num += len(nat & fc); den += len(nat)
    return num / den if den else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--budget", type=float, default=0.33)
    args = ap.parse_args()

    for path in args.traces:
        seqs = load_trace(path)
        if not seqs:
            print(f"{path}: no data"); continue
        n_exp = max(max(e) for s in seqs.values() for e in s) + 1
        M = max(1, round(args.budget * n_exp))
        base = lru_hit_rate(seqs, M)
        print(f"{path}  (M={M}/{n_exp}, natural LRU hit={base:.1%})")
        print(f"  {'N':>2} {'forced hit':>11} {'fidelity':>9}  (1=lossless lock)")
        for N in (2, 4, 6, 8):
            forced = force_windows(seqs, N)
            hr = lru_hit_rate(forced, M)
            fid = fidelity(seqs, forced, N)
            print(f"  {N:>2} {hr:>10.1%} {fid:>9.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
