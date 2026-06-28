#!/usr/bin/env python3
"""Analyze MoE routing traces produced by GGML_MOE_ROUTING_TRACE.

Each trace line is "<layer> <e0> <e1> ... <e_{k-1}>": the experts selected for
one token at one layer, in token order (prefill tokens first, then decode).

The question this answers: would a *dynamic, expert-granularity* VRAM cache beat
the *static, layer-granularity* residency that the stock `-ncmoe` flag already
gives? At a VRAM budget that holds fraction f of the expert pool:

  - static `-ncmoe`: caches whole layers -> hit rate ~= f (f of layers always
    hit, the rest never).
  - dynamic top-M-per-layer (M = f * n_experts): hit rate = fraction of
    activations landing in each layer's most-popular experts. Beats static iff
    activation is skewed.

We report an *oracle* hit rate (top-M chosen with full knowledge) and a *warm*
hit rate (top-M learned from the first half of the trace, measured on the second
half) -- the warm number is what a real history/frequency cache could achieve.
Also reports lag-1 overlap: the share of a token's experts also used by the
previous token, i.e. the ceiling for naive reactive prefetch.
"""
import argparse
from collections import defaultdict, Counter
from pathlib import Path


def load_trace(path: Path):
    """layer -> list of frozenset(experts), in token order."""
    per_layer = defaultdict(list)
    n_experts_seen = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                layer = int(parts[0])
                experts = [int(x) for x in parts[1:]]
            except ValueError:
                continue
            per_layer[layer].append(frozenset(experts))
            if experts:
                n_experts_seen = max(n_experts_seen, max(experts) + 1)
    return per_layer, n_experts_seen


def gini(counts):
    vals = sorted(counts)
    n = len(vals)
    if n == 0 or sum(vals) == 0:
        return 0.0
    cum = 0
    for i, v in enumerate(vals, 1):
        cum += i * v
    return (2 * cum) / (n * sum(vals)) - (n + 1) / n


def hit_rate_topM(seqs, M):
    """Oracle: per layer cache the M most-active experts, measure activation hit."""
    hit = tot = 0
    for seq in seqs.values():
        c = Counter()
        for s in seq:
            c.update(s)
        cached = {e for e, _ in c.most_common(M)}
        for s in seq:
            for e in s:
                tot += 1
                if e in cached:
                    hit += 1
    return hit / tot if tot else 0.0


def hit_rate_warm(seqs, M):
    """Realistic: learn top-M from first half, measure hit on second half."""
    hit = tot = 0
    for seq in seqs.values():
        half = len(seq) // 2
        if half < 1:
            continue
        c = Counter()
        for s in seq[:half]:
            c.update(s)
        cached = {e for e, _ in c.most_common(M)}
        for s in seq[half:]:
            for e in s:
                tot += 1
                if e in cached:
                    hit += 1
    return hit / tot if tot else 0.0


def lag1_overlap(seqs):
    """Mean fraction of a token's experts shared with the previous token."""
    num = den = 0
    for seq in seqs.values():
        for a, b in zip(seq, seq[1:]):
            num += len(a & b)
            den += len(b)
    return num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--budget", type=float, default=0.33,
                    help="VRAM budget as fraction of expert pool (default 0.33)")
    args = ap.parse_args()

    for path in args.traces:
        per_layer, n_exp = load_trace(path)
        if not per_layer:
            print(f"{path}: no routing data")
            continue
        n_layers = len(per_layer)
        tokens = sum(len(v) for v in per_layer.values()) // n_layers
        k = max((len(s) for seq in per_layer.values() for s in seq), default=0)
        M = max(1, round(args.budget * n_exp))

        # skew: pooled activation counts across all layers
        pooled = Counter()
        distinct_per_layer = []
        for seq in per_layer.values():
            c = Counter()
            for s in seq:
                c.update(s)
            pooled.update(c)
            distinct_per_layer.append(len(c))

        print(f"{path}")
        print(f"  layers={n_layers}  experts={n_exp}  top-k={k}  ~tokens/layer={tokens}")
        print(f"  distinct experts used / layer: mean={sum(distinct_per_layer)/n_layers:.1f}"
              f"  min={min(distinct_per_layer)}  max={max(distinct_per_layer)} of {n_exp}")
        print(f"  activation Gini (pooled): {gini(list(pooled.values())):.3f}  (0=uniform, 1=concentrated)")
        print(f"  VRAM budget f={args.budget:.0%} -> cache top-{M} of {n_exp} experts per layer")
        print(f"    static -ncmoe (layer-granularity) hit rate ~= {args.budget:.1%}")
        print(f"    dynamic oracle  top-{M} hit rate         = {hit_rate_topM(per_layer, M):.1%}")
        print(f"    dynamic warm    top-{M} hit rate         = {hit_rate_warm(per_layer, M):.1%}  (learned from 1st half)")
        print(f"  hit-rate curve (oracle):")
        for f in (0.0625, 0.125, 0.25, 0.33, 0.5):
            m = max(1, round(f * n_exp))
            print(f"    f={f:5.1%}  M={m:3d}  hit={hit_rate_topM(per_layer, m):.1%}")
        print(f"  lag-1 expert overlap (reactive-prefetch ceiling): {lag1_overlap(per_layer):.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
